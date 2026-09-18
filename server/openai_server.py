#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for the Ternary-Bonsai-2-27B MLX model.

Serves ``POST /v1/chat/completions`` (and ``GET /health``) by loading the
local LM Studio model pack with the official Prism hadamard runtime and
generating with MLX.

Usage:
    uv run openai_server.py
    uv run openai_server.py --model /path/to/Ternary-Bonsai-2-27B-mlx-2bit
    uv run openai_server.py --port 8270 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models as models
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from safetensors import safe_open
from transformers import AutoTokenizer

from runtime.runtime import Packed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = str(
    Path.home() / ".lmstudio" / "models" / "prism-ml" / "Ternary-Bonsai-2-27B-mlx-2bit"
)

DEFAULT_PORT = 8270
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_NEW_TOKENS = 512
MAX_CTX = 4096

# Built from pieces so the literal marker tokens never appear in this source.
_THINK_CLOSE = "</" + "think" + ">"


# ---------------------------------------------------------------------------
# Model loading (official Prism hadamard runtime)
# ---------------------------------------------------------------------------

def _resolve(parent, part: str):
    """Traverse one dotted-path segment (digits index into a ``layers`` list)."""
    return parent[int(part)] if part.isdigit() else getattr(parent, part)


def rebuild_packed(model, modules: List[dict], weights: Dict[str, mx.array]) -> None:
    """Replace the dense modules named in ``modules`` with ternary ``Packed`` modules.

    Mirrors the official Prism ``artifact.load_model``: each record's ``path`` is a
    dotted path to a whole ``nn.Linear`` / ``nn.Embedding`` and the artifact stores
    ``{path}.weight`` (packed u32), ``{path}.scales`` (f16), ``{path}.biases`` (f16,
    per-group dequant bias) and optionally ``{path}.signs`` (f32 Hadamard signs).
    """
    seen = set()
    for record in modules:
        path = record["path"]
        if path in seen:
            raise ValueError(f"duplicate packed module {path!r}")
        seen.add(path)

        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = _resolve(parent, part)

        arrays = [
            weights[f"{path}.weight"],
            weights[f"{path}.scales"],
            weights[f"{path}.biases"],
        ]
        block = int(record["block"])
        signs = weights.get(f"{path}.signs")
        if block and signs is None:
            raise ValueError(f"missing sign vector for {path!r}")

        setattr(parent, parts[-1],
                Packed(arrays, block, signs, record["embedding"], mx.float16))


def _count_packed(model) -> int:
    """Recursively count ``Packed`` submodules (``tree_flatten`` does not recurse
    into nested modules, so walk the tree explicitly)."""
    total = 0
    def walk(node):
        nonlocal total
        for v in vars(node).values():
            if isinstance(v, Packed):
                total += 1
            elif isinstance(v, nn.Module):
                walk(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, nn.Module):
                        walk(item)
    walk(model)
    return total


def load_model(model_dir: str):
    """Load the (text-only) model from an LM Studio pack directory.

    Handles the v2 artifact layout (``schema_version == 2``), whose
    ``model.safetensors`` tensors are namespaced with a ``language_model.``
    prefix and also ship unused ``vision_tower.`` weights.
    Returns ``(model, tokenizer)``.
    """
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)

    schema = config.get("schema_version", 1)
    if schema not in (1, 2):
        raise ValueError(f"unsupported schema_version {schema} (got {schema!r})")

    # The text backbone lives in `text_config` for v2 and at the top level for v1.
    text_config = config.get("text_config", config)
    model_type = text_config.get("model_type")

    # Resolve the concrete MLX class.  The v2 text backbone's ``model_type``
    # (e.g. "qwen3_5_text") is not itself an mlx_lm module name, so default to
    # the known backbone class unless the config names a real mlx module.
    cls_name = model_type if (model_type and hasattr(models, model_type)) else "qwen3_5"
    cls = getattr(models, cls_name)

    # Build the bare text model (NOT the VL wrapper) so we can load weights into it.
    model = cls.TextModel(cls.TextModelArgs.from_dict(text_config))

    # -- Load raw tensors -----------------------------------------------------
    # v2 namespaces the language tensors with 'language_model.' and additionally
    # carries unused 'vision_tower.' weights.  Keep only the language side and
    # strip the prefix so the keys match the bare TextModel.
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")

    PREFIX = "language_model."
    with safe_open(safetensors_path, framework="np") as f:
        keys = list(f.keys())
    wanted = [k for k in keys if k.startswith(PREFIX)]
    with safe_open(safetensors_path, framework="np") as f:
        weights = {k[len(PREFIX):]: mx.array(f.get_tensor(k)) for k in wanted}
    print(f"loaded {len(weights)} language tensors (dropped {len(keys) - len(wanted)})")

    # -- Load the module list -------------------------------------------------
    modules = config.get("modules", [])
    if not modules:
        raise ValueError("config.json has no 'modules' list")

    # -- Replace dense params with Packed and load weights --------------------
    rebuild_packed(model, modules, weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())

    packed_count = _count_packed(model)
    print(f"model ready: {packed_count} packed (ternary) modules")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    gc.collect()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def sample(logits: mx.array, temperature: float, top_p: float,
           top_k: Optional[int]) -> int:
    """Pick a next token id from a (1, V) logits array."""
    if temperature <= 0:
        return int(mx.argmax(logits[0]))

    z = (logits[0] / temperature).astype(mx.float32)
    if top_k is not None and top_k > 0:
        k = min(top_k, z.shape[0])
        keep = mx.topk(z, k).values
        z = mx.where(z >= mx.min(keep), z, -1e30)
    if top_p is not None and 0.0 < top_p < 1.0:
        vals = mx.sort(z, descending=True)
        probs = mx.softmax(vals, axis=0)
        cdf = mx.cumsum(probs)
        cutoff = cdf[mx.argmax(cdf >= top_p)]
        z = mx.where(z >= mx.log(cutoff), z, -1e30)
    probs = mx.softmax(z, axis=0)
    return int(mx.random.categorical(probs))


def generate(model: nn.Module, tokenizer, messages: List[Dict[str, str]],
             max_tokens: int, temperature: float, top_p: float,
             top_k: Optional[int]) -> Dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > MAX_CTX:
        input_ids = input_ids[-MAX_CTX:]

    eos = tokenizer.eos_token_id
    cache = model.make_cache()

    x = mx.array([input_ids])
    logits = model(x, cache=cache)
    ids = list(input_ids)
    finish = "stop"
    for _ in range(max_tokens):
        nxt = sample(logits[:, -1, :], temperature, top_p, top_k)
        ids.append(nxt)
        if nxt == eos:
            finish = "stop"
            break
        logits = model(mx.array([[nxt]]), cache=cache)
    else:
        finish = "length"

    # This is a reasoning model: the chat template pre-fills the open-think
    # token (id 248068) after the assistant turn start, so the model's
    # reasoning runs from the end of the prompt until the close-think token
    # (id 248069), followed by the final answer.  The tokenizer decodes those
    # two tokens asymmetrically, so split at the token-id level: reasoning =
    # generated tokens up to and including the close token, content = the rest.
    tail = ids[len(input_ids):]
    close_id = tokenizer.convert_tokens_to_ids(_THINK_CLOSE)
    close_pos = next((i for i, t in enumerate(tail) if t == close_id), None)
    if close_pos is not None:
        reasoning_ids = tail[:close_pos]   # token right before the close marker
        kept = tail[close_pos + 1:]
    else:
        reasoning_ids = []
        kept = tail

    content = tokenizer.decode(kept, skip_special_tokens=True).strip()
    reasoning = (
        tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip()
        or None
    )
    return {
        "prompt_tokens": len(input_ids),
        "completion_tokens": len(ids) - len(input_ids),
        "content": content,
        "reasoning": reasoning,
        "finish_reason": finish,
    }


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False


class _ChoiceMsg(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class _Choice(BaseModel):
    index: int = 0
    message: _ChoiceMsg
    finish_reason: str


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[_Choice]
    usage: _Usage


class _StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class _StreamChoice(BaseModel):
    index: int = 0
    delta: _StreamDelta
    finish_reason: Optional[str] = None


class _StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[_StreamChoice]


app = FastAPI(title="Ternary-Bonsai OpenAI Server")

state: Dict[str, Any] = {"model": None, "tokenizer": None, "model_name": None}


@app.on_event("startup")
def _startup():
    model_dir = DEFAULT_MODEL_DIR
    print(f"loading model from {model_dir} ...")
    t0 = time.time()
    model, tokenizer = load_model(model_dir)
    state["model"], state["tokenizer"] = model, tokenizer
    state["model_name"] = Path(model_dir).name
    mx.metal.clear_cache()
    print(f"model loaded in {time.time() - t0:.1f}s")


def _not_ready() -> None:
    raise HTTPException(status_code=503, detail="Model is still loading.")


@app.get("/health")
def health():
    ready = state["model"] is not None
    return {
        "status": "ok" if ready else "loading",
        "model": state["model_name"],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if state["model"] is None or state["tokenizer"] is None:
        _not_ready()

    messages = [{"role": m.role, "content": m.content or ""} for m in req.messages]
    max_tokens = (
        req.max_tokens
        or req.max_completion_tokens
        or DEFAULT_MAX_NEW_TOKENS
    )

    if req.stream:
        return _stream(messages, max_tokens, req.temperature, req.top_p, req.top_k, req.model)

    t0 = time.time()
    res = generate(
        state["model"], state["tokenizer"], messages,
        max_tokens, req.temperature, req.top_p, req.top_k,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model or state["model_name"] or "unknown",
        choices=[
            _Choice(
                index=0,
                message=_ChoiceMsg(
                    content=res["content"],
                    reasoning_content=res["reasoning"],
                ),
                finish_reason=res["finish_reason"],
            )
        ],
        usage=_Usage(
            prompt_tokens=res["prompt_tokens"],
            completion_tokens=res["completion_tokens"],
            total_tokens=res["prompt_tokens"] + res["completion_tokens"],
        ),
    )


def _stream(messages, max_tokens, temperature, top_p, top_k, model_name):
    """Non-incremental 'stream': yields the whole completion as a few SSE chunks.

    True per-token streaming would require reworking the generator to yield
    incrementally; this returns the completed text in a stream-shaped response
    so OpenAI `stream=true` clients still work.
    """
    res = generate(
        state["model"], state["tokenizer"], messages,
        max_tokens, temperature, top_p, top_k,
    )
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def sse(chunk):
        payload = _StreamChunk(
            id=cid, created=created,
            model=model_name or state["model_name"] or "unknown",
            choices=[chunk],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    def gen():
        yield sse(_StreamChoice(index=0, delta=_StreamDelta(role="assistant")))
        yield sse(_StreamChoice(
            index=0,
            delta=_StreamDelta(content=res["content"],
                               reasoning_content=res["reasoning"]),
        ))
        yield sse(_StreamChoice(index=0, delta=_StreamDelta(),
                                finish_reason=res["finish_reason"]))
        yield "data: [DONE]\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global DEFAULT_MODEL_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL_DIR,
                    help="path to the model pack directory")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=DEFAULT_HOST)
    args = ap.parse_args()

    DEFAULT_MODEL_DIR = args.model

    print(f"serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()