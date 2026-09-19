#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for the Ternary-Bonsai-2-27B MLX model.

Serves ``POST /v1/chat/completions`` (and ``GET /health``) by loading the
local LM Studio model pack with the official Prism hadamard runtime and
generating with MLX.

Usage:
    uv run openai_server.py --model /path/to/Ternary-Bonsai-2-27B-mlx-2bit
    uv run openai_server.py --model /path/to/... --port 8270 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    """Recursively count ``Packed`` submodules.

    MLX ``nn.Module`` is a ``Mapping``: its parameters and submodules are stored
    in the module's mapping (accessible via ``dict(node)`` / ``node.keys()``),
    NOT in ``__dict__`` (``vars(node)`` only yields ``_no_grad``/``_training``).
    Walk the mapping so nested modules are reached."""
    total = 0
    def walk(node):
        nonlocal total
        for v in dict(node).values():
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
# Tool-call parsing
# ---------------------------------------------------------------------------

# Tag markers are built from pieces so the two-char open/close markers never
# appear as literals in this source (some tooling mis-parses them).
_TC_OPEN = "<tool_call>"
_TC_CLOSE = "\u003c/" + "tool_call" + "\u003e"
_FN_CLOSE = "\u003c/" + "function" + "\u003e"
_PAR_CLOSE = "\u003c/" + "parameter" + "\u003e"

_TOOL_CALL_RE = re.compile(
    "<tool_call>\\s*<function=([^>]+)>\\s*"
    f"((?:<parameter=[^>]+>\\s*\\n.*?\\n{_PAR_CLOSE}\\s*)*)"
    f"{_FN_CLOSE}\\s*{_TC_CLOSE}",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    rf"<parameter=([^>]+)>\s*\n(.*?)\n</parameter>",
    re.DOTALL,
)

# Token IDs for the opening tag of a tool call. The model emits the
# tag as exactly four tokens: '<', 'tool', '_call', '>'. We detect the
# start of the tool-call zone by scanning the content token stream for
# this run, which lets us suppress raw tool-call text from streaming
# deltas as soon as it begins (before the full block is parseable).
#
# IMPORTANT: these IDs are for this model's tokenizer (Ternary-Bonsai-2
# byte-level BPE). If the tokenizer changes, re-derive them:
#   tok = tokenizer; [tok.convert_tokens_to_ids(t) for t in ('<','tool','_call','>')]
_LT_TOK_ID = 27
_TOOL_TOK_ID = 13766
_CALL_TOK_ID = 13042
_GT_TOK_ID = 29
_SLASH_TOK_ID = 510          # '</'
_FUNC_TOK_ID = 1628          # 'function'
# Opening tag: '<','tool','_call','>'
_TC_OPEN_TOKS = (_LT_TOK_ID, _TOOL_TOK_ID, _CALL_TOK_ID, _GT_TOK_ID)
# Closing tag: '</','tool','_call','>'  (marks end of one tool-call block)
_TC_CLOSE_TOKS = (_SLASH_TOK_ID, _TOOL_TOK_ID, _CALL_TOK_ID, _GT_TOK_ID)


def _parse_tool_calls(text: str) -> Tuple[Optional[List[Dict]], str]:
    """Extract structured tool calls from the model's raw output.

    Returns ``(tool_calls, residual)`` where ``tool_calls`` is a list of
    ``{"id", "type", "function": {"name", "arguments"}}`` dicts (or ``None``
    if none were found) and ``residual`` is the text with the tool-call
    blocks removed (stripped).
    """
    if not text:
        return None, ""

    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return None, text

    tool_calls: List[Dict] = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        args_block = m.group(2) or ""
        params: Dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(args_block):
            key = pm.group(1)
            val = pm.group(2).strip()
            try:
                params[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                params[key] = val
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(params)},
        })

    residual = _TOOL_CALL_RE.sub("", text).strip()
    return tool_calls, residual


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


def _content_before_tool_call(text: str) -> str:
    """Text up to (excluding) the first tool-call open tag, or the whole text
    if no tool call has started. Keeps raw tool-call XML out of streaming
    content deltas."""
    i = text.find(_TC_OPEN)
    return text[:i] if i != -1 else text


def _step_logits(model: nn.Module, x: mx.array, cache) -> mx.array:
    """Run one forward pass and return ONLY the final position's logits
    (shape (1, V)).

    Calling ``model(x, cache)`` applies ``lm_head`` to every prompt position,
    producing a ``(1, seq_len, V)`` tensor. For a long prompt (e.g. ~47k
    tokens, V=248320) that is ~43 GB and exceeds the Metal buffer limit, even
    though we only need the last row. Splitting the step (as
    ``runtime/artifact.py`` does): run the transformer for the hidden states,
    then project just the last hidden vector through ``lm_head``.
    """
    hidden = model.model(x, cache=cache)
    h = hidden[:, -1, :]
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def generate_stream(model: nn.Module, tokenizer, messages: List[Dict[str, str]],
                    max_tokens: int, temperature: float, top_p: float,
                    top_k: Optional[int], tools: Optional[List[Dict]] = None):
    """Incremental completion generator for a reasoning model.

    Yields one dict per sampled token carrying ``reasoning_delta`` /
    ``content_delta``, then a final dict with ``reasoning`` / ``content`` /
    ``tool_calls`` / ``finish_reason`` plus usage.

    Raw tool-call XML is withheld from content deltas: as soon as the open
    tool-call tag appears in the decoded content, only the text before it is
    streamed. Tool calls are parsed from the full final content and reported
    in the final dict.
    """
    # Inject tool definitions into the rendered prompt so the model knows they
    # are available; without this it never emits a tool call.
    tkwargs = {"tools": tools} if tools else {}
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **tkwargs
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > MAX_CTX:
        input_ids = input_ids[-MAX_CTX:]

    eos = tokenizer.eos_token_id
    cache = model.make_cache()

    x = mx.array([input_ids])
    logits = _step_logits(model, x, cache)
    ids = list(input_ids)
    finish = "stop"

    close_id = tokenizer.convert_tokens_to_ids(_THINK_CLOSE)
    in_reasoning = True
    reasoning_buf: List[int] = []
    content_buf: List[int] = []
    reasoning_text = ""
    # Full raw content (incl. tool calls). Doubles as the raw prefix already
    # streamed, which is the marker for computing content deltas.
    raw_content_text = ""

    for _ in range(max_tokens):
        nxt = sample(logits, temperature, top_p, top_k)
        ids.append(nxt)

        if in_reasoning and nxt == close_id:
            in_reasoning = False      # separator token dropped
        elif in_reasoning:
            reasoning_buf.append(nxt)
        else:
            content_buf.append(nxt)

        rt = tokenizer.decode(reasoning_buf, skip_special_tokens=True)
        raw_ct = tokenizer.decode(content_buf, skip_special_tokens=True)
        safe_ct = _content_before_tool_call(raw_ct)
        rd = rt[len(reasoning_text):] if rt.startswith(reasoning_text) else rt
        # Emit only the part of the safe (pre-tool-call) text that comes after
        # what has already been streamed. The marker is the FULL raw prefix,
        # not the safe prefix: while the open tool-call tag is still assembling
        # token-by-token, safe_ct transiently SHRINKS to a strict prefix of the
        # raw prefix already streamed, and the negative slice then yields ""
        # instead of re-emitting the preamble.
        if raw_ct.startswith(raw_content_text):
            cd = safe_ct[len(raw_content_text):]
        else:
            cd = safe_ct
        reasoning_text, raw_content_text = rt, raw_ct
        yield {"reasoning_delta": rd, "content_delta": cd}

        if nxt == eos:
            finish = "stop"
            break
        logits = _step_logits(model, mx.array([[nxt]]), cache)
    else:
        finish = "length"

    raw_content = raw_content_text.strip()
    if tools:
        tool_calls, residual = _parse_tool_calls(raw_content)
    else:
        tool_calls, residual = None, raw_content
    shown = residual if tool_calls else raw_content
    yield {
        "reasoning": reasoning_text.strip() or None,
        "content": shown.strip(),
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else finish,
        "prompt_tokens": len(input_ids),
        "completion_tokens": len(ids) - len(input_ids),
    }


def generate(model: nn.Module, tokenizer, messages: List[Dict[str, str]],
             max_tokens: int, temperature: float, top_p: float,
             top_k: Optional[int], tools: Optional[List[Dict]] = None
             ) -> Dict[str, Any]:
    """Non-streaming completion: consume :func:`generate_stream` to completion."""
    final: Dict[str, Any] = {}
    for item in generate_stream(
        model, tokenizer, messages, max_tokens, temperature, top_p, top_k,
        tools=tools,
    ):
        if "reasoning_delta" not in item:
            final = item
    return final


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None


class _ChoiceMsg(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


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
    tool_calls: Optional[List[Dict]] = None


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


def _build_messages(msgs: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert request messages to the dict form the chat template expects.

    Assistant ``tool_calls`` carry ``arguments`` as a JSON string (as produced
    by the parser). The Qwen template iterates ``arguments`` with ``|items``,
    which requires a mapping, so decode it back to a dict here -- otherwise a
    multi-turn request that echoes a prior tool call fails to render.
    """
    out: List[Dict[str, Any]] = []
    for m in msgs:
        d: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            tcs = []
            for tc in m.tool_calls:
                fn = dict(tc.get("function", {}))
                raw = fn.get("arguments")
                if isinstance(raw, str):
                    try:
                        fn["arguments"] = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        fn["arguments"] = {"_raw": raw}
                elif raw is None:
                    fn["arguments"] = {}
                tcs.append({
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": fn,
                })
            d["tool_calls"] = tcs
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


state: Dict[str, Any] = {
    "model_dir": None, "model": None, "tokenizer": None, "model_name": None
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    model_dir = state["model_dir"]
    print(f"loading model from {model_dir} ...")
    t0 = time.time()
    model, tokenizer = load_model(model_dir)
    state["model"], state["tokenizer"] = model, tokenizer
    state["model_name"] = Path(model_dir).name
    mx.clear_cache()
    print(f"model loaded in {time.time() - t0:.1f}s")
    yield
    mx.clear_cache()


app = FastAPI(title="Ternary-Bonsai OpenAI Server", lifespan=_lifespan)


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

    messages = _build_messages(req.messages)
    max_tokens = (
        req.max_tokens
        or req.max_completion_tokens
        or DEFAULT_MAX_NEW_TOKENS
    )
    tools = req.tools

    if req.stream:
        return _stream(messages, max_tokens, req.temperature, req.top_p, req.top_k,
                       req.model, tools=tools)

    t0 = time.time()
    res = generate(
        state["model"], state["tokenizer"], messages,
        max_tokens, req.temperature, req.top_p, req.top_k,
        tools=tools,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model or state["model_name"] or "unknown",
        choices=[
            _Choice(
                index=0,
                message=_ChoiceMsg(
                    content=res["content"] or None,
                    reasoning_content=res["reasoning"],
                    tool_calls=res["tool_calls"],
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


def _stream(messages, max_tokens, temperature, top_p, top_k, model_name,
            tools: Optional[List[Dict]] = None):
    """Incremental SSE stream: one chunk per sampled token (plus the initial
    role chunk and a final finish chunk), so reasoning and content deltas reach
    the client as they are produced rather than in a single end-of-generation
    burst."""
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    mname = model_name or state["model_name"] or "unknown"

    def sse(delta=None, finish=None):
        payload = _StreamChunk(
            id=cid, created=created, model=mname,
            choices=[_StreamChoice(
                index=0,
                delta=delta if delta is not None else _StreamDelta(),
                finish_reason=finish,
            )],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    def gen():
        yield sse(delta=_StreamDelta(role="assistant"))
        for item in generate_stream(
            state["model"], state["tokenizer"], messages,
            max_tokens, temperature, top_p, top_k,
            tools=tools,
        ):
            if "reasoning_delta" in item:
                rd, cd = item["reasoning_delta"], item["content_delta"]
                if rd or cd:
                    yield sse(_StreamDelta(
                        content=cd or None,
                        reasoning_content=rd or None,
                    ))
            else:
                # Final chunk: include parsed tool_calls if present.
                tcs = item.get("tool_calls")
                yield sse(
                    delta=_StreamDelta(tool_calls=tcs) if tcs else None,
                    finish=item["finish_reason"],
                )
        yield "data: [DONE]\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global MAX_CTX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="path to the model pack directory")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX,
                    help="max prompt context in tokens; longer prompts are "
                         "front-truncated (default %(default)s)")
    args = ap.parse_args()

    state["model_dir"] = args.model
    MAX_CTX = args.max_ctx

    print(f"serving on http://{args.host}:{args.port} (max_ctx={MAX_CTX})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()