# Ternary-Bonsai OpenAI-compatible server

A minimal OpenAI-compatible HTTP server for the **Ternary-Bonsai-2-27B** MLX model
(2-bit ternary / group-128 Hadamard quantization of Qwen3.5-27B, text-only).

It loads a local LM Studio model pack with the official Prism hadamard runtime
(vendored under [`runtime/`](runtime/)) and generates with MLX, exposing
`POST /v1/chat/completions` and `GET /health`.

## Requirements

- Python 3.10+ (this project pins `3.12` via [`.python-version`](.python-version)).
- [`uv`](https://docs.astral.sh/uv/) to install dependencies and run.
- **Forked MLX** — the standard PyPI `mlx` release does not support
  `sdpa_head_dim=256`, which the Prism hadamard runtime requires. Build and
  install from the fork before running `uv sync`:

  ```bash
  git clone https://github.com/mestery/mlx.git
  cd mlx
  git checkout 8ef88c055   # feat/sdpa-head-dim-256
  pip install -e .
  ```

  (Requires a C++ compiler and CMake. On macOS, `brew install cmake` is enough.)

  After this step, `import mlx.core` resolves to your local build. The `uv sync`
  below will see the already-installed fork and skip the PyPI pin.

- The model pack, present at the default location:

  ```
  ~/.lmstudio/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/
  ```

  containing `config.json`, `model.safetensors`, and the Hugging Face tokenizer
  files (`tokenizer.json`, `tokenizer_config.json`, ...).

## Install

1. Build and install the forked MLX (see [Requirements](#requirements) above).
2. Sync the rest of the runtime:

```bash
cd server
uv sync
```

This creates a `.venv/` with the exact pinned runtime
(`mlx-lm==0.31.3`, `transformers==5.5.0`, ...) and the web stack.

## Run

The model pack path is a required option:

```bash
cd server
uv run openai_server.py --model /path/to/Ternary-Bonsai-2-27B-mlx-2bit
```

Optional flags:

```bash
uv run openai_server.py --model /path/to/... --port 8270 --host 127.0.0.1
```

The server binds to `127.0.0.1:8270` by default and reports progress on stdout:

```
serving on http://127.0.0.1:8270
loading model from /path/to/Ternary-Bonsai-2-27B-mlx-2bit ...
model loaded in 3.0s
```

## API

### `GET /health`

```bash
curl http://127.0.0.1:8270/health
```

```json
{"status": "ok", "model": "Ternary-Bonsai-2-27B-mlx-2bit"}
```

### `POST /v1/chat/completions`

Standard OpenAI chat-completions request body. Supported fields: `messages`,
`temperature`, `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `stream`.

Non-streaming:

```bash
curl -s -X POST http://127.0.0.1:8270/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In one sentence, what is a CPU?"}],"max_tokens":128}'
```

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "created": 1789756496,
  "model": "Ternary-Bonsai-2-27B-mlx-2bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A CPU (Central Processing Unit) is the main electronic component …",
        "reasoning_content": "We need to answer user's request: … Keep concise."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 61, "completion_tokens": 55, "total_tokens": 116}
}
```

Streaming (`"stream": true`) returns the same content/reasoning split as a
standard SSE `chat.completion.chunk` stream, ending with `data: [DONE]`.

### Reasoning split

This is a reasoning model: the chat template pre-fills the open-think token after
the assistant turn start, so the model first emits its reasoning and then the
final answer. The server splits on the close-think token at the **token-id**
level (the tokenizer decodes the two markers asymmetrically, so regexing decoded
text is unreliable). The reasoning text is returned in `reasoning_content` and the
final answer in `content`.

## Layout

```
server/
├── openai_server.py     # the HTTP server (this README's subject)
├── pyproject.toml       # uv project + pinned deps
├── uv.lock              # lockfile for reproducible installs
├── .python-version      # 3.12
└── runtime/             # official Prism hadamard runtime (vendored verbatim)
    ├── __init__.py      # sys.path shim for the runtime's flat imports
    ├── runtime.py       # Packed ternary module + fwht
    ├── artifact.py      # official model-pack loader (reference)
    ├── codec.py         # ternary transcode
    ├── vision_artifact.py
    ├── requirements.txt # the runtime's own pinned deps
    └── LICENSE
```
