# ternary-bonsai-2-27b-mlx loader

A local, OpenAI-compatible server for the **Ternary-Bonsai-2-27B** MLX model — a
2-bit ternary (group-128 Hadamard) quantization of Qwen3.5-27B, text-only.

The server loads a local LM Studio model pack with the official Prism hadamard
runtime and serves `POST /v1/chat/completions` and `GET /health` over HTTP.

## Model

The model pack lives at:

```
~/.lmstudio/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/
```

## Server

See [`server/`](server/README.md) for full details. In short:

```bash
cd server
uv sync                                                        # installs the pinned deps into server/.venv
uv run openai_server.py --model /path/to/Ternary-Bonsai-2-27B-mlx-2bit
```

Then:

```bash
curl -s -X POST http://127.0.0.1:8270/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In one sentence, what is a CPU?"}],"max_tokens":128}'
```

The model is a reasoning model, so responses carry both the final answer
(`content`) and the reasoning trace (`reasoning_content`).

## Dependencies

Installed via [`uv`](https://docs.astral.sh/uv/) from `server/pyproject.toml`
(pinned: `mlx==0.32.0`, `mlx-lm==0.31.3`, `transformers==5.5.0`, plus the web
stack). Python 3.12.
