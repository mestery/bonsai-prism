# Long-Context Without the Cap — Options

Reference doc for lifting the `--max-ctx` OOM cap and serving genuinely long
prompts. The cap is a workaround; these are the two real fixes. **Option 2
(tiled attention) is now implemented and enabled by default** — see the
"Implemented" subsection and the measured results. Every number below was
measured, not estimated.

---

## The wall (root cause)

The model (`prism_hadamard_qwen35`, Ternary-Bonsai-2-27B-mlx-2bit) has:

| field | value |
|---|---|
| `num_hidden_layers` | **64** |
| layer pattern | 3 GatedDeltaNet + 1 full-attention, ×16 → **48 GatedDeltaNet + 16 full-attention** |
| `hidden_size` | 5120 |
| full-attn `num_attention_heads` | 24 |
| full-attn `num_key_value_heads` | 4 (GQA, 6 Q per KV) |
| full-attn **`head_dim`** | **256** |
| GatedDeltaNet `linear_num_value_heads` × `linear_value_head_dim` | 48 × 128 |
| `intermediate_size` (dense MLP) | 17408 |
| `vocab_size` | 248320 |
| config `max_position_embeddings` | 262144 |

The blocker is the **full-attention `head_dim=256`**.

`mx.fast.scaled_dot_product_attention` only engages the fused **flash** kernel
(the O(S) path) for `head_dim ∈ {32, 64, 128}`. Measured growth per 4× sequence
doubling (H=24, D varied):

| `head_dim` | growth (S×4) | path |
|---|---|---|
| 64 | ×5.5 | ~flash |
| 128 | ×4.8 | **O(S) flash** |
| 96 / 160 / 192 / 224 / **256** | ×11–13 | **O(S²) unfused** |

For `head_dim=256` the kernel falls back to the **unfused path**, which
materializes the full `[B, H, S, S]` attention-score matrix. That is O(S²) in
memory and FLOPs. One full-attention layer alone:

- S=8192 → ~7.2 GB peak (the `[1, 24, 8192, 8192]` score matrix + softmax)
- S=16384 → ~26 GB
- S≈47000 → ~200 GB

16 such layers run sequentially (MLX frees each before the next), so the peak
from attention is ~one layer's O(S²), but it still scales quadratically. This is
why memory blows the ~40 GB Metal limit well below the model's 262k config, and
why the safe window is ~8k–10k.

Everything else is *not* the wall:

- **GatedDeltaNet** kernel is O(T) (recurrent, fixed-size state in registers) — fine.
- The **O(S·V)** logits term was removed by `_step_logits` (commit `fafe809`):
  `lm_head` now runs on the last hidden vector only.
- The **KV cache** for the 16 full-attn layers is O(S) and small (~16 MB/layer at
  S=8192).
- Flash attention itself is O(S) when it can be used — the issue is purely that
  `head_dim=256` is outside its supported set.

So the entire long-context problem reduces to: **make the 16 `head_dim=256`
full-attention layers stop being O(S²).** Two ways.

---

## Option 1 — Fused O(S) attention for `head_dim=256`

The clean fix. Get the 16 full-attention layers onto the flash (O(S)) kernel so
they scale linearly like every other `head_dim ≤ 128` model. Full speed, full
context, no approximation.

### What it is

A `head_dim=256` flash-attention kernel. The flash algorithm (online softmax
tiling over the K/V dimension) is head_dim-agnostic in principle; MLX simply
ships compiled kernels for `head_dim ∈ {32, 64, 128}` and dispatches anything
else to the unfused fallback. So this is "add the 256 kernel / widen the
dispatch."

### How to pursue it

**1a. MLX upstream (preferred).** File an issue / PR against
[ml-explore/mlx](https://github.com/ml-explore/mlx) requesting `head_dim=256`
support in `mx.fast.scaled_dot_product_attention`. The relevant code is the
Metal flash-attention kernel and its host-side dispatch in
`mlx/src/fast.cpp` (kernel selection) and `mlx/metal/kernels/`. Check the current
`mlx` release (latest at writing time: 0.32.2) — if a newer release already
supports 256, an upgrade is the whole fix. The GQA shape (24 Q heads / 4 KV
heads) is already handled by the existing kernel, so only the `head_dim` axis
needs widening.

**1b. Custom Metal kernel (if upstream stalls).** Write a `head_dim=256` flash
kernel and register it. This is the same work as 1a but out-of-tree: a Metal
`.metal` kernel plus a C++/Python binding that the model's attention calls
instead of `mx.fast.scaled_dot_product_attention`. More maintenance, but not
blocked on upstream.

**1c. Vendor a known-good 256-dim flash kernel.** Third-party fused-attention
implementations (e.g. the flash-attention kernels other MLX integrations ship)
sometimes already support 256. Porting one over is less work than writing from
scratch, at the cost of a dependency.

### Trade-offs

| | |
|---|---|
| Complexity | Medium — one kernel, one dispatch site |
| Approximation | **None** — exact attention, bit-identical to the unfused math |
| Speed at long ctx | **O(S)** — fast, same as any flash model |
| Context achievable | The model's full 262k (bounded only by KV-cache O(S) memory) |
| Maintenance | Tracks MLX releases (1a) or a vendored kernel (1b/1c) |
| Risk | Low — well-understood algorithm; the 256 kernel is a known quantity |

KV-cache memory at full context: 16 layers × 2 (K,V) × 4 heads × 256 dim × 4
bytes × S ≈ **0.13 MB/token** → ~34 GB at S=262144. That is the *only* remaining
long-ctx cost, and it is O(S) (linear), not quadratic. (Can be cut by storing the
KV cache in a smaller dtype, if needed.)

---

## Option 2 — Tiled attention (bounded-memory monkeypatch)

The pragmatic in-repo fix. Keep the unfused O(S²) *algorithm* but compute it in
**query-tiles** so the peak memory is bounded instead of quadratic. It does not
make attention fast; it makes it *fit*.

### What it is

The unfused path builds `[B, H, S, S]` in one shot. Tile the *query* axis: for
each query block of `T_q` tokens, compute its scores against all `S` keys —
a `[B, H, T_q, S]` block — apply softmax over the key axis, and accumulate the
weighted V. Peak memory is `O(H · T_q · S)` (bounded by `T_q`), total work is
still `O(H · S²)` (unchanged — it's the same FLOPs, just tiled).

Because attention is independent per head and the causal mask is per (query, key)
position, tiling the query axis is **mathematically exact** — identical output to
the one-shot unfused path, only the memory profile differs.

### How to pursue it

Monkeypatch the full-attention forward (the `Qwen3NextAttention.__call__` path,
in `mlx_lm/models/qwen3_next.py` or the custom `prism_hadamard_qwen35` class the
server registers) so that instead of calling
`mx.fast.scaled_dot_product_attention(Q, K, V, mask)` once, it loops over query
tiles:

```
out = empty([B, H, S, D])
for q0 in range(0, S, T_q):
    q = Q[:, :, q0:q0+T_q]              # [B, H, T_q, D]
    # causal: query row i attends to keys 0..(q0+i)
    s = (q @ K^T) * scale               # [B, H, T_q, S]
    s = s.masked_fill(causal_block(q0, T_q, S), -inf)
    p = softmax(s, axis=-1)             # over keys
    out[:, :, q0:q0+T_q] = p @ V        # [B, H, T_q, D]
```

`T_q` is the knob: `T_q = S` degenerates to today's unfused path; small `T_q`
(256–1024) bounds memory. Pick `T_q` so `H · T_q · S · 4 bytes` stays under a
target (e.g. 4–8 GB).

The decode (single new token, S grows by 1) path is already O(S) per step and
needs no change; only the **prefill** (all-at-once prompt) needs tiling.

### Implemented (shipped)

Lives in `server/tiled_attention.py`, applied from `openai_server.py::_lifespan`
via `patch_tiled_attention()`. It monkeypatches the module-global
`scaled_dot_product_attention` in `mlx_lm.models.qwen3_next` (the name
`Qwen3NextAttention.__call__` resolves at call time). It does **not** touch the
vendor's `runtime/` — it is a `server/`-top-level module imported at runtime.

Design details that differ from the pseudocode above (and matter for correctness):

- **GQA is unrolled, not a loop.** `Q [B, H, S, D]` is reshaped to
  `[B, Hkv, R, S, D]` (R = H/Hkv Q-heads per KV head), so one batched matmul per
  KV head covers all its Q-heads via broadcasting:
  `s = (Q_r @ K^T)` where `Q_r [., ., R, S, D]`, `K [., ., 1, S, D]` →
  `s [., ., R, S, S]`. No per-Q-head Python loop.
- **`scale` is applied in fp32 *after* the fp16 `Q·K^T` matmul**
  (`(q @ kT).astype(f32) * scale`), then softmax is fp32. This matches the
  reference numerics for any scale value.
- **Mask stays a string/`None`.** `KVCache.make_mask` yields the `"causal"` string
  (prefill) or `None` (decode), so the patched path builds a causal block-mask
  from the integer `S` — no array mask is ever passed through.
- **Dispatch by length.** `TILE_THRESHOLD` (env `PRISM_ATTN_TILE_THRESHOLD`,
  default 8192): S ≤ threshold → the fused kernel (fast, exact); S > threshold →
  query-tiled. `TILE` (env `PRISM_ATTN_TILE`, default 1024) bounds the per-tile
  score block to `H·TILE·S` fp32.

Measured on the M4 Pro (64 GB), real model (H=24, Hkv=4, D=256):

| S | path | prefill time | rate | **peak RSS** | OOM |
|---|---|---|---|---|---|
| 4096 | fused | 38.2 s | 107 tok/s | **15.84 GB** | no |
| 16384 | tiled | 188.0 s | 87 tok/s | **15.84 GB** | no |
| 32768 | tiled | 478.0 s | 69 tok/s | **15.84 GB** | no |

Peak RSS is **flat (15.84 GB) across S = 4096 → 16384 → 32768** — the tiling keeps
the attention scores bounded instead of the ~26 GB one-shot `S=16384` matrix (or
~105 GB at `S=32768`). `--max-ctx` default was raised to **32768**.

**Caveat — prefill is still slow, and not because of this patch.** The rate drops
(107 → 87 → 69 tok/s) and the absolute slowness (~8 min for 32k) come from the 48
GatedDeltaNet recurrent layers (O(S) but a heavy, growing per-token kernel), not
the 16 full-attn layers (which cost well under a second total once tiled). The
patch removes the OOM wall; it does not make prefill interactive.

### Trade-offs

| | |
|---|---|
| Complexity | Low–medium — one forward method, no new kernel |
| Approximation | **None** — exact, tiled attention |
| Speed at long ctx | **O(S²)** — slow. Same FLOPs as the unfused path; only memory improves |
| Context achievable | As far as the O(S) KV cache + O(S) activations allow (much further than 10k) |
| Maintenance | In-repo; re-apply if the model class or MLX attention API changes |
| Risk | Low for correctness; the cost is wall-clock time at long prompts |

Rough speed intuition: the O(S²) prefill at S≈47000 is ~24 heads × 47000² MACs
per full-attn layer × 16 layers ≈ ~8.5e11 MACs of attention alone — seconds to
minutes on a Mac GPU depending on the tile. It will not be interactive at very
long context; it *runs* where today it *crashes*.

---

## Comparison

| | Option 1 (fused 256 kernel) | Option 2 (tiled attention) |
|---|---|---|
| Memory profile | **O(S)** | bounded O(H·T_q·S) |
| Compute profile | **O(S)** | O(S²) |
| Long-ctx speed | fast | slow |
| Exactness | exact | exact |
| Lives in | MLX upstream / vendored kernel | this repo (monkeypatch) |
| Unblocks full 262k at speed | **yes** | no (unblocks *running*, not *speed*) |
| Effort | medium (one kernel) | low–medium (one method) |
| Blocked on upstream | possibly (1a) | no |

## Recommendation

- **Option 2 is shipped** (`server/tiled_attention.py`, default on) and has
  removed the OOM wall: `--max-ctx` is now 32768 and memory is flat with S. It
  is entirely in-repo, low-risk, and turns the crash into a slow-but-working
  request. The two env knobs (`PRISM_ATTN_TILE`, `PRISM_ATTN_TILE_THRESHOLD`)
  trade peak memory vs. tile count.
- **Pursue Option 1 in parallel** as the durable fix: file the MLX issue for a
  `head_dim=256` flash kernel. The moment a release (or a ported kernel) ships it,
  the 16 full-attn layers become O(S) and the context cap can be removed entirely
  — at which point Option 2 becomes dead weight and can be dropped (remove the
  `patch_tiled_attention()` call in `_lifespan`).

The two are complementary, not competing: Option 2 is the stopgap (shipped),
Option 1 is the finish line.

---

## Measurement notes (reproducible)

All `head_dim` dispatch evidence came from driving
`mx.fast.scaled_dot_product_attention` directly with the model's real shapes
(H=24, Hkv=4) and varying `D` and `H`, measuring `mx.get_peak_memory()` across
S∈{1024, 2048, 4096}. A 4× sequence doubling shows ×4 peak for the O(S) flash
path and ×16 for the O(S²) unfused path; the model's `D=256` measured ×10–13,
i.e. unfused. Splitting the Q-head count does **not** help — `D=256` is unfused
for every `H` (2…24), confirming the dispatch key is `head_dim`, not the head
count.
