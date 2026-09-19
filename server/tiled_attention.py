"""Query-tiled attention monkeypatch for long-context prefill on non-NAX Apple Silicon.

Background
----------
The full-attention layers of this model use ``head_dim = 256``. MLX's fused
``mx.fast.scaled_dot_product_attention`` kernel ships ``head_dim``
specializations for {32, 64, 128} plus a newer *fused-256* path that is gated
behind ``is_nax_available()`` -- which requires macOS >= 26.2 *and* a
generation-17+ GPU (NAX). On an M4 Pro (generation 16) that gate is ``False``,
so a 256-head-dim prompt falls back to the unfused O(S^2) attention path: the
``[B, H, S, S]`` scores tensor is materialized and memory grows quadratically
with the prompt. Prefill OOMs above ~8k-10k tokens.

Fix
---
We monkeypatch the ``scaled_dot_product_attention`` wrapper that the full-
attention layers call (the ``mlx_lm.models.qwen3_next`` module) so that, once a
prefill is longer than a threshold, the query axis is computed in tiles. Each
query tile attends to the *full* key/value set, so peak memory is bounded by
``O(H_kv * n_repeats * TILE * S)`` instead of ``O(H * S^2)``. The math is exact
(per-tile causal masking, ``scale`` applied in fp32, softmax in fp32); decode
(a single query) and short prefills keep using the fast fused kernel unchanged.

The GatedDeltaNet (linear-attention) layers are O(T) recurrent and never
materialize an S^2 tensor, so they are left untouched.

Tunables (environment variables)
--------------------------------
``PRISM_ATTN_TILE``            query tile size, default ``1024``
``PRISM_ATTN_TILE_THRESHOLD``  prefill length at/above which tiling kicks in,
                               default ``8192``
"""

from __future__ import annotations

import os

import mlx.core as mx

# Query tile size (tokens). Bounds the per-tile [B, H_kv, n_repeats, TILE, S]
# scores tensor. Lower it to trade a little throughput for longer contexts.
TILE = int(os.environ.get("PRISM_ATTN_TILE", "1024"))

# Prefill lengths at or below this keep the fast fused kernel (it is memory-safe
# for these on ~64 GB); longer prefills are tiled. On a 64 GB machine the fused
# 256-head-dim kernel OOMs above ~8k-10k tokens, so 8192 is the safe cutover.
TILE_THRESHOLD = int(os.environ.get("PRISM_ATTN_TILE_THRESHOLD", "8192"))

_PATCHED = False
_ORIG = None


def _tiled_sdp(queries, keys, values, scale, mask):
    """Query-tiled SDPA for the non-quantized (KVCache) path.

    ``queries``: ``[B, H, L, D]``  (H query heads)
    ``keys`` / ``values``: ``[B, Hkv, S, D]``  (Hkv KV heads, S >= L)
    Returns ``[B, H, L, D]``.

    GQA is handled by unrolling the query heads into ``[B, Hkv, R, L, D]`` with
    ``R = H // Hkv`` and broadcasting the (single) KV head over the ``R`` query
    heads -- the same convention ``mx.fast.scaled_dot_product_attention`` uses.
    ``scale`` is applied in fp32 and the softmax is computed in fp32 for
    numerical stability (the scores are O(1) after scaling but ``exp`` overflows
    fp16, so fp32 is required).
    """
    B, H, L, D = queries.shape
    Hkv = keys.shape[1]
    S = keys.shape[2]
    R = H // Hkv  # grouped-query attention: query heads per KV head

    # Unroll GQA so each query head is explicit: [B, Hkv, R, L, D].
    q = queries.reshape(B, Hkv, R, L, D)
    kT = keys.transpose(0, 1, 3, 2)[:, :, None, :, :]  # [B, Hkv, 1, D, S]
    v = values[:, :, None, :, :]                       # [B, Hkv, 1, S, D]
    finfo_min = mx.finfo(mx.float32).min

    out = mx.zeros((B, Hkv, R, L, D), dtype=values.dtype)

    for i in range(0, L, TILE):
        Li = min(TILE, L - i)
        qi = q[:, :, :, i:i + Li, :]                          # [B,Hkv,R,Li,D]
        scores = (qi @ kT).astype(mx.float32) * scale         # [B,Hkv,R,Li,S]

        if mask is not None and mask == "causal":
            qpos = mx.arange(i, i + Li)[:, None]              # [Li, 1]
            kpos = mx.arange(S)[None, :]                     # [1, S]
            m = (kpos <= qpos)[None, None, None, :, :]       # [1,1,1,Li,S]
            scores = mx.where(m, scores, finfo_min)

        probs = mx.softmax(scores, axis=-1)                  # fp32, stable
        out[:, :, :, i:i + Li, :] = (probs @ v).astype(values.dtype)

    return out.reshape(B, H, L, D)


def _patched_sdp(queries, keys, values, cache, scale, mask, sinks=None):
    """Replacement for ``mlx_lm.models.base.scaled_dot_product_attention``.

    - Quantized KV caches (``cache.bits``) are delegated to the original wrapper.
    - Short prefills and decode (``L <= TILE_THRESHOLD``) use the fused kernel.
    - Long prefills (``L > TILE_THRESHOLD``) use query-tiled attention.
    """
    # Quantized cache: leave it to the original (this model uses the plain
    # KVCache, so this branch is not taken, but keep it correct regardless).
    if cache is not None and hasattr(cache, "bits"):
        return _ORIG(queries, keys, values, cache=cache, scale=scale,
                     mask=mask, sinks=sinks)

    L = queries.shape[2]
    if L <= TILE_THRESHOLD:
        # Fast fused kernel: handles GQA + the "causal" string mask.
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks
        )
    return _tiled_sdp(queries, keys, values, scale, mask)


def patch_tiled_attention(verbose: bool = True) -> bool:
    """Monkeypatch the full-attention SDPA wrapper to use query tiling.

    Idempotent: only the first call rebinding the function. Returns ``True`` if
    the patch was applied now, ``False`` if it was already applied.
    """
    global _PATCHED, _ORIG
    if _PATCHED:
        return False
    import mlx_lm.models.qwen3_next as q3n
    _ORIG = q3n.scaled_dot_product_attention
    q3n.scaled_dot_product_attention = _patched_sdp
    _PATCHED = True
    if verbose:
        print(f"[tiled-attention] patch active "
              f"(tile={TILE}, threshold={TILE_THRESHOLD})")
    return True