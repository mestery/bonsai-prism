"""Hadamard transform dedup for shared-input Packed module groups.

In the Ternary-Bonsai-2 model, multiple Packed modules receive the same
activation tensor (e.g. GDN in_proj_qkv and in_proj_z both take the
layernorm output). Because the sign vectors are per-width (not per-module),
the fwht is bit-identical for all modules with the same input width.

This patch hoists the shared transform out of the individual module calls,
computing it once per group. Expected decode savings: ~15 ms/token
(36% of transform time, ~19% of total dense time).

Redundancy groups:
  - GDN:     in_proj_qkv + in_proj_z   (2x -> 1x)
  - Attn:    q_proj + k_proj + v_proj  (3x -> 1x)
  - MLP:     gate_proj + up_proj       (2x -> 1x)

The patch is idempotent and safe to call multiple times.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.gated_delta import gated_delta_update

from .runtime import Packed

_applied = False


def _packed_block(m) -> bool:
    """True if ``m`` is a Packed module with a non-zero Hadamard block."""
    return isinstance(m, Packed) and m.block != 0


def _patch_gdn(gdn_cls):
    """Patch GatedDeltaNet.__call__: share fwht between in_proj_qkv / in_proj_z."""

    def patched(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv_p = _packed_block(self.in_proj_qkv)
        z_p = _packed_block(self.in_proj_z)

        if qkv_p or z_p:
            x_t = self.in_proj_qkv.transform(inputs) if qkv_p else inputs
            qkv = self.in_proj_qkv.gemm(x_t) if qkv_p else self.in_proj_qkv(inputs)
            z = self.in_proj_z.gemm(x_t) if z_p else self.in_proj_z(inputs)
        else:
            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs)

        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    gdn_cls.__call__ = patched


def _patch_attn(attn_cls):
    """Patch Qwen3NextAttention.__call__: share fwht between q/k/v projections."""

    def patched(self, x, mask=None, cache=None):
        B, L, D = x.shape

        q_p = _packed_block(self.q_proj)
        k_p = _packed_block(self.k_proj)
        v_p = _packed_block(self.v_proj)

        if q_p or k_p or v_p:
            x_t = self.q_proj.transform(x) if q_p else x
            q_proj_output = self.q_proj.gemm(x_t) if q_p else self.q_proj(x)
            keys = self.k_proj.gemm(x_t) if k_p else self.k_proj(x)
            values = self.v_proj.gemm(x_t) if v_p else self.v_proj(x)
        else:
            q_proj_output = self.q_proj(x)
            keys, values = self.k_proj(x), self.v_proj(x)

        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))

    attn_cls.__call__ = patched


def _patch_mlp(mlp_cls):
    """Patch Qwen3NextMLP.__call__: share fwht between gate_proj and up_proj."""

    def patched(self, x):
        g_p = _packed_block(self.gate_proj)
        u_p = _packed_block(self.up_proj)

        if g_p or u_p:
            x_t = self.gate_proj.transform(x) if g_p else x
            g = self.gate_proj.gemm(x_t) if g_p else self.gate_proj(x)
            u = self.up_proj.gemm(x_t) if u_p else self.up_proj(x)
        else:
            g = self.gate_proj(x)
            u = self.up_proj(x)

        return self.down_proj(swiglu(g, u))

    mlp_cls.__call__ = patched


def apply():
    """Apply all transform-dedup patches. Idempotent."""
    global _applied
    if _applied:
        return
    import mlx_lm.models.qwen3_5 as qwen3_5
    import mlx_lm.models.qwen3_next as qwen3_next

    _patch_gdn(qwen3_5.GatedDeltaNet)
    _patch_attn(qwen3_next.Qwen3NextAttention)
    _patch_mlp(qwen3_next.Qwen3NextMLP)
    _applied = True
    print("transform dedup patch applied")
