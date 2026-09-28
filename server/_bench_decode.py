"""Decode / prefill / per-layer benchmark for the 2-bit Bonsai-2 model.

Run from the server venv (CLEAN GPU required - stop openai_server first):
  .venv/bin/python _bench_decode.py
"""
import re
import time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn

import openai_server as S
from runtime.runtime import Packed, fwht

MODEL = "/Users/mestery/.lmstudio/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"


def collect_packed(model):
    """Return [(path, packed_module), ...] for every Packed in the model."""
    out = []

    def walk(node, prefix):
        for k, v in dict(node).items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, Packed):
                out.append((p, v))
            elif isinstance(v, nn.Module):
                walk(v, p)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, nn.Module):
                        walk(item, f"{p}[{i}]")

    walk(model, "")
    return out


def group_of(path):
    p = re.sub(r"\[\d+\]", "", path)
    segs = [s for s in p.split(".") if s]
    last = segs[-1]
    parent = segs[-2] if len(segs) >= 2 else ""
    if last == "lm_head":
        return "lm_head"
    if last == "embed_tokens":
        return "embedding"
    if parent == "linear_attn":
        return f"gdn.{last}"
    if parent == "self_attn":
        return f"attn.{last}"
    if parent == "mlp":
        return f"mlp.{last}"
    return f"other.{last}"


def in_dim_for(path, args):
    """Actual activation width feeding a Packed at ``path`` (the compressed
    ``p.weight.shape[1]`` is NOT the true input width)."""
    p = re.sub(r"\[\d+\]", "", path)
    segs = [s for s in p.split(".") if s]
    last = segs[-1]
    hidden = args.hidden_size
    inter = args.intermediate_size
    attn_out = args.num_attention_heads * args.head_dim
    gdn_v = args.linear_num_value_heads * args.linear_value_head_dim
    if last == "lm_head":
        return hidden
    if last in ("q_proj", "k_proj", "v_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                "gate_proj", "up_proj"):
        return hidden
    if last in ("o_proj", "out_proj"):
        return attn_out  # == gdn_v for this model (both 6144)
    if last == "down_proj":
        return inter
    raise ValueError(f"unknown module {last!r} in {path}")


def bench_dense_split(model, make_prompt_ids):
    """Split per-token dense cost into Hadamard transform vs 2-bit GEMM.

    Measures each non-embedding Packed module in isolation (batch=1, R=16 reps):
      t_full  = mean time of mx.eval(p(x))         (transform + GEMM)
      t_trans = mean time of mx.eval(fwht(x,...))  (transform only, if p.block)
      t_gemm  = t_full - t_trans                   (GEMM; equal R cancels sync)
    No monkeypatching: avoids the instance-__call__ dispatch bug.
    """
    args = model.args
    REPS = 16
    group_acc = defaultdict(lambda: [0.0, 0.0])  # [transform_s, gemm_s]
    total = [0.0, 0.0]
    n = 0

    for path, p in collect_packed(model):
        if p.embedding:
            continue
        g = group_of(path)
        in_dim = in_dim_for(path, args)
        x = mx.zeros((1, in_dim)).astype(p.dtype)
        mx.eval(x)
        for _ in range(3):  # warmup (kernel compile / L2 cache)
            mx.eval(p(x))
        t0 = time.perf_counter()
        for _ in range(REPS):
            mx.eval(p(x))
        t_full = (time.perf_counter() - t0) / REPS
        if p.block:
            t0 = time.perf_counter()
            for _ in range(REPS):
                mx.eval(fwht(x, p.block, p.signs))
            t_trans = (time.perf_counter() - t0) / REPS
        else:
            t_trans = 0.0
        t_gemm = max(t_full - t_trans, 0.0)
        group_acc[g][0] += t_trans
        group_acc[g][1] += t_gemm
        total[0] += t_trans
        total[1] += t_gemm
        n += 1

    tt, tg = total
    dense = tt + tg
    print(f"  packed modules measured (non-embedding): {n}")
    print(f"  Hadamard transform: {tt * 1000:7.2f} ms  ({tt / max(dense, 1e-9):.2f} of dense)")
    print(f"  2-bit GEMM:         {tg * 1000:7.2f} ms  ({tg / max(dense, 1e-9):.2f} of dense)")
    print(f"  dense total:        {dense * 1000:7.2f} ms")
    print("  by group (transform | gemm, ms; frac of dense):")
    order = sorted(group_acc, key=lambda g: -(group_acc[g][0] + group_acc[g][1]))
    for g in order:
        a = group_acc[g]
        s = a[0] + a[1]
        print(f"    {g:<18} {a[0] * 1000:7.2f} | {a[1] * 1000:7.2f}   ({s / max(dense, 1e-9):.2f})")


def main():
    print("device:", mx.default_device(), "| metal:", mx.metal.is_available())
    model, tok = S.load_model(MODEL)
    mx.set_default_device(mx.gpu)

    layers = model.model.layers
    n_gdn = sum(1 for l in layers if hasattr(l, "linear_attn"))
    print(f"layers: {len(layers)} (GDN={n_gdn}, full-attn={len(layers) - n_gdn})")
    print(f"vocab: {model.args.vocab_size}, hidden: {model.args.hidden_size}")

    base_ids = tok.encode(
        "Quantum entanglement is a phenomenon in which pairs of particles "
        "interact in ways that cannot be explained by classical physics. ",
        add_special_tokens=False,
    )

    def make_prompt_ids(n):
        reps = n // len(base_ids) + 1
        return (base_ids * reps)[:n]

    def bench_decode(pre_len, dec_tokens, warmup=3):
        cache = model.make_cache()
        x = mx.array([make_prompt_ids(pre_len)])
        lg = S._step_logits(model, x, cache)
        mx.eval(lg)
        last = int(lg[0].argmax())
        for _ in range(warmup):
            lg = S._step_logits(model, mx.array([[last]]), cache)
            mx.eval(lg)
            last = int(lg[0].argmax())
        t0 = time.perf_counter()
        for _ in range(dec_tokens):
            lg = S._step_logits(model, mx.array([[last]]), cache)
            mx.eval(lg)
            last = int(lg[0].argmax())
        t1 = time.perf_counter()
        return dec_tokens / (t1 - t0), (t1 - t0) / dec_tokens

    print("\n=== decode throughput (greedy) vs context length ===")
    for pre in (256, 8192, 16384):
        tps, ms = bench_decode(pre, 64)
        print(f"  prefill={pre:>6}: {tps:7.2f} tok/s  ({ms * 1000:6.1f} ms/token)")

    print("\n=== prefill throughput (cap 16384: 32768 OOMs the 64 GB GPU) ===")
    mx.eval(model.parameters())
    for pre in (256, 4096, 16384):
        cache = model.make_cache()
        x = mx.array([make_prompt_ids(pre)])
        t0 = time.perf_counter()
        lg = S._step_logits(model, x, cache)
        mx.eval(lg)
        t1 = time.perf_counter()
        print(f"  prefill={pre:>6}: {pre / (t1 - t0):8.1f} tok/s")

    print("\n=== lm_head isolated (per-token cost) ===")
    h = mx.random.normal((1, model.args.hidden_size)).astype(mx.float16)
    for _ in range(3):
        mx.eval(model.lm_head(h))
    t0 = time.perf_counter()
    for _ in range(32):
        mx.eval(model.lm_head(h))
    t1 = time.perf_counter()
    print(f"  lm_head: {(t1 - t0) / 32 * 1000:.2f} ms/token")

    print("\n=== per-layer profile (one decode token, prefill=8192) ===")
    cache = model.make_cache()
    x = mx.array([make_prompt_ids(8192)])
    lg = S._step_logits(model, x, cache)
    mx.eval(lg)
    last = int(lg[0].argmax())
    lg = S._step_logits(model, mx.array([[last]]), cache)
    mx.eval(lg)
    last = int(lg[0].argmax())

    times = [0.0] * len(layers)
    origs = []
    for i, l in enumerate(layers):
        orig = l.__call__

        def timed(*a, _i=i, _orig=orig, **k):
            t0 = time.perf_counter()
            r = _orig(*a, **k)
            times[_i] = time.perf_counter() - t0

        l.__call__ = timed
        origs.append((i, l, orig))
    try:
        lg = S._step_logits(model, mx.array([[last]]), cache)
        mx.eval(lg)
        total = sum(times)
        gdn_sum = sum(times[i] for i, l in enumerate(layers) if hasattr(l, "linear_attn"))
        full_sum = total - gdn_sum
        print(f"  transformer layers total: {total * 1000:.1f} ms")
        print(f"    GDN layers ({n_gdn}):        {gdn_sum * 1000:.1f} ms "
              f"({gdn_sum / max(total, 1e-9):.2f})")
        print(f"    full-attn ({len(layers) - n_gdn}): {full_sum * 1000:.1f} ms "
              f"({full_sum / max(total, 1e-9):.2f})")
        worst = sorted(range(len(times)), key=lambda i: -times[i])[:5]
        for i in worst:
            kind = "GDN" if hasattr(layers[i], "linear_attn") else "full"
            print(f"    layer {i:>2} ({kind}): {times[i] * 1000:.2f} ms")
    finally:
        for i, l, orig in origs:
            l.__call__ = orig

    print("\n=== dense-portion split: Hadamard transform vs 2-bit GEMM (one decode token, prefill=8192) ===")
    bench_dense_split(model, make_prompt_ids)

    print("\ndone.")


if __name__ == "__main__":
    main()