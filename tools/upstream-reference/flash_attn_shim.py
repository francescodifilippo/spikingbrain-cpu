"""Minimal stand-in for the flash_attn functions used by the upstream model code.

Semantics follow flash-attn 2: causal masks are aligned to the bottom-right
corner when the query is shorter than the key; window_size=(left, right)
limits keys to [i - left, i + right] relative to the aligned query position.
Plain fp32 einsum/softmax, independent of the CPU port's attention code.
"""

import torch


def _attend(q, k, v, causal, window_size, softmax_scale=None):
    # q [Tq, H, D], k/v [Tk, Hk, D]
    tq, h, d = q.shape
    tk, hk = k.shape[0], k.shape[1]
    k = k.repeat_interleave(h // hk, 1).float()
    v = v.repeat_interleave(h // hk, 1).float()
    scale = softmax_scale if softmax_scale is not None else d ** -0.5
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    i = torch.arange(tq)[:, None] + (tk - tq)
    j = torch.arange(tk)[None, :]
    allowed = torch.ones(tq, tk, dtype=torch.bool)
    left, right = window_size
    if causal:
        allowed &= j <= i
    if left >= 0:
        allowed &= j >= i - left
    if right >= 0 and not causal:
        allowed &= j <= i + right
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).to(q.dtype)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **_):
    return torch.stack([_attend(q[b], k[b], v[b], causal, window_size, softmax_scale) for b in range(q.shape[0])])


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **_):
    outs = []
    for b in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        ks, ke = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
        outs.append(_attend(q[qs:qe], k[ks:ke], v[ks:ke], causal, window_size, softmax_scale))
    return torch.cat(outs, 0)
