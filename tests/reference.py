"""Independent re-statement of the upstream sparse state expansion routing.

This follows the upstream ``sort_along_l`` + varlen kernel structure: tokens
are duplicated per selected partition, sorted by (partition, position), and the
gated delta rule runs over each segment. It is used only to cross-check the
per-partition loop in ``spikingbrain_cpu.model``.
"""

import torch
import torch.nn.functional as F


def l2norm(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def naive_recurrent(q, k, v, g, beta, h0, scale):
    # q, k, v: [T, H, D]; g, beta: [T, H]; h0: [H, K, V]
    h = h0.clone()
    out = torch.zeros_like(v)
    for t in range(q.shape[0]):
        for head in range(q.shape[1]):
            qt = l2norm(q[t, head]) * scale
            kt = l2norm(k[t, head])
            h[head] = h[head] * torch.exp(g[t, head])
            vt = beta[t, head] * (v[t, head] - (h[head] * kt[:, None]).sum(0))
            h[head] = h[head] + kt[:, None] * vt[None, :]
            out[t, head] = (h[head] * qt[:, None]).sum(0)
    return out, h


def sse_varlen_reference(q1, q2, k1, k2, v, g1, g2, b1, b2, eta_logits, shared, parts, writers):
    """Mirror of upstream sse_linear_attention_varlen for one sequence."""
    T, H, D = q1.shape
    N = eta_logits.shape[-1]
    scale = D ** -0.5
    e = F.softmax(eta_logits, -1)
    top_val, top_idx = torch.topk(e, k=writers, dim=-1)
    experts = top_idx.reshape(-1)
    values = top_val.reshape(-1)
    pos = torch.arange(T)[:, None].expand(T, writers).reshape(-1)
    order = torch.argsort(experts * (T + 1) + pos)
    experts, values, pos = experts[order], values[order], pos[order]
    q2s = q2[pos] * values[:, None, None]
    k2s = k2[pos] * values[:, None, None]
    o1, shared = naive_recurrent(q1, k1, v, g1, b1, shared, scale)
    o2 = torch.zeros(len(pos), H, D)
    parts = parts.clone()
    for n in range(N):
        sel = (experts == n).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        o, parts[n] = naive_recurrent(q2s[sel], k2s[sel], v[pos[sel]], g2[pos[sel]], b2[pos[sel]], parts[n], scale)
        o2[sel] = o
    out = o1.clone()
    out.index_add_(0, pos, o2)
    return out, shared, parts


def naive_attention(q, k, v, window=None):
    """Full-sequence causal attention with optional window, GQA by repetition."""
    T, Hq, D = q.shape
    rep = Hq // k.shape[1]
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    scores = torch.einsum("thd,shd->hts", q, k) * D ** -0.5
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    allowed = j <= i
    if window is not None:
        allowed &= j > i - window
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hts,shd->thd", scores.softmax(-1), v)
