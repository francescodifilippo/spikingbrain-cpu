"""Pure PyTorch CPU inference port of the SpikingBrain2.0 SSE-SWA-MoBA model.

The upstream implementation depends on CUDA-only kernels (flash-attn and
Triton kernels from flash-linear-attention). This module re-implements the
same forward pass for batch size 1 with standard PyTorch operators:

* Hybrid layers: gated delta rule with sparse state expansion (one shared
  state plus top-k routed partition states) averaged with a sliding window
  softmax attention.
* Softmax layers: MoBA layers and the full attention layer. Below
  ``moba_chunk_size * moba_topk`` tokens MoBA selects every past chunk, so all
  of them are computed as causal attention and longer contexts are refused.

Only inference is supported; training-time branches (dropout of the window
path, auxiliary losses) are not ported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class Rotary:
    """GPT-NeoX style rotary embedding (rotate halves), computed on demand."""

    def __init__(self, dim: int, base: float):
        exponent = torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim
        self.inv_freq = 1.0 / (base ** exponent)

    def __call__(self, x: torch.Tensor, start: int) -> torch.Tensor:
        # x: [T, H, D]
        t = torch.arange(start, start + x.shape[0], dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], -1)[:, None, :]
        sin = torch.cat([freqs.sin(), freqs.sin()], -1)[:, None, :]
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], -1)
        return (x.float() * cos + rotated.float() * sin).to(x.dtype)


class KVBuffer:
    """Append-only key/value storage that can be truncated cheaply."""

    def __init__(self, heads: int, dim: int, window: int | None = None):
        self.heads, self.dim, self.window = heads, dim, window
        self.k = torch.empty(0, heads, dim)
        self.v = torch.empty(0, heads, dim)
        self.length = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new keys/values and return the keys/values visible to them."""
        if self.window is not None:
            keys = torch.cat([self.k[: self.length], k], 0)
            values = torch.cat([self.v[: self.length], v], 0)
            self.k = keys[-self.window:].clone()
            self.v = values[-self.window:].clone()
            self.length = self.k.shape[0]
            return keys, values
        need = self.length + k.shape[0]
        if need > self.k.shape[0]:
            capacity = max(need, 2 * self.k.shape[0], 256)
            grown_k = torch.empty(capacity, self.heads, self.dim, dtype=k.dtype)
            grown_v = torch.empty(capacity, self.heads, self.dim, dtype=v.dtype)
            grown_k[: self.length] = self.k[: self.length]
            grown_v[: self.length] = self.v[: self.length]
            self.k, self.v = grown_k, grown_v
        self.k[self.length:need] = k
        self.v[self.length:need] = v
        self.length = need
        return self.k[:need], self.v[:need]


@dataclass
class LayerState:
    kv: KVBuffer
    shared: torch.Tensor | None = None  # [H, K, V]
    partitions: torch.Tensor | None = None  # [P, H, K, V]


@dataclass
class InferenceState:
    layers: list[LayerState]
    position: int = 0
    tokens: list[int] = field(default_factory=list)


@dataclass
class Snapshot:
    position: int
    tokens: list[int]
    layers: list[tuple]


def snapshot(state: InferenceState) -> Snapshot:
    saved = []
    for layer in state.layers:
        if layer.shared is not None:
            saved.append(("hybrid", layer.shared.clone(), layer.partitions.clone(),
                          layer.kv.k.clone(), layer.kv.v.clone(), layer.kv.length))
        else:
            saved.append(("attn", layer.kv.length))
    return Snapshot(state.position, list(state.tokens), saved)


def restore(state: InferenceState, snap: Snapshot) -> None:
    for layer, saved in zip(state.layers, snap.layers, strict=True):
        if saved[0] == "hybrid":
            layer.shared = saved[1].clone()
            layer.partitions = saved[2].clone()
            layer.kv.k, layer.kv.v, layer.kv.length = saved[3].clone(), saved[4].clone(), saved[5]
        else:
            layer.kv.length = saved[1]
    state.position = snap.position
    state.tokens = list(snap.tokens)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def recurrent_gated_delta_rule(q, k, v, g, beta, state):
    """Token-by-token gated delta rule.

    q, k: [T, H, K] (already L2 normalised and q scaled), v: [T, H, V],
    g, beta: [T, H], state: [H, K, V] (fp32). Returns (o [T, H, V], state).
    """
    outputs = []
    for t in range(q.shape[0]):
        state = state * g[t].exp()[:, None, None]
        kt = k[t]
        predicted = torch.einsum("hkv,hk->hv", state, kt)
        update = beta[t][:, None] * (v[t] - predicted)
        state = state + kt[:, :, None] * update[:, None, :]
        outputs.append(torch.einsum("hkv,hk->hv", state, q[t]))
    return torch.stack(outputs, 0), state


def chunk_gated_delta_rule(q, k, v, g, beta, state, chunk_size: int = 64):
    """Chunked (WY representation) gated delta rule, same contract as the recurrent form."""
    T = q.shape[0]
    if T == 0:
        return v.new_zeros(0, *v.shape[1:]), state
    pad = (-T) % chunk_size
    if pad:
        # zero keys and gates leave the state untouched; padded outputs are dropped
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, 0, 0, pad)) for x in (g, beta))
    n = q.shape[0] // chunk_size
    # [H, N, C, D]
    q, k, v = (x.reshape(n, chunk_size, *x.shape[1:]).permute(2, 0, 1, 3) for x in (q, k, v))
    g, beta = (x.reshape(n, chunk_size, -1).permute(2, 0, 1) for x in (g, beta))
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    decay = g.cumsum(-1)  # [H, N, C]
    lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
    strict = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), -1)
    diff = (decay[..., :, None] - decay[..., None, :]).masked_fill(~lower, float("-inf"))
    gamma = diff.exp()  # [H, N, C, C], gamma[i, j] = prod g over (j, i]
    a = (k_beta @ k.transpose(-1, -2) * gamma).masked_fill(~strict, 0.0)
    eye = torch.eye(chunk_size, dtype=q.dtype)
    t_inv = torch.linalg.solve_triangular(eye + a, eye.expand_as(a), upper=False)
    u = t_inv @ v_beta
    w = t_inv @ (k_beta * decay[..., None].exp())
    outputs = []
    for i in range(n):
        qi, ki = q[:, i], k[:, i]
        local = (qi @ ki.transpose(-1, -2) * gamma[:, i]).masked_fill(~lower, 0.0)
        v_new = u[:, i] - w[:, i] @ state
        o = (qi * decay[:, i, :, None].exp()) @ state + local @ v_new
        outputs.append(o)
        last = decay[:, i, -1]
        state = state * last.exp()[:, None, None] + (
            ki * (last[:, None] - decay[:, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new
    o = torch.stack(outputs, 1).reshape(q.shape[0], n * chunk_size, -1).permute(1, 0, 2)
    return o[:T], state


def gated_delta_rule(q, k, v, g, beta, state, chunk_threshold: int = 64):
    if q.shape[0] <= chunk_threshold:
        return recurrent_gated_delta_rule(q, k, v, g, beta, state)
    return chunk_gated_delta_rule(q, k, v, g, beta, state)


def attention(q, k, v, start: int, window: int | None, block: int = 256):
    """Causal (optionally windowed) GQA attention for queries at positions start..start+T-1.

    q: [T, Hq, D]; k, v: [S, Hk, D] covering positions start+T-S .. start+T-1.
    """
    T, hq, d = q.shape
    s, hk = k.shape[0], k.shape[1]
    first_key = start + T - s
    key_pos = torch.arange(first_key, first_key + s)
    kk = k.transpose(0, 1)  # [Hk, S, D]
    vv = v.transpose(0, 1)
    out = []
    for lo in range(0, T, block):
        hi = min(T, lo + block)
        qpos = torch.arange(start + lo, start + hi)
        allowed = key_pos[None, :] <= qpos[:, None]
        if window is not None:
            allowed &= key_pos[None, :] > qpos[:, None] - window
        qb = q[lo:hi].transpose(0, 1)  # [Hq, B, D]
        o = F.scaled_dot_product_attention(
            qb.unsqueeze(0), kk.unsqueeze(0), vv.unsqueeze(0),
            attn_mask=allowed, enable_gqa=hq != hk,
        )
        out.append(o[0].transpose(0, 1))
    return torch.cat(out, 0)


class GatedMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SoftmaxAttention(nn.Module):
    """MoBA or full attention layer (identical below the exact context limit)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h, kvh, d = cfg.attn_num_heads, cfg.attn_num_kv_heads, cfg.head_dim
        self.heads, self.kv_heads, self.head_dim = h, kvh, d
        self.q_proj = nn.Linear(cfg.hidden_size, h * d, bias=cfg.attn_qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, kvh * d, bias=cfg.attn_qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, kvh * d, bias=cfg.attn_qkv_bias)
        self.o_proj = nn.Linear(h * d, cfg.hidden_size, bias=False)
        self.qk_norm = cfg.attn_qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(d, cfg.norm_eps)
            self.k_norm = RMSNorm(d, cfg.norm_eps)
        self.rotary = Rotary(d, cfg.rope_theta)

    def forward(self, x, layer: LayerState, start: int):
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.heads, self.head_dim)
        k = self.k_proj(x).view(T, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(T, self.kv_heads, self.head_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary(q, start), self.rotary(k, start)
        keys, values = layer.kv.append(k, v)
        o = attention(q, keys, values, start, None)
        return self.o_proj(o.reshape(T, -1))


class HybridSSEAttention(nn.Module):
    """Gated delta net with sparse state expansion, merged with window attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hid, h, d = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        self.heads, self.head_dim = h, d
        self.partitions = cfg.num_sparse_partition
        self.writers = cfg.num_writer
        self.allow_neg_eigval = cfg.allow_neg_eigval
        self.qk_act = F.relu if cfg.sse_qk_relu else F.silu
        self.sse_q_proj = nn.Linear(hid, h * d, bias=False)
        self.sse_k_proj = nn.Linear(hid, h * d, bias=False)
        self.sse_v_proj = nn.Linear(hid, h * d, bias=False)
        self.lora_q_proj = nn.Sequential(nn.Linear(hid, d, bias=False), nn.Linear(d, h * d, bias=False))
        self.lora_k_proj = nn.Sequential(nn.Linear(hid, d, bias=False), nn.Linear(d, h * d, bias=False))
        self.sse_a_proj = nn.Linear(hid, 2 * h, bias=False)
        self.sse_b_proj = nn.Linear(hid, 2 * h, bias=False)
        self.A_log = nn.Parameter(torch.zeros(2 * h))
        self.dt_bias = nn.Parameter(torch.zeros(2 * h))
        self.sse_e_proj = nn.Linear(hid, self.partitions, bias=False)
        self.sse_o_norm = RMSNorm(d, cfg.norm_eps)
        self.sse_o_proj = nn.Linear(h * d, hid, bias=False)
        kvh = cfg.attn_num_kv_heads
        self.swa_kv_heads = kvh
        self.swa_q_proj = nn.Linear(hid, h * d, bias=cfg.attn_qkv_bias)
        self.swa_k_proj = nn.Linear(hid, kvh * d, bias=cfg.attn_qkv_bias)
        self.swa_v_proj = nn.Linear(hid, kvh * d, bias=cfg.attn_qkv_bias)
        self.swa_o_proj = nn.Linear(h * d, hid, bias=False)
        self.swa_qk_norm = cfg.attn_qk_norm
        if self.swa_qk_norm:
            self.swa_q_norm = RMSNorm(d, cfg.norm_eps)
            self.swa_k_norm = RMSNorm(d, cfg.norm_eps)
        self.sse_merge_norm = RMSNorm(hid, cfg.norm_eps)
        self.swa_merge_norm = RMSNorm(hid, cfg.norm_eps)
        self.window = cfg.window_size
        self.rotary = Rotary(d, cfg.rope_theta)
        self.scale = d ** -0.5

    def _prepare(self, q, k):
        return _l2norm(q.float()) * self.scale, _l2norm(k.float())

    def sse_inputs(self, x):
        T, h, d = x.shape[0], self.heads, self.head_dim
        q1 = self.sse_q_proj(x)
        k1 = self.sse_k_proj(x)
        q2 = q1 + self.lora_q_proj(x)
        k2 = k1 + self.lora_k_proj(x)
        v = F.silu(self.sse_v_proj(x).view(T, h, d))
        q1, q2, k1, k2 = (self.qk_act(t.view(T, h, d)) for t in (q1, q2, k1, k2))
        b = self.sse_b_proj(x).sigmoid().float()
        if self.allow_neg_eigval:
            b = b * 2.0
        g = -self.A_log.float().exp() * F.softplus(self.sse_a_proj(x).float() + self.dt_bias.float())
        b1, b2 = b.chunk(2, -1)
        g1, g2 = g.chunk(2, -1)
        eta_logits = self.sse_e_proj(x).float()
        return q1, q2, k1, k2, v, g1, g2, b1, b2, eta_logits

    def sse_heads(self, x, layer: LayerState):
        q1, q2, k1, k2, v, g1, g2, b1, b2, eta_logits = self.sse_inputs(x)
        eta = F.softmax(eta_logits, -1)
        top_val, top_idx = torch.topk(eta, self.writers, dim=-1)  # [T, W]

        # shared state sees every token
        qs, ks = self._prepare(q1, k1)
        out, layer.shared = gated_delta_rule(qs, ks, v.float(), g1, b1, layer.shared)

        # each token writes to and reads from its top-k partitions; the token's
        # routing weight scales q and k before the in-kernel L2 normalisation
        for p in range(self.partitions):
            hit = top_idx == p  # [T, W]
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            weight = (top_val * hit).sum(-1)[rows].to(q2.dtype)[:, None, None]
            qp, kp = self._prepare(q2[rows] * weight, k2[rows] * weight)
            op, layer.partitions[p] = gated_delta_rule(
                qp, kp, v[rows].float(), g2[rows], b2[rows], layer.partitions[p]
            )
            out.index_add_(0, rows, op)
        return out

    def sse(self, x, layer: LayerState):
        out = self.sse_heads(x, layer)
        o = self.sse_o_norm(out.to(x.dtype))
        return self.sse_o_proj(o.reshape(x.shape[0], -1))

    def swa(self, x, layer: LayerState, start: int):
        T, d = x.shape[0], self.head_dim
        q = self.swa_q_proj(x).view(T, self.heads, d)
        k = self.swa_k_proj(x).view(T, self.swa_kv_heads, d)
        v = self.swa_v_proj(x).view(T, self.swa_kv_heads, d)
        if self.swa_qk_norm:
            q, k = self.swa_q_norm(q), self.swa_k_norm(k)
        q, k = self.rotary(q, start), self.rotary(k, start)
        keys, values = layer.kv.append(k, v)
        o = attention(q, keys, values, start, self.window)
        return self.swa_o_proj(o.reshape(T, -1))

    def forward(self, x, layer: LayerState, start: int):
        sse_o = self.sse(x, layer)
        swa_o = self.swa(x, layer, start)
        return (self.sse_merge_norm(sse_o) + self.swa_merge_norm(swa_o)) / 2


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.hybrid = idx not in cfg.attn_layers
        self.attn = HybridSSEAttention(cfg) if self.hybrid else SoftmaxAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = GatedMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, layer: LayerState, start: int):
        residual = x
        x = self.attn(self.attn_norm(x), layer, start)
        residual = x + residual
        return residual + self.mlp(self.mlp_norm(residual))


class Model(nn.Module):
    """Container whose parameter names match the upstream checkpoint keys."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)


class SpikingBrainForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def new_state(self) -> InferenceState:
        layers = []
        cfg = self.config
        for block in self.model.layers:
            if block.hybrid:
                h, d = cfg.num_heads, cfg.head_dim
                layers.append(LayerState(
                    kv=KVBuffer(cfg.attn_num_kv_heads, d, cfg.window_size),
                    shared=torch.zeros(h, d, d),
                    partitions=torch.zeros(cfg.num_sparse_partition, h, d, d),
                ))
            else:
                layers.append(LayerState(kv=KVBuffer(cfg.attn_num_kv_heads, cfg.head_dim)))
        return InferenceState(layers)

    @torch.inference_mode()
    def forward_tokens(self, tokens: list[int], state: InferenceState, chunk: int = 512,
                       all_logits: bool = False) -> torch.Tensor:
        """Feed tokens, update state in place and return logits of the last token.

        With ``all_logits`` the logits of every token of the last chunk are returned.
        """
        if not tokens:
            raise ValueError("no tokens")
        if state.position + len(tokens) > self.config.exact_context_limit:
            raise ValueError("context exceeds exact MoBA-equivalent limit")
        hidden = None
        for lo in range(0, len(tokens), chunk):
            ids = torch.tensor(tokens[lo:lo + chunk], dtype=torch.long)
            x = self.model.embeddings(ids).float()
            for block, layer in zip(self.model.layers, state.layers, strict=True):
                x = block(x, layer, state.position)
            state.position += ids.numel()
            hidden = x if all_logits else x[-1:]
        state.tokens.extend(tokens)
        logits = self.lm_head(self.model.norm(hidden)).float()
        return logits if all_logits else logits[0]


def build_reference_config(**overrides) -> ModelConfig:
    """A tiny configuration with the same structure, for tests."""
    base = dict(
        hidden_size=64, num_hidden_layers=4, intermediate_size=96, vocab_size=97,
        head_dim=16, num_heads=4, attn_layers=[0, 3], attn_full_layers=[3],
        attn_num_heads=4, attn_num_kv_heads=2, window_size=8, moba_chunk_size=64,
        moba_topk=4, num_sparse_partition=4, num_writer=2, num_reader=2,
        rope_theta=10000.0,
    )
    base.update(overrides)
    cfg = ModelConfig(**base)
    cfg.validate()
    return cfg


def init_random(model: SpikingBrainForCausalLM, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.empty_like(p).uniform_(1, 16, generator=gen).log())
            elif name.endswith("dt_bias"):
                p.copy_(torch.empty_like(p).uniform_(-4, -2, generator=gen))
            elif "norm" in name:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(p.shape[-1]))
