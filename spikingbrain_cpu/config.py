"""Model configuration for the SpikingBrain2.0 (SSE-SWA-MoBA) text model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# MoBA layers select the top-k key chunks. While the context holds no more
# than moba_topk * moba_chunk_size tokens every past chunk is selected, so
# MoBA is exactly causal softmax attention. This port relies on that.


@dataclass
class ModelConfig:
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 9728
    vocab_size: int = 151936
    head_dim: int = 128
    num_heads: int = 32
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    # softmax attention layers (MoBA or full)
    attn_layers: list[int] = field(default_factory=list)
    attn_full_layers: list[int] = field(default_factory=list)
    attn_num_heads: int = 32
    attn_num_kv_heads: int = 8
    attn_qk_norm: bool = True
    attn_qkv_bias: bool = False
    rope_theta: float = 5_000_000.0
    window_size: int = 128
    moba_chunk_size: int = 4096
    moba_topk: int = 12
    # hybrid gated-delta-net + sparse state expansion layers
    linear_attn_type: str = "gdn"
    num_sparse_partition: int = 4
    num_writer: int = 2
    num_reader: int = 2
    expand_v: float = 1.0
    use_output_gate: bool = False
    use_short_conv: bool = False
    allow_neg_eigval: bool = False
    sse_qk_relu: bool = False
    sse_implementation: str = "varlen"
    fuse_norm: bool = True

    @property
    def exact_context_limit(self) -> int:
        """Longest context for which MoBA equals causal attention."""
        return self.moba_chunk_size * self.moba_topk

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        attn = raw.get("attn") or {}
        cfg = cls(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            head_dim=raw["head_dim"],
            num_heads=raw["num_heads"],
            norm_eps=raw.get("norm_eps", 1e-6),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
            bos_token_id=raw.get("bos_token_id", 151643),
            eos_token_id=raw.get("eos_token_id", 151645),
            attn_layers=list(attn.get("layers", [])),
            attn_full_layers=list(attn.get("full_layers", [])),
            attn_num_heads=attn.get("num_heads", raw["num_heads"]),
            attn_num_kv_heads=attn.get("num_kv_heads", raw["num_heads"]),
            attn_qk_norm=attn.get("qk_norm", False),
            attn_qkv_bias=attn.get("qkv_bias", False),
            rope_theta=attn.get("rope_theta", 10000.0),
            window_size=attn.get("window_size", 128),
            moba_chunk_size=attn.get("moba_chunk_size", 4096),
            moba_topk=attn.get("moba_topk", 12),
            linear_attn_type=raw.get("linear_attn_type", "gdn"),
            num_sparse_partition=raw.get("num_sparse_partition", 4),
            num_writer=raw.get("num_writer", 1),
            num_reader=raw.get("num_reader", 1),
            expand_v=raw.get("expand_v", 1.0),
            use_output_gate=raw.get("use_output_gate", False),
            use_short_conv=raw.get("use_short_conv", False),
            allow_neg_eigval=raw.get("allow_neg_eigval", False),
            sse_qk_relu=raw.get("sse_qk_relu", False),
            sse_implementation=raw.get("sse_implementation", "varlen"),
            fuse_norm=raw.get("fuse_norm", True),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        unsupported = []
        if self.linear_attn_type != "gdn":
            unsupported.append("linear_attn_type=" + self.linear_attn_type)
        if self.use_output_gate:
            unsupported.append("use_output_gate")
        if self.use_short_conv:
            unsupported.append("use_short_conv")
        if self.expand_v != 1.0:
            unsupported.append("expand_v")
        if self.sse_implementation != "varlen":
            unsupported.append("sse_implementation=" + self.sse_implementation)
        if self.num_writer != self.num_reader:
            unsupported.append("num_writer != num_reader")
        if not self.num_writer < self.num_sparse_partition:
            unsupported.append("num_writer >= num_sparse_partition")
        if self.attn_num_heads != self.num_heads:
            unsupported.append("attn num_heads differs from num_heads")
        if unsupported:
            raise ValueError("unsupported configuration: " + ", ".join(unsupported))
