"""Load an upstream SpikingBrain2.0 checkpoint without executing its remote code.

Shards are read one at a time (``.bin`` with ``weights_only=True`` and mmap,
or ``.safetensors``). Linear weights are converted to the requested storage
format as soon as they are read, so a full-precision copy of the model never
exists in memory.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

from .config import ModelConfig
from .model import SpikingBrainForCausalLM
from .quant import make_linear

log = logging.getLogger(__name__)


def _iter_shards(model_dir: Path):
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = model_dir / index_name
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            break
    else:
        files = [p.name for p in sorted(model_dir.glob("*.safetensors"))] or ["pytorch_model.bin"]
    for name in files:
        path = model_dir / name
        if name.endswith(".safetensors"):
            from safetensors.torch import load_file
            yield name, load_file(str(path))
        else:
            yield name, torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _assign(model: nn.Module, key: str, tensor: torch.Tensor, fmt: str, embed_dtype) -> None:
    parent_name, _, leaf = key.rpartition(".")
    parent = model.get_submodule(parent_name)
    if isinstance(parent, nn.Linear) and leaf == "weight":
        grand_name, _, child = parent_name.rpartition(".")
        grand = model.get_submodule(grand_name)
        replacement = make_linear(fmt, tensor, None)
        setattr(grand, child, replacement)
        return
    current = getattr(parent, leaf)
    if tuple(current.shape) != tuple(tensor.shape):
        raise ValueError(f"shape mismatch for {key}: {tuple(tensor.shape)} != {tuple(current.shape)}")
    dtype = embed_dtype if isinstance(parent, nn.Embedding) else torch.float32
    setattr(parent, leaf, nn.Parameter(tensor.to(dtype).clone(), requires_grad=False))


def expected_keys(model: nn.Module) -> set[str]:
    return {name for name, _ in model.named_parameters()}


def load_model(model_dir: str | Path, fmt: str = "float16", embed_dtype=torch.bfloat16):
    """Return (model, report). ``report`` records what was loaded and how."""
    model_dir = Path(model_dir)
    cfg = ModelConfig.from_file(model_dir / "config.json")
    with torch.device("meta"):
        model = SpikingBrainForCausalLM(cfg)
    expected = expected_keys(model)
    loaded: set[str] = set()
    unexpected: list[str] = []
    lm_head = None
    started = time.monotonic()
    for name, shard in _iter_shards(model_dir):
        for key, tensor in shard.items():
            if key == "lm_head.weight" and cfg.tie_word_embeddings:
                lm_head = tensor.to(torch.bfloat16).clone()
                loaded.add(key)
                continue
            if key not in expected:
                unexpected.append(key)
                continue
            if key.endswith(".bias"):
                raise ValueError("linear biases are not used by this model: " + key)
            _assign(model, key, tensor, fmt, embed_dtype)
            loaded.add(key)
        del shard
        log.info("loaded shard %s (%.0fs)", name, time.monotonic() - started)
    if cfg.tie_word_embeddings:
        # Upstream ties the output projection to the input embeddings after
        # loading, so the embedding matrix is authoritative.
        embedding = model.model.embeddings.weight
        tied_equal = lm_head is None or bool(torch.equal(lm_head, embedding.to(torch.bfloat16)))
        model.lm_head = make_linear(fmt, embedding.float() if fmt == "float32" else embedding, None)
        loaded.add("lm_head.weight")
    else:
        tied_equal = None
    missing = sorted(expected - loaded)
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    leftovers = [n for n, p in model.named_parameters() if p.is_meta]
    leftovers += [n for n, b in model.named_buffers() if b.is_meta]
    if leftovers:
        raise ValueError("parameters left on meta device: " + ", ".join(leftovers[:10]))
    model.eval()
    report = {
        "weight_format": fmt,
        "embedding_dtype": str(embed_dtype).replace("torch.", ""),
        "tensors_loaded": len(loaded),
        "stored_lm_head_equals_embeddings": tied_equal,
        "load_seconds": round(time.monotonic() - started, 1),
    }
    return model, report
