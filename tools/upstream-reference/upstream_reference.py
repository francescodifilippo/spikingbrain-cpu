"""Layer-by-layer comparison of the CPU port against the upstream implementation.

Runs the original upstream block code (spb2/*.py and the flash-linear-attention
Triton kernels in TRITON_INTERPRET=1 mode, fp32) with real checkpoint weights,
one layer at a time, and compares every block output with the CPU port given
the same input. Covers a prefill of the prompt plus incremental decode steps.

Environment: PYTHONPATH must expose the upstream sources (spb2, fla, moba), a
flash_attn shim (see flash_attn_shim.py), and spikingbrain_cpu.
Usage: python upstream_reference.py MODEL_DIR OUT.jsonl [DECODE_STEPS]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.runtime.autotuner  # noqa: E402
from triton.runtime.jit import KernelInterface  # noqa: E402


class _FirstConfig(KernelInterface):
    """Interpreter mode has no GPU driver to benchmark with: use the first config.

    Tile sizes change scheduling, not the mathematics of the kernels.
    """

    def __init__(self, fn, config):
        self.fn, self.config = fn, config
        self.arg_names = getattr(fn, "arg_names", None)

    def run(self, *args, **kwargs):
        merged = dict(self.config.kwargs)
        merged.update(kwargs)
        return self.fn.run(*args, **merged)


def _autotune(configs, key=None, **_):
    return lambda fn: _FirstConfig(fn, configs[0])


triton.autotune = _autotune
triton.runtime.autotuner.autotune = _autotune

class _CpuTarget:
    backend, arch, warp_size = "cpu", 0, 32


class _CpuUtils:
    @staticmethod
    def get_device_properties(index=0):
        return {"multiprocessor_count": 4, "max_shared_mem": 232448, "max_num_regs": 65536, "warpSize": 32}


class _CpuDriver:
    """Neutral device properties for fla's grid/tiling helpers in interpreter mode."""

    utils = _CpuUtils()

    def get_current_target(self):
        return _CpuTarget()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        return torch.device("cpu")


import inspect  # noqa: E402
import textwrap  # noqa: E402

import triton.runtime.interpreter as _interp  # noqa: E402

# Triton 3.2 interpreter: tensor.__index__ = int(self.handle.data) fails when a
# scalar loaded from memory is held as a one-element array. Re-define only the
# function that installs it, using .item() for single-element arrays.
_BROKEN = "tensor.__index__ = lambda self: int(self.handle.data)"
_FIXED = "tensor.__index__ = lambda self: int(self.handle.data.reshape(-1)[0]) if getattr(self.handle.data, 'size', 1) == 1 else int(self.handle.data)"
for _fname, _func in list(vars(_interp).items()):
    if inspect.isfunction(_func) and _func.__module__ == _interp.__name__:
        try:
            _src = inspect.getsource(_func)
        except OSError:
            continue
        if _BROKEN in _src:
            exec(textwrap.dedent(_src).replace(_BROKEN, _FIXED), vars(_interp))

from triton.runtime import driver as _triton_driver  # noqa: E402

_triton_driver.set_active(_CpuDriver())

# fla selects torch.<backend>.device(index) as a context manager; the CPU
# backend has no such manager, and there is nothing to switch on a CPU.
if not hasattr(torch.cpu, "device"):
    import contextlib

    torch.cpu.device = lambda index=None: contextlib.nullcontext()

# Triton 3.2's interpreter patches tl.* builtins but not module-level aliases
# captured before a kernel runs (fla.ops.utils.op: exp = tl.exp, ...). Rebind
# them to jitted wrappers before any kernel module imports them.
import triton.language as tl  # noqa: E402
import fla.ops.utils.op as fla_op  # noqa: E402


@triton.jit
def _exp(x):
    return tl.exp(x)


@triton.jit
def _exp2(x):
    return tl.math.exp2(x)


@triton.jit
def _log(x):
    return tl.log(x)


@triton.jit
def _log2(x):
    return tl.log2(x)


_aliases = {id(tl.exp): _exp, id(tl.math.exp2): _exp2, id(tl.log): _log, id(tl.log2): _log2}
for _name, _module in list(sys.modules.items()):
    if _name == "fla" or _name.startswith("fla."):
        for _attr, _value in list(vars(_module).items()):
            if id(_value) in _aliases:
                setattr(_module, _attr, _aliases[id(_value)])

import moba  # noqa: E402
import spb2.moba_attn as upstream_moba  # noqa: E402
from fla.models.utils import Cache  # noqa: E402
from spb2.configuration_sse_swa_moba import SSESWAMoBAConfig  # noqa: E402
from spb2.modeling_sse_swa_moba import SSESWAMoBABlock  # noqa: E402

from spikingbrain_cpu.chat import ChatTemplate  # noqa: E402
from spikingbrain_cpu.config import ModelConfig  # noqa: E402
from spikingbrain_cpu.model import Block, KVBuffer, LayerState  # noqa: E402
from spikingbrain_cpu.quant import make_linear  # noqa: E402

# The efficient MoBA kernel needs flash-attn internals; the upstream naive
# reference implements the same selection rule with plain tensor operations.
upstream_moba.moba_attn_varlen = moba.moba_attn_varlen_naive

torch.set_grad_enabled(False)
model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
decode_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
raw = json.loads((model_dir / "config.json").read_text())
upstream_cfg = SSESWAMoBAConfig(**{k: v for k, v in raw.items() if k not in ("auto_map", "architectures")})
port_cfg = ModelConfig.from_dict(raw)

index = json.loads((model_dir / "pytorch_model.bin.index.json").read_text())["weight_map"]
shards = {name: torch.load(model_dir / name, map_location="cpu", weights_only=True, mmap=True)
          for name in sorted(set(index.values()))}


def tensor(key):
    return shards[index[key]][key].float()


def layer_state(i):
    prefix = f"model.layers.{i}."
    return {k[len(prefix):]: tensor(k) for k in index if k.startswith(prefix)}


def port_block(i):
    with torch.device("meta"):
        block = Block(port_cfg, i)
    for key, value in layer_state(i).items():
        parent_name, _, leaf = key.rpartition(".")
        parent = block.get_submodule(parent_name)
        if isinstance(parent, torch.nn.Linear):
            grand, _, child = parent_name.rpartition(".")
            setattr(block.get_submodule(grand), child, make_linear("float32", value))
        else:
            setattr(parent, leaf, torch.nn.Parameter(value, requires_grad=False))
    return block


def upstream_block(i):
    block = SSESWAMoBABlock(upstream_cfg, i).float().eval()
    block.load_state_dict(layer_state(i), strict=True)
    if hasattr(block.attn, "mode"):
        # Upstream picks the chunked kernel for long prefills; its solve_tril
        # kernel does not run under the Triton 3.2 interpreter. The fused
        # recurrent kernel (used upstream for short inputs and decoding)
        # computes the same recurrence.
        block.attn.mode = "fused_recurrent"
    return block


def port_state(i):
    d = port_cfg.head_dim
    if i in port_cfg.attn_layers:
        return LayerState(kv=KVBuffer(port_cfg.attn_num_kv_heads, d))
    return LayerState(kv=KVBuffer(port_cfg.attn_num_kv_heads, d, port_cfg.window_size),
                      shared=torch.zeros(port_cfg.num_heads, d, d),
                      partitions=torch.zeros(port_cfg.num_sparse_partition, port_cfg.num_heads, d, d))


def cache_for(i):
    """A cache whose slot ``i`` is the next one to be created, as inside the full model."""
    cache = Cache()
    if i and getattr(cache, "use_layer_class_to_replicate", True):
        while len(cache.layers) < i:
            cache.layers.append(cache.layer_class_to_replicate())
    elif i:
        cache.append_new_layers(i - 1)
    return cache


from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
template = ChatTemplate(json.loads((model_dir / "tokenizer_config.json").read_text())["chat_template"])
tools = [{"type": "function", "function": {
    "name": "pods_list_in_namespace", "description": "List the Kubernetes pods in the specified namespace",
    "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}}, "required": ["namespace"]}}},
    {"type": "function", "function": {
        "name": "events_list", "description": "List Kubernetes events in a namespace or cluster-wide",
        "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}}}}}]
prompt = template.render([
    {"role": "system", "content": "You inspect a Kubernetes cluster with tools. Read-only."},
    {"role": "user", "content": (
        "Context from the on-call notes: since 10:40 UTC the payments API reports intermittent 502 errors. "
        "The database client logs show connection refused, the node worker-3 had MemoryPressure and two pods "
        "were evicted. A deployment rollout happened at 10:35 UTC with a new image tag and a lower memory limit. "
        "Monitoring dashboards show container memory close to the limit and restarts increasing. "
        "Which pods are running in namespace monitoring? Check warning events too, then tell me which "
        "evidence you still need before drawing any conclusion about the root cause.")}], tools)
ids = tok.encode(prompt, add_special_tokens=False).ids
extra = tok.encode('<tool_call>\n{"name": "pods', add_special_tokens=False).ids[:decode_steps]
all_ids = ids + extra
T = len(ids)

embed = tensor("model.embeddings.weight")
x_up = embed[torch.tensor(all_ids)][None]  # upstream chain input [1, L, H]
x_port = x_up[0].clone()                    # port chain input [L, H]

out = out_path.open("w")


def emit(**row):
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(json.dumps(row), flush=True)


# CACHE_MODE=training: decode token j is at position T+j in every layer.
# CACHE_MODE=upstream-native: reproduce the full upstream model, where FLACache
#   increments _seen_tokens at layer 0, so layers >= 1 see T+j+1 while decoding.
CACHE_MODE = os.environ.get("CACHE_MODE", "training")
LAST_LAYER = int(os.environ.get("LAST_LAYER", str(port_cfg.num_hidden_layers - 1)))
COMPONENT_LAYERS = {int(v) for v in os.environ.get("COMPONENT_LAYERS", "").split(",") if v}


def record(block, names):
    """Capture outputs of the named submodules on every call."""
    store = {n: [] for n in names}
    handles = []
    for n in names:
        try:
            module = block.get_submodule(n)
        except AttributeError:
            continue
        handles.append(module.register_forward_hook(
            lambda _m, _i, out, n=n: store[n].append(out.reshape(-1, out.shape[-1]).detach().clone())))
    return store, handles


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


emit(step="start", prompt_tokens=T, decode_steps=len(extra), layers=port_cfg.num_hidden_layers,
     cache_mode=CACHE_MODE, last_layer=LAST_LAYER, component_layers=sorted(COMPONENT_LAYERS))
for i in range(LAST_LAYER + 1):
    started = time.monotonic()
    up, mine = upstream_block(i), port_block(i)
    cache = cache_for(i)
    # Positions follow training semantics (prompt 0..T-1, decode token j at T+j).
    # Note: with transformers > 4.53.3 the upstream FLACache bumps _seen_tokens
    # at layer 0, so layers 1..N see decode positions shifted by one; this
    # harness sets the count explicitly instead.
    # upstream: prefill then token-by-token decode through its own cache
    components = i in COMPONENT_LAYERS
    names = ["attn.sse_o_proj", "attn.swa_o_proj", "attn.o_proj"]
    up_store, up_hooks = record(up, names) if components else ({}, [])
    my_store, my_hooks = record(mine, names) if components else ({}, [])
    up_states, my_states = [], []
    hybrid = i not in port_cfg.attn_layers

    def snap_up():
        if hybrid and components:
            up_states.append(cache[i]["recurrent_state"].detach().clone().float())

    def snap_mine(st):
        if hybrid and components:
            my_states.append(torch.cat([st.shared[None], st.partitions], 0).clone())

    cache._seen_tokens = 0
    y_up = [up(x_up[:, :T], past_key_values=cache, use_cache=True)[0]]
    snap_up()
    for j in range(len(extra)):
        cache._seen_tokens = T + j + (1 if CACHE_MODE == "upstream-native" and i > 0 else 0)
        y_up.append(up(x_up[:, T + j:T + j + 1], past_key_values=cache, use_cache=True)[0])
        snap_up()
    y_up = torch.cat(y_up, 1)
    # port on the same upstream input (isolates this block)
    state = port_state(i)
    y_same = [mine(x_up[0, :T], state, 0)]
    snap_mine(state)
    for j in range(len(extra)):
        y_same.append(mine(x_up[0, T + j:T + j + 1], state, T + j))
        snap_mine(state)
    y_same = torch.cat(y_same, 0)
    for h in up_hooks + my_hooks:
        h.remove()
    if components:
        for n in names:
            if up_store.get(n) and my_store.get(n):
                a, b = torch.cat(my_store[n], 0), torch.cat(up_store[n], 0)
                emit(step="component", layer=i, component=n, cache_mode=CACHE_MODE,
                     prefill_rel_err=round(rel(a[:T], b[:T]), 8),
                     decode_rel_err=[round(rel(a[T + j:T + j + 1], b[T + j:T + j + 1]), 8) for j in range(len(extra))])
        for k, (a, b) in enumerate(zip(my_states, up_states)):
            emit(step="recurrent_state", layer=i, after="prefill" if k == 0 else f"decode_{k}",
                 shared_rel_err=round(rel(a[0], b[0]), 8),
                 partition_rel_err=[round(rel(a[p + 1], b[p + 1]), 8) for p in range(a.shape[0] - 1)],
                 partition_norms=[round(float(b[p + 1].norm()), 3) for p in range(b.shape[0] - 1)])
    # port on its own chain (end-to-end drift)
    fresh = port_state(i)
    y_chain = [mine(x_port[:T], fresh, 0)]
    for j in range(len(extra)):
        y_chain.append(mine(x_port[T + j:T + j + 1], fresh, T + j))
    y_chain = torch.cat(y_chain, 0)
    ref = y_up[0]
    diff = (y_same - ref).abs()
    scale = ref.abs().max()

    emit(step="layer", layer=i, cache_mode=CACHE_MODE, kind="hybrid" if i not in port_cfg.attn_layers else
         ("full" if i in port_cfg.attn_full_layers else "moba"),
         prefill_rel_err=round(rel(y_same[:T], ref[:T]), 7),
         prefill_rel_err_pos_0_127=round(rel(y_same[:128], ref[:128]), 7),
         prefill_rel_err_pos_128_255=round(rel(y_same[128:256], ref[128:256]), 7),
         prefill_rel_err_pos_256_end=round(rel(y_same[256:T], ref[256:T]), 7),
         decode_rel_err=round(rel(y_same[T:], ref[T:]), 7) if extra else None,
         max_abs_err=round(float(diff.max()), 6), ref_max_abs=round(float(scale), 3),
         chain_rel_err=round(rel(y_chain, ref), 6), seconds=round(time.monotonic() - started, 1))
    x_up, x_port = y_up, y_chain
    del up, mine

if LAST_LAYER < port_cfg.num_hidden_layers - 1:
    sys.exit(0)
norm_w = tensor("model.norm.weight")


def final_logits(h):
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + port_cfg.norm_eps) * norm_w
    return h @ embed.t()


lu = torch.log_softmax(final_logits(x_up[0]), -1)
lp = torch.log_softmax(final_logits(x_port), -1)
kl = (lu.exp() * (lu - lp)).sum(-1)
first = torch.softmax(final_logits(x_up[0, T - 1:T]), -1)[0]
top = torch.topk(first, 8)
emit(step="logits", positions=len(all_ids), top1_agreement=round(float((lu.argmax(-1) == lp.argmax(-1)).float().mean()), 4),
     mean_kl=round(float(kl.mean()), 6), max_kl=round(float(kl.max()), 5),
     upstream_first_token_top=[[tok.id_to_token(int(t)), int(t), round(float(p), 4)] for p, t in zip(top.values, top.indices)],
     upstream_tool_call_prob=round(float(first[tok.token_to_id("<tool_call>")]), 5))
