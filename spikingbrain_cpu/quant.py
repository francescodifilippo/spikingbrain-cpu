"""Weight storage formats for linear layers on CPU.

``float32``  plain fp32 weights (reference; doubles the checkpoint size).
``bfloat16`` bf16 weights, activations cast to bf16 for the matmul
             (closest to the checkpoint; slow on CPUs without AVX512-BF16/AMX).
``float16``  fp16 weights through the fbgemm fp16 GEMM with fp32 activations.
             Recommended: near-bf16 accuracy, fastest accurate decode.
``int8``     symmetric per-output-channel int8 weights, fp32 activations
             (weight-only quantisation, dequantised block by block). About
             half the memory of float16 and about half its decode speed.
``int8-dyn`` int8 weights through the fbgemm dynamic quantised linear
             (activations quantised per tensor at run time). Fastest by far
             on AVX2/AVX-512 VNNI; measure its fidelity before relying on it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMATS = ("float32", "bfloat16", "float16", "int8", "int8-dyn")


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


class Int8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, block: int = 128):
        super().__init__()
        q, scale = quantize_per_channel(weight)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        self.bias = None if bias is None else nn.Parameter(bias.float(), requires_grad=False)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch._weight_int8pack_mm is ~80x slower than fp32 GEMM for prefill
        # shapes on CPU. Dequantising small row blocks keeps each block in
        # cache, which is the fastest accurate option measured for both
        # decode and prefill; the per-row scale is applied afterwards.
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        parts = [x2 @ self.qweight[lo:lo + self.block].float().t()
                 for lo in range(0, self.qweight.shape[0], self.block)]
        out = torch.cat(parts, -1) * self.scale
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], -1)


class DynamicInt8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        q, scale = quantize_per_channel(weight)
        qweight = torch._make_per_channel_quantized_tensor(q, scale.double(), torch.zeros_like(scale, dtype=torch.long), 0)
        self.packed = torch.ops.quantized.linear_prepack(qweight, None if bias is None else bias.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out = torch.ops.quantized.linear_dynamic(x.reshape(-1, shape[-1]).float().contiguous(), self.packed)
        return out.reshape(*shape[:-1], -1)


class FP16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.packed = torch.ops.quantized.linear_prepack_fp16(weight.float(), None if bias is None else bias.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out = torch.ops.quantized.linear_dynamic_fp16(x.reshape(-1, shape[-1]).float().contiguous(), self.packed)
        return out.reshape(*shape[:-1], -1)


class BF16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight", weight.to(torch.bfloat16).contiguous())
        self.bias = None if bias is None else nn.Parameter(bias.to(torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(torch.bfloat16), self.weight, self.bias).float()


class FloatLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight", weight.float().contiguous())
        self.bias = None if bias is None else nn.Parameter(bias.float(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight, self.bias)


def make_linear(fmt: str, weight: torch.Tensor, bias: torch.Tensor | None = None) -> nn.Module:
    if fmt == "float32":
        return FloatLinear(weight, bias)
    if fmt == "bfloat16":
        return BF16Linear(weight, bias)
    if fmt == "float16":
        return FP16Linear(weight, bias)
    if fmt == "int8":
        return Int8Linear(weight, bias)
    if fmt == "int8-dyn":
        return DynamicInt8Linear(weight, bias)
    raise ValueError("unknown weight format " + fmt)
