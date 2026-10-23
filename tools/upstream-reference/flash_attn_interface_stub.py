"""Import-time stub: the efficient MoBA kernel is replaced by the naive reference."""


def _flash_attn_varlen_forward(*args, **kwargs):
    raise RuntimeError("flash-attn internals are not available in the CPU reference")


_flash_attn_varlen_backward = _flash_attn_varlen_forward
