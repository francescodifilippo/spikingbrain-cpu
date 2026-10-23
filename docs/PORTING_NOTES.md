# Porting notes: SpikingBrain2.0 on CPU

This document explains how the CPU port maps to the upstream implementation and which assumptions it relies on. Upstream references are to `BICLab/SpikingBrain2.0` (code under `spb2/`) and to the remote code shipped with the `SpikingBrain-2.0-instruct` checkpoint.

## Why a port is needed

The upstream text model (`SSESWAMoBAForCausalLM`) imports CUDA-only and Triton-only kernels:

| Upstream operator | Source | CPU replacement in this repository |
| --- | --- | --- |
| `flash_attn_func` / `flash_attn_varlen_func` (sliding window and causal attention) | flash-attn | `torch.nn.functional.scaled_dot_product_attention` with an explicit band/causal mask, processed in query blocks |
| `moba_attn_varlen` (MoBA sparse attention) | MoBA | causal attention, valid only below the exact context limit (see below) |
| `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule` | flash-linear-attention (Triton) | `chunk_gated_delta_rule` (WY form, triangular solve) and `recurrent_gated_delta_rule` in plain PyTorch |
| `RMSNorm`, fused residual norm, `GatedMLP`, `RotaryEmbedding` | flash-linear-attention | plain PyTorch equivalents |

The layer constructors upstream raise `ImportError` when flash-attn is not installed, so the upstream code cannot run on a CPU-only host even though its example script accepts `--device cpu`.

The upstream remote code is **never executed**. Weights are read with `torch.load(weights_only=True)` and mapped by name onto this implementation; loading is strict (every checkpoint tensor must be consumed and every parameter assigned).

## Model structure (instruct checkpoint)

* 36 blocks, hidden size 2560, MLP 9728 (SwiGLU), vocabulary 151936, tied input/output embeddings (Qwen tokenizer and chat template).
* 10 softmax-attention blocks (layers 0, 1, 2, 3, 6, 12, 17, 21, 24 use MoBA, layer 35 is full attention): 32 query heads, 8 KV heads, head dim 128, QK RMSNorm, RoPE base 5e6.
* 26 hybrid blocks: a gated delta net with **sparse state expansion (SSE)** averaged with a 128-token sliding-window attention (same head layout).

### Sparse state expansion

Each hybrid block keeps one shared recurrent state and four partition states (32 heads x 128 x 128 each, fp32). For every token:

1. The shared state is updated with (q1, k1, v, g1, beta1).
2. A router (`sse_e_proj`, softmax over 4 partitions) selects the top-2 partitions. q2/k2 (= q1/k1 plus a low-rank correction) are scaled by the routing weight, then L2-normalised inside the delta rule (so the scale only matters numerically through the epsilon), and only the selected partitions are updated and read. Unselected partitions keep their state unchanged (no decay).
3. Outputs of the shared path and the selected partitions are summed, RMS-normalised per head and projected.

Upstream implements step 2 by sorting duplicated tokens per partition (`sort_along_l`) and running a variable-length kernel. The port loops over the four partitions with index selection; `tests/reference.py` restates the upstream sorting structure and the tests check that both agree.

### MoBA equals causal attention below a limit

MoBA splits keys into 4096-token chunks. Each query attends to the top-12 chunks by gating score. Its own chunk is always selected and counts towards the 12, and future chunks are excluded (`moba_naive.py` sets their gate to -inf and the current chunk to +inf). With at most 12 chunks every past chunk is selected, so the result is exactly causal attention.

The upstream Hugging Face implementation (`spb2/moba_attn.py`) calls MoBA only when the query and key lengths are equal, that is a prefill without cache. Decoding, and any continuation over an existing cache, already use dense causal flash attention there, at every context length. (The vLLM plugin uses the separate `flash-moba` package and was not examined.)

The port therefore computes dense causal attention everywhere and refuses contexts longer than `moba_chunk_size * moba_topk` = 49152 tokens instead of approximating MoBA.

Why this is reasonable on a CPU:

* MoBA saves work only above 49152 tokens. Below that, it would select every chunk and add the cost of the gating step.
* At the measured ~40 tokens/s, a 49152-token prompt would take about 20 minutes to prefill (extrapolated; prefill was measured up to 6144 tokens). By operation count, the linear and MLP layers still outweigh attention at that length, so sparse attention would shorten it only partly.
* Above the limit, the key/value cache of the 10 attention layers grows by about 80 KB per token in fp32 (about 4 GB at 49152 tokens), because decode stays dense even upstream.

Lifting the limit would need a real MoBA prefill (gating on mean-pooled chunk keys, then attention over the selected chunks). It is not implemented.

## Numerical choices

* Activations are fp32. Weights are stored as fp16 (fbgemm fp16 GEMM), int8 (per-output-channel symmetric, weight-only), bf16, or fp32. The default is fp16: it is the most accurate fast format on typical server CPUs, which run bf16 matrix multiplications slowly unless they have AVX512-BF16 or AMX. int8 halves the memory at about half the decode speed. fp32 weights would need about 21 GB. Measured figures are in [PERFORMANCE.md](PERFORMANCE.md).
* `int8-dyn` uses the fbgemm dynamically quantised linear operator (activations quantised per tensor at run time). It loses visible accuracy (mean KL 0.17 against bf16) and is offered for experiments only.
* Recurrent states and the attention caches are fp32.

## Prefix reuse

The recurrent state has a fixed size, so the server can continue a conversation without recomputing its history. After each prompt it keeps a snapshot (recurrent states, window caches and the attention-cache length). A new request that extends the live token sequence or a snapshot only computes the new suffix. Agent loops that re-send the whole conversation each turn benefit the most. Results are identical to recomputing from scratch (covered by tests).

## Not ported

Training paths (window-attention dropout, auxiliary routing loss), batched requests, the vision-language model, the vLLM plugin (including its optional FP8 KV cache), and contexts above 49152 tokens.

The spiking (INT8-Spiking, event-driven) activation coding described in the upstream paper is not part of the released code at commit `0256113`: no file in `spb2/`, `spb2_vllm/`, `flash-linear-attention_dev/` or `MoBA/` implements it. Upstream and this port are both ordinary hybrid dense-plus-linear-attention models at inference time, and the neuromorphic efficiency claims do not apply to either.
