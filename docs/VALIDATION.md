# Validation against upstream

The question this document answers: does the CPU port compute the same function as the original SpikingBrain2.0 code with the same weights?

## Method

`tools/upstream-reference/upstream_reference.py` runs the original upstream block code next to the port, one layer at a time:

* Upstream sources: `BICLab/SpikingBrain2.0` at commit `02561132682c2a66925bb0df9f6a1d3f9e816723`, including its bundled flash-linear-attention and MoBA trees.
* The Triton kernels run on CPU through `TRITON_INTERPRET=1` (Triton 3.2.0, Transformers 4.57.1, PyTorch 2.14.1+cpu). flash-attn has no CPU build, so `flash_attn_shim.py` implements `flash_attn_func` and `flash_attn_varlen_func` with plain fp32 softmax attention, following the flash-attn 2 mask semantics. It is written independently of the port's attention code. MoBA's fused kernel is replaced by MoBA's own naive reference path.
* Both sides use fp32 weights from the pinned checkpoint. Each upstream block and the matching port block receive the same input. The script records the relative error of the block output for the prefill and for 4 incremental decode steps, and the error accumulated through the whole stack ("chain").
* Prompt: a 333-token chat prompt with tool definitions, plus 4 decoded tokens (337 positions). Finally the logits of both stacks are compared.

`tools/upstream-reference/run-upstream-reference.sh` shows how the environment was prepared inside a container. In interpreter mode each hybrid layer takes about 25 minutes on 4 CPU threads, so a full run takes about 11 hours.

## Results

All 36 layers agree:

| Layer type | Layers | Max prefill relative error | Max decode relative error | Max chain relative error |
| --- | --- | --- | --- | --- |
| MoBA attention | 0, 1, 2, 3, 6, 12, 17, 21, 24 | 3e-7 | 4e-7 | 0 |
| hybrid (gated delta net + SSE + sliding window) | 26 layers | 3e-7 | 2e-7 | 1e-6 |
| full attention | 35 | 3e-7 | 3e-7 | 1e-6 |

The prefill error is the same at positions 0-127, 128-255 and 256 or later, so the 128-token sliding-window boundary agrees. Decode agrees under training positions (see the next section). The largest absolute difference in any block output was 0.002 (layer 6), which is fp32 rounding on large activations.

Final logits over 337 positions: top-1 agreement 100 %, mean and max KL 0.

The per-layer records are in [evidence/upstream-reference.jsonl](evidence/upstream-reference.jsonl).

## Scope of this result

* It proves that the `float32` port equals the `float32` upstream code. The other weight formats are compared against `bfloat16` separately, in [PERFORMANCE.md](PERFORMANCE.md). They are close, but not identical.
* One upstream behaviour was not copied. With Transformers newer than 4.53.3 (the upstream README recommends 4.57.1), the bundled `flash-linear-attention_dev/fla/models/utils.py` uses `FLACache`, whose `update()` increments the global `_seen_tokens` counter when layer 0 writes, and whose `get_seq_length()` returns that global counter. Layer 0 reads the counter before it writes, but layers 1 and above read it after, so during decoding they apply rotary embeddings one position ahead (prefill is not affected, because an empty layer reports length 0). The older `LegacyFLACache` path counts at the last layer and is correct. fla-org/flash-linear-attention fixed the same code in PR #748 (per-layer sequence length); the bundled copy does not include it. The port uses the training positions (prompt 0..T-1, decode token j at T+j). The full run above sets these training positions explicitly on both sides (`CACHE_MODE=training`, the default). Prefill is not affected.
* A separate run of layers 0-4 with `CACHE_MODE=upstream-native` reproduces the shift. Prefill still agrees at 3e-7 or better. Decode then differs by a relative error of 0.019-0.048 in MoBA layers 1-3 and 0.005 in hybrid layer 4, where only the sliding-window branch moves (0.004-0.014) and the recurrent states still agree at 4e-7. Records: [training positions](evidence/upstream-reference-l0-4-training-positions.jsonl), [upstream-native positions](evidence/upstream-reference-l0-4-upstream-native-positions.jsonl).
* The probability of `<tool_call>` as the first answer token was 0.0081 in upstream itself. The weak tool calling described in the README therefore comes from the model weights, not from the port.

## Other checks

* Decode after a prefill gives the same next-token logits as a single prefill at positions 127, 128, 129, 255, 256, 4095 and 4096 (KL below 1e-7, `python -m spikingbrain_cpu.bench consistency MODEL_DIR float16`).
* Prefill in chunks of 512 and of 48 tokens gives identical per-position log-likelihood (`bench positions`).
* The unit tests in `tests/` run without the checkpoint.
