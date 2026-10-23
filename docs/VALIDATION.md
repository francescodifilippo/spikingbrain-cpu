# Validation against upstream

The question this document answers: does the CPU port compute the same function as the original SpikingBrain2.0 code with the same weights?

## Method

`tools/upstream-reference/upstream_reference.py` runs the original upstream block code next to the port, one layer at a time:

* Upstream sources: `BICLab/SpikingBrain2.0` at commit `02561132682c2a66925bb0df9f6a1d3f9e816723`, including its bundled flash-linear-attention and MoBA trees.
* The Triton kernels run on CPU through `TRITON_INTERPRET=1` (Triton 3.2.0, Transformers 4.57.1, PyTorch 2.14.1+cpu). The interpreter has no GPU to benchmark with, so the harness pins each autotuned kernel to its first configuration and provides neutral device properties.
* Three upstream components cannot run on CPU and are substituted:
  * flash-attn has no CPU build. `flash_attn_shim.py` implements `flash_attn_func` and `flash_attn_varlen_func` with plain fp32 softmax attention, following the flash-attn 2 mask semantics. It is written independently of the port's attention code.
  * MoBA's fused kernel is replaced by MoBA's own naive reference path.
  * For the hybrid layers, upstream picks its chunked gated-delta-rule kernel for long prefills (here 333 // 4 = 83 > 64). Its triangular-solve kernel does not run under the Triton 3.2 interpreter, so the harness forces upstream's fused recurrent kernel, which upstream itself uses for short inputs and for decoding. Both kernels compute the same recurrence. The port's chunked (WY) prefill is therefore compared against upstream's recurrent kernel, not against upstream's chunked kernel; the port's own unit tests check its chunked form against its recurrent form.
* Both sides use fp32 weights from the pinned checkpoint. Each upstream block and the matching port block receive the same input. The script records the relative L2 error of the block output for the prefill and for 4 incremental decode steps, and the error accumulated through the whole stack ("chain"). The final logits of both stacks are computed with the same final norm and output head.
* Prompt: a 333-token chat prompt with tool definitions, plus 4 decoded tokens taken from a fixed continuation (337 positions). The prompt fits inside one 4096-token MoBA chunk, so this run does not exercise MoBA's multi-chunk selection; the equivalence of MoBA and causal attention below 49,152 tokens rests on reading the upstream code (see [PORTING_NOTES.md](PORTING_NOTES.md#moba-equals-causal-attention-below-a-limit)).

`tools/upstream-reference/run-upstream-reference.sh` shows how the environment was prepared inside a container. In interpreter mode each complete layer-comparison iteration (loading the block, running upstream, running the port) takes about 14 seconds for a MoBA layer and about 25 minutes for a hybrid layer on 4 CPU threads, so a full run takes about 11 hours.

The full-run records were produced by an earlier revision of `upstream_reference.py` that did not yet print the `cache_mode` field; its position handling was the current default (`CACHE_MODE=training`), as the decode errors of 2e-7 or less in layers 1-35 confirm; the shifted counter gives 0.02-0.05 (see below).

## Results

All 36 layers agree:

| Layer type | Layers | Max prefill relative error | Max decode relative error | Max chain relative error |
| --- | --- | --- | --- | --- |
| MoBA attention | 0, 1, 2, 3, 6, 12, 17, 21, 24 | 3e-7 | 4e-7 | < 1e-6 (rounds to 0) |
| hybrid (gated delta net + SSE + sliding window) | 26 layers | 3e-7 | 2e-7 | 1e-6 |
| full attention | 35 | 3e-7 | 3e-7 | 1e-6 |

The prefill error is the same at positions 0-127, 128-255 and 256 or later, so the 128-token sliding-window boundary agrees. Decode agrees under training positions (see the next section). The largest absolute difference in any block output was 0.002 (layer 6), which is fp32 rounding on large activations.

Final logits over 337 positions: top-1 agreement 100 %. Mean and max KL divergence round to zero at the recorded precision (6 and 5 decimals, so below 1e-5).

The per-layer records are in [evidence/upstream-reference.jsonl](evidence/upstream-reference.jsonl).

## Scope of this result

* It shows that the `float32` port matches the `float32` upstream code on this input, with the substitutions listed above. Not covered by this run: MoBA multi-chunk selection, upstream's chunked delta-rule kernel, batching and padding, and the quantised formats. The other weight formats are compared against `bfloat16` separately, in [PERFORMANCE.md](PERFORMANCE.md). They are close, but not identical.
* One upstream behaviour was not copied. With Transformers newer than 4.53.3 (the upstream README recommends 4.57.1), the bundled `flash-linear-attention_dev/fla/models/utils.py` uses `FLACache`, whose `update()` increments the global `_seen_tokens` counter when layer 0 writes, and whose `get_seq_length()` returns that global counter. Layer 0 reads the counter before it writes, but layers 1 and above read it after, so during decoding they apply rotary embeddings one position ahead (prefill is not affected, because an empty layer reports length 0). The older `LegacyFLACache` path counts at the last layer and is correct. fla-org/flash-linear-attention fixed the same code in PR #748 (per-layer sequence length); the bundled copy does not include it. This affects decoding in the Hugging Face path only; the vLLM plugin does not use this cache. Reported upstream as [BICLab/SpikingBrain2.0#7](https://github.com/BICLab/SpikingBrain2.0/issues/7).
* The port uses the training positions (prompt 0..T-1, decode token j at T+j). Because the harness runs each block in isolation, it sets upstream's counter explicitly before every call: to the training positions in the full run (`CACHE_MODE=training`, the default).
* A separate run of layers 0-4 with `CACHE_MODE=upstream-native` sets the counter to the shifted value that `FLACache` produces. Prefill still agrees at 3e-7 or better. Decode then differs by a relative error of 0.019-0.048 in MoBA layers 1-3 and 0.005 in hybrid layer 4, where only the sliding-window branch moves (0.004-0.014) and the recurrent states still agree at 4e-7. Records: [training positions](evidence/upstream-reference-l0-4-training-positions.jsonl), [upstream-native positions](evidence/upstream-reference-l0-4-upstream-native-positions.jsonl).
* On the harness prompt, upstream gave `<tool_call>` a probability of 0.0081 as the first answer token, and the port's prediction there matches it within the recorded precision (KL below 1e-5). The low probability of the tool-call tag is therefore a property of the released checkpoint (weights and chat template), not of the port.

## Other checks

* Decode after a prefill gives the same next-token logits as a single prefill at positions 127, 128, 129, 255, 256, 4095 and 4096 (KL below 1e-7, `python -m spikingbrain_cpu.bench consistency MODEL_DIR float16`).
* Prefill in chunks of 512 and of 48 tokens gives identical per-position log-likelihood (`bench positions`).
* The unit tests in `tests/` run without the checkpoint.
