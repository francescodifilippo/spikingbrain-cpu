# Performance

All numbers below were measured once with `python -m spikingbrain_cpu.bench` on the pinned instruct checkpoint. Treat them as an order of magnitude for similar hardware, not as a guarantee.

## Test system

* One virtual machine with 8 vCPU (x86-64, AVX-512 reported by `torch.backends.cpu.get_cpu_capability()`), 32 GiB RAM.
* The server ran in a container limited to 4 CPU, with `SB_THREADS=4` and `OMP_NUM_THREADS=4`.
* PyTorch 2.14.1+cpu, Python 3.12, batch size 1, greedy decoding.
* The CPU model was not recorded. Scaling with thread count and behaviour on other CPUs (AVX2 only, AMX, ARM) were not measured.

## Results

### Loading and memory

| Format | Load time | RSS after load | Peak RSS during the benchmark |
| --- | --- | --- | --- |
| `float16` | 110.7 s | 10.9 GiB | 13.0 GiB |
| `int8` | 33.2 s | 6.1 GiB | 8.0 GiB |

Load time depends on storage speed. The figures above used a network block volume.

### Prefill (prompt processing)

Synthetic prompt, one forward pass per length:

| Prompt tokens | `float16` | `int8` |
| --- | --- | --- |
| 512 | 40.0 tokens/s (12.8 s) | 36.6 tokens/s (14.0 s) |
| 2,048 | 41.2 tokens/s (49.8 s) | 36.7 tokens/s (55.8 s) |
| 6,144 | 39.4 tokens/s (155.8 s) | 35.3 tokens/s (173.9 s) |

Speed stayed flat up to 6,144 tokens. Longer prompts were not measured. Very short prompts are slower per token: a 42-token chat prompt ran at 22.2 tokens/s (`float16`) and 15.1 tokens/s (`int8`).

### Decode (generation)

| Case | Completion tokens | `float16` | `int8` |
| --- | --- | --- | --- |
| short chat answer | 74-76 | 3.15 tokens/s | 1.61 tokens/s |
| tool call, `tool_choice: auto` | 23 | 3.06 tokens/s | 1.58 tokens/s |
| tool call, `tool_choice: required` | 13 | 2.98 tokens/s | 1.47 tokens/s |

### Accuracy of the weight formats

Teacher-forced comparison on a 466-token prompt against `bfloat16` weights (the checkpoint precision):

| Format | Top-1 agreement | Mean KL | Max KL | NLL (bf16: 0.9168) |
| --- | --- | --- | --- | --- |
| `float16` | 98.71 % | 0.00009 | 0.002 | 0.9173 |
| `int8` | 98.28 % | 0.00076 | 0.059 | 0.9183 |
| `int8-dyn` | 90.13 % | 0.17326 | 8.90 | 1.0549 |

`int8-dyn` quantises activations at run time and loses visible accuracy. Do not use it without measuring its quality on your own prompts.

## Sizing guidance

* **`float16`** is the default and the recommended format. Give the container at least 14 GiB of memory and a 16 GiB limit.
* **`int8`** fits in about 8 GiB (use a 10 GiB limit). Choose it when memory is the constraint and half the decode speed is acceptable.
* **Context length.** Ten attention layers keep a full fp32 key/value cache. At the 24,576-token default limit this cache is estimated at about 2 GiB, and the engine keeps up to two snapshots for prefix reuse. Memory at long context was not measured: leave headroom, or lower `SB_MAX_CONTEXT`.
* **Time budget.** First-token latency is roughly prompt tokens / 40 seconds, and generation adds roughly 1 second per 3 tokens with `float16`.
* **Threads.** Match `SB_THREADS` to the CPU limit of the container. More threads than physical cores usually make things slower.
* **One request at a time.** The server serialises requests. Run several replicas for parallel users, each with its own memory budget.

## Reproduce

```sh
# speed, memory and three generation cases
python -m spikingbrain_cpu.bench MODEL_DIR float16 int8
# accuracy of the formats against bfloat16
python -m spikingbrain_cpu.bench fidelity MODEL_DIR bfloat16 float16 int8 int8-dyn
```

Both commands print one JSON object per line. The published figures were produced by an earlier revision of `bench.py` whose chat prompt and fidelity prompt were partly written in Italian; the prompts are now English, so your token counts can differ slightly. Prefill and decode speed do not depend on the prompt language.
