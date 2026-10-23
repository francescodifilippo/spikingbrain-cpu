# spikingbrain-cpu

A reference CPU implementation and verification harness for [BICLab/SpikingBrain2.0](https://github.com/BICLab/SpikingBrain2.0): exact, tested and readable. It is not a fast runtime.

The upstream text model depends on CUDA-only and Triton-only kernels, so it cannot run without a GPU. This repository re-implements its inference path in plain PyTorch, loads the released checkpoint unchanged, and checks it against the original code:

* On the reference input, the largest recorded relative L2 error of any block output against the original upstream code is 4e-7 (1e-6 accumulated through all 36 layers), and the final predictions agree on every position (top-1 100 %, KL below 1e-5). The harness that runs the upstream code on CPU (Triton interpreter mode, with three documented substitutions) is included, see [docs/VALIDATION.md](docs/VALIDATION.md).
* The hybrid gated delta net, sparse state expansion (SSE) and sliding-window attention are written as ordinary PyTorch code with unit tests. You can read, step through and test the architecture on any machine.
* A small OpenAI-compatible server, measured weight formats and Kubernetes manifests let you run the model on CPU-only hosts, slowly ([docs/PERFORMANCE.md](docs/PERFORMANCE.md)).

Use it to study the architecture, to check claims against the released code, or as the reference for another port (for example GGUF or ONNX). For production inference on CPU, an ordinary Transformer of similar size on an optimised runtime is likely faster and needs less memory.

This is an unofficial, independent project, not affiliated with the SpikingBrain authors. Model weights are not included.

## Findings

Running the released code and weights side by side with this port showed:

* **The spiking computation is not in the released code.** The INT8-Spiking, event-driven activation coding described in the paper has no implementation at commit `0256113` (see also upstream issue #6, a request for that inference code). Upstream and this port both compute every activation.
* **MoBA only matters above 49,152 tokens** with the instruct checkpoint's settings (4096-token chunks, top-12). Below that length its chunk selection picks every past chunk, so it is exactly causal attention. The upstream Hugging Face code also decodes with dense attention.
* **Weak tool calling is not a porting artefact.** Upstream itself gives the `<tool_call>` token a probability below 1 % at the start of an answer on the harness prompt, and the port matches it there. A full malformed call was not reproduced with upstream.
* **Upstream decode shifts rotary positions by one** when used with Transformers newer than 4.53.3 (the README recommends 4.57.1): the bundled flash-linear-attention cache increments its token counter at layer 0, so layers 1-35 read a position one step ahead. The fix already exists in fla-org/flash-linear-attention (PR #748). Measured effect: [docs/VALIDATION.md](docs/VALIDATION.md#scope-of-this-result).

## What is ported

Features of the released upstream code (commit `0256113`) and their state in this port:

| Upstream feature | In this port |
| --- | --- |
| Hybrid layout: 26 hybrid layers (gated delta net + 128-token sliding-window attention) and 10 softmax-attention layers | Yes, identical (verified layer by layer) |
| Sparse State Expansion (SSE): a router selects 2 of 4 state partitions per token | Yes, identical: only the selected partitions are updated and read |
| Fixed-size recurrent state in the hybrid layers | Yes; it also enables exact prefix reuse |
| Chunked gated delta rule kernels (Triton) | Yes, same chunked (WY) algorithm rewritten in PyTorch |
| MoBA sparse attention (12 chunks of 4096 tokens per query) | Same output, but no MoBA speed-up. Below 49,152 tokens MoBA selects every past chunk, so it equals dense causal attention, which the port computes. Longer contexts, where MoBA would save work, are rejected. See [PORTING_NOTES](docs/PORTING_NOTES.md#moba-equals-causal-attention-below-a-limit). |
| Spiking / event-driven activation coding (INT8-Spiking) | Not available: it is described in the upstream paper and README, but the released code contains no implementation of it. The upstream GPU code also computes every activation. |
| Optional FP8 KV cache (vLLM `cache_dtype` setting) | No. fp32 caches |
| Vision-language model, vLLM plugin, training | No |

## What you get

* `spikingbrain_cpu/`: model, weight loader, chat template, generation engine, benchmark tools and an OpenAI-compatible server (`/v1/chat/completions`, `/v1/models`, `/health`).
* Five weight formats: `float16` (recommended), `int8`, `bfloat16`, `float32` and the experimental `int8-dyn`.
* Exact prefix reuse: a request that extends the previous conversation only computes the new tokens.
* `weights-manifest.json`: the pinned checkpoint (ModelScope `Panyuqi/SpikingBrain-2.0-instruct`, revision `866364cb`) with SHA-256 for every file, plus a downloader that verifies them.
* A Containerfile and Kubernetes manifests (non-root, read-only root filesystem, no external route).
* `tools/upstream-reference/`: the harness that runs the original upstream code on CPU (Triton interpreter mode, with the substitutions listed in [docs/VALIDATION.md](docs/VALIDATION.md)) and compares it with this port.

## Expected performance

Measured on one 8 vCPU x86-64 virtual machine with AVX-512 and 32 GiB RAM, using 4 PyTorch threads, batch size 1 and PyTorch 2.14.1+cpu. Details, conditions and caveats are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| Weight format | Load time | Peak memory (RSS) | Prefill | Decode | Mean KL vs bf16 |
| --- | --- | --- | --- | --- | --- |
| `float16` (default) | ~110 s | ~13 GiB | ~40 tokens/s | ~3 tokens/s | 0.00009 |
| `int8` | ~35 s | ~8 GiB | ~36 tokens/s | ~1.5 tokens/s | 0.00076 |
| `int8-dyn` | not measured | not measured | not measured | not measured | 0.17 (not recommended) |

What this means in practice with `float16`:

* A 2,000-token prompt takes about 50 seconds before the first output token. A 6,000-token prompt takes about 2.5 minutes.
* A 200-token answer takes about 65 seconds to generate.
* In an agent loop that resends the whole conversation, only the new tokens are computed, so later turns are much faster than the first one.
* Use `int8` when memory is tight. It needs about 5 GiB less, but generates at about half the speed.

This is slow inference for one user at a time. It suits offline text generation, summarisation and experiments. It does not suit chat traffic with many concurrent users, and it is not validated for tool-calling agents (see [Known limitations](#known-limitations)).

## Quick start

Build the image and download the checkpoint (about 11 GB):

```sh
podman build -t spikingbrain-cpu -f Containerfile .
mkdir -p models
podman run --rm --userns=keep-id -v "$PWD/models:/models:Z" --entrypoint python3 spikingbrain-cpu \
  -m spikingbrain_cpu.fetch /opt/spikingbrain/weights-manifest.json /models/spikingbrain
```

Start the server and send a request:

```sh
podman run --rm --userns=keep-id -p 8080:8080 --memory 16g -v "$PWD/models:/models:ro,Z" spikingbrain-cpu
# wait until /health answers (loading takes about 2 minutes)
curl -s localhost:8080/health
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "SpikingBrain-2.0-5B-instruct",
  "messages": [{"role": "user", "content": "Explain what a Kubernetes Pod is in two sentences."}],
  "max_tokens": 128
}'
```

Without containers:

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m spikingbrain_cpu.fetch weights-manifest.json models/spikingbrain
SB_MODEL_DIR=models/spikingbrain python -m spikingbrain_cpu.server
```

For Kubernetes or OpenShift, see `kubernetes/weights.yaml` (volume and verified download) and `kubernetes/server.yaml` (server, Service and NetworkPolicy). Replace `REPLACE_WITH_IMAGE_DIGEST` with your image digest.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SB_MODEL_DIR` | `/models/spikingbrain` | checkpoint directory |
| `SB_MODEL_NAME` | `SpikingBrain-2.0-5B-instruct` | model id reported by the API |
| `SB_WEIGHT_FORMAT` | `float16` | `float16`, `int8`, `bfloat16`, `float32` or `int8-dyn` |
| `SB_THREADS` | `4` | PyTorch intra-op threads |
| `SB_MAX_CONTEXT` | `24576` | prompt plus completion token limit |
| `SB_MAX_TOKENS` | `2048` | upper bound for `max_tokens` |
| `SB_TOOL_CALL_RECOVERY` | `0` | `1` accepts bare JSON calls to declared tools (experimental) |
| `SB_HOST`, `SB_PORT` | `0.0.0.0`, `8080` | listen address |

The server handles one generation at a time. Other requests wait in order. Streaming is not implemented.

## Known limitations

* **Tool calling is weak, and the cause is the model, not this port.** The upstream implementation gives the `<tool_call>` token a probability below 2 % (0.8-2 % in our probes) at the start of an answer. The model often writes the token `agger` instead and then a valid JSON call. With `"tool_choice": "required"` the server forces the `<tool_call>` prefix and calls are reliable. With `"auto"`, expect missed calls. `SB_TOOL_CALL_RECOVERY=1` accepts bare JSON calls to declared tools. It is an experimental workaround, not a fix.
* Contexts longer than 49,152 tokens are rejected. Below that limit, MoBA attention is exactly causal attention, so the port does not approximate it. See [docs/PORTING_NOTES.md](docs/PORTING_NOTES.md).
* Only the text model is supported. The vision-language model, training and batched requests are not ported.
* No spiking computation. The released upstream code has no spiking path, so there is nothing to port. Both upstream and this port compute every activation, and the neuromorphic efficiency claims of the upstream project do not apply to them. A dense CPU would not benefit from spike sparsity anyway: it computes all matrix entries.

## Tests

The unit tests need no checkpoint (they use small random models):

```sh
python -m unittest discover -s tests -v
```

They check the chunked delta rule against the recurrent form, the sparse-state-expansion routing against a restatement of the upstream sorting structure, prefill against token-by-token decoding, snapshots, prefix reuse and the checkpoint key layout.

## Licence

MIT, see [LICENSE](LICENSE). Attribution for the upstream projects is in [NOTICE](NOTICE). Check the licence of the model weights before use: the upstream GitHub repository states MIT, while the ModelScope model card states Apache License 2.0.
