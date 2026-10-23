"""Single-slot generation engine with recurrent-state prefix reuse.

Hybrid linear-attention models carry a fixed-size recurrent state, so a
conversation can be continued without re-reading its history: when a new
prompt extends the tokens already processed (or a saved snapshot taken at the
end of an earlier prompt), only the new suffix is computed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch

from .model import InferenceState, SpikingBrainForCausalLM, restore, snapshot


@dataclass
class SamplingParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    seed: int | None = None


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    if params.temperature <= 0:
        return int(logits.argmax())
    logits = logits / params.temperature
    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, -1)
    if 0 < params.top_p < 1:
        sorted_probs, order = probs.sort(descending=True)
        keep = sorted_probs.cumsum(0) - sorted_probs < params.top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
    return int(torch.multinomial(probs / probs.sum(), 1, generator=generator))


class Engine:
    def __init__(self, model: SpikingBrainForCausalLM, stop_ids: set[int], max_context: int,
                 max_snapshots: int = 2, prefill_chunk: int = 512):
        if max_context > model.config.exact_context_limit:
            raise ValueError("max_context above the exact MoBA-equivalent limit")
        self.model = model
        self.stop_ids = stop_ids
        self.max_context = max_context
        self.max_snapshots = max_snapshots
        self.prefill_chunk = prefill_chunk
        self.state: InferenceState = model.new_state()
        self.snapshots = []
        self.lock = threading.Lock()

    def _common(self, a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _position_for(self, tokens: list[int]) -> str:
        """Bring the state to a prefix of ``tokens`` and report how."""
        live = self.state.tokens
        if len(live) < len(tokens) and self._common(live, tokens) == len(live):
            return "live" if live else "empty"
        best = None
        for snap in self.snapshots:
            n = len(snap.tokens)
            if n < len(tokens) and self._common(snap.tokens, tokens) == n:
                if best is None or n > len(best.tokens):
                    best = snap
        if best is not None:
            restore(self.state, best)
            return "snapshot"
        self.state = self.model.new_state()
        return "reset"

    def _remember(self) -> None:
        self.snapshots.append(snapshot(self.state))
        del self.snapshots[:-self.max_snapshots]

    def generate(self, prompt: list[int], params: SamplingParams, forced: list[int] | None = None) -> dict:
        forced = forced or []
        if len(prompt) + len(forced) + params.max_tokens > self.max_context:
            raise ContextError(len(prompt), params.max_tokens, self.max_context)
        with self.lock:
            started = time.monotonic()
            reuse = self._position_for(prompt)
            reused = len(self.state.tokens)
            logits = self.model.forward_tokens(prompt[reused:], self.state, self.prefill_chunk)
            self._remember()
            if forced:
                logits = self.model.forward_tokens(forced, self.state, self.prefill_chunk)
            prefill_done = time.monotonic()
            generator = torch.Generator()
            generator.manual_seed(params.seed if params.seed is not None else torch.seed())
            output: list[int] = []
            finish = "length"
            while len(output) < params.max_tokens:
                token = sample(logits, params, generator)
                logits = self.model.forward_tokens([token], self.state)
                if token in self.stop_ids:
                    finish = "stop"
                    break
                output.append(token)
            ended = time.monotonic()
            computed = len(prompt) - reused + len(forced)
            return {
                "tokens": output,
                "finish_reason": finish,
                "timings": {
                    "prefix_reuse": reuse,
                    "prompt_tokens": len(prompt),
                    "prompt_tokens_reused": reused,
                    "prompt_tokens_computed": computed,
                    "prefill_seconds": round(prefill_done - started, 3),
                    "prefill_tokens_per_second": round(computed / max(prefill_done - started, 1e-9), 2),
                    "completion_tokens": len(output),
                    "decode_seconds": round(ended - prefill_done, 3),
                    "decode_tokens_per_second": round(len(output) / max(ended - prefill_done, 1e-9), 2),
                },
            }


class ContextError(ValueError):
    def __init__(self, prompt: int, max_tokens: int, limit: int):
        super().__init__(f"prompt of {prompt} tokens plus max_tokens {max_tokens} exceeds context {limit}")
