import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import naive_attention, naive_recurrent, sse_varlen_reference  # noqa: E402
from spikingbrain_cpu.engine import Engine, SamplingParams  # noqa: E402
from spikingbrain_cpu.model import (  # noqa: E402
    LayerState,
    SpikingBrainForCausalLM,
    attention,
    build_reference_config,
    chunk_gated_delta_rule,
    init_random,
    recurrent_gated_delta_rule,
    restore,
    snapshot,
)

torch.manual_seed(0)


def _gdn_inputs(T, H=3, D=8):
    q = torch.nn.functional.normalize(torch.randn(T, H, D), dim=-1) * D ** -0.5
    k = torch.nn.functional.normalize(torch.randn(T, H, D), dim=-1)
    v = torch.randn(T, H, D)
    g = -torch.rand(T, H) * 0.5
    beta = torch.rand(T, H)
    return q, k, v, g, beta


class GatedDeltaRuleTest(unittest.TestCase):
    def test_chunk_matches_recurrent(self):
        for T in (1, 7, 64, 150):
            q, k, v, g, beta = _gdn_inputs(T)
            h0 = torch.randn(3, 8, 8) * 0.1
            o_r, h_r = recurrent_gated_delta_rule(q, k, v, g, beta, h0.clone())
            o_c, h_c = chunk_gated_delta_rule(q, k, v, g, beta, h0.clone(), chunk_size=16)
            torch.testing.assert_close(o_c, o_r, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(h_c, h_r, atol=1e-4, rtol=1e-4)

    def test_recurrent_matches_kernel_semantics(self):
        q, k, v, g, beta = _gdn_inputs(9)
        raw_q, raw_k = torch.randn(9, 3, 8), torch.randn(9, 3, 8)
        h0 = torch.randn(3, 8, 8) * 0.1
        ref, h_ref = naive_recurrent(raw_q, raw_k, v, g, beta, h0, 8 ** -0.5)
        qn = raw_q / torch.sqrt((raw_q * raw_q).sum(-1, keepdim=True) + 1e-6) * 8 ** -0.5
        kn = raw_k / torch.sqrt((raw_k * raw_k).sum(-1, keepdim=True) + 1e-6)
        out, h = recurrent_gated_delta_rule(qn, kn, v, g, beta, h0.clone())
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h, h_ref, atol=1e-5, rtol=1e-5)


class AttentionTest(unittest.TestCase):
    def test_windowed_and_full_attention(self):
        T = 40
        q, k, v = torch.randn(T, 4, 16), torch.randn(T, 2, 16), torch.randn(T, 2, 16)
        for window in (None, 8):
            ref = naive_attention(q, k, v, window)
            torch.testing.assert_close(attention(q, k, v, 0, window, block=7), ref, atol=1e-5, rtol=1e-5)
            # queries at the end, keys trimmed to what a cache would hold
            keep = T if window is None else window + 5
            got = attention(q[-5:], k[-keep:], v[-keep:], T - 5, window)
            torch.testing.assert_close(got, ref[-5:], atol=1e-5, rtol=1e-5)


class SparseStateExpansionTest(unittest.TestCase):
    def test_routing_matches_upstream_varlen_structure(self):
        cfg = build_reference_config()
        model = SpikingBrainForCausalLM(cfg)
        init_random(model, 1)
        block = model.model.layers[1]
        self.assertTrue(block.hybrid)
        layer = model.new_state().layers[1]
        layer.shared = torch.randn_like(layer.shared) * 0.05
        layer.partitions = torch.randn_like(layer.partitions) * 0.05
        x = torch.randn(30, cfg.hidden_size)
        with torch.no_grad():
            inputs = block.attn.sse_inputs(x)
            ref, shared, parts = sse_varlen_reference(
                *inputs, layer.shared.clone(), layer.partitions.clone(), cfg.num_writer
            )
            out = block.attn.sse_heads(x, layer)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(layer.shared, shared, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(layer.partitions, parts, atol=1e-4, rtol=1e-4)


class IncrementalDecodeTest(unittest.TestCase):
    def setUp(self):
        self.cfg = build_reference_config()
        self.model = SpikingBrainForCausalLM(self.cfg)
        init_random(self.model, 2)
        self.tokens = torch.randint(0, self.cfg.vocab_size, (150,)).tolist()

    def _logits_full(self, tokens, chunk=512):
        return self.model.forward_tokens(tokens, self.model.new_state(), chunk)

    def test_prefill_equals_token_by_token(self):
        state = self.model.new_state()
        for t in self.tokens[:-1]:
            self.model.forward_tokens([t], state)
        step = self.model.forward_tokens(self.tokens[-1:], state)
        full = self._logits_full(self.tokens)
        torch.testing.assert_close(step, full, atol=2e-4, rtol=2e-4)

    def test_chunked_prefill_equals_single_pass(self):
        full = self._logits_full(self.tokens)
        chunked = self._logits_full(self.tokens, chunk=37)
        torch.testing.assert_close(chunked, full, atol=2e-4, rtol=2e-4)

    def test_snapshot_restore(self):
        state = self.model.new_state()
        self.model.forward_tokens(self.tokens[:100], state)
        snap = snapshot(state)
        a = self.model.forward_tokens(self.tokens[100:], state)
        self.model.forward_tokens([1, 2, 3], state)
        restore(state, snap)
        b = self.model.forward_tokens(self.tokens[100:], state)
        torch.testing.assert_close(a, b)

    def test_context_limit_enforced(self):
        with self.assertRaises(ValueError):
            self._logits_full([1] * (self.cfg.exact_context_limit + 1))


class EngineTest(unittest.TestCase):
    def test_prefix_reuse_is_exact(self):
        cfg = build_reference_config()
        model = SpikingBrainForCausalLM(cfg)
        init_random(model, 3)
        engine = Engine(model, stop_ids={0}, max_context=cfg.exact_context_limit)
        params = SamplingParams(max_tokens=5, temperature=0)
        prompt = torch.randint(1, cfg.vocab_size, (60,)).tolist()
        first = engine.generate(prompt, params)
        self.assertEqual(first["timings"]["prefix_reuse"], "empty")
        # follow-up turn: earlier prompt, a different assistant text, new user text
        follow = prompt + [5, 6, 7] + torch.randint(1, cfg.vocab_size, (20,)).tolist()
        second = engine.generate(follow, params)
        self.assertEqual(second["timings"]["prefix_reuse"], "snapshot")
        self.assertEqual(second["timings"]["prompt_tokens_reused"], len(prompt))
        fresh = Engine(model, stop_ids={0}, max_context=cfg.exact_context_limit).generate(follow, params)
        self.assertEqual(second["tokens"], fresh["tokens"])


class LoaderTest(unittest.TestCase):
    def test_upstream_key_layout_and_formats(self):
        from spikingbrain_cpu.loader import load_model

        torch.manual_seed(4)
        cfg = build_reference_config()
        model = SpikingBrainForCausalLM(cfg)
        init_random(model, 4)
        model.lm_head.weight = model.model.embeddings.weight
        state_dict = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        raw = {
            "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
            "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size,
            "head_dim": cfg.head_dim, "num_heads": cfg.num_heads, "norm_eps": cfg.norm_eps,
            "num_sparse_partition": 4, "num_writer": 2, "num_reader": 2, "linear_attn_type": "gdn",
            "attn": {"layers": cfg.attn_layers, "full_layers": cfg.attn_full_layers, "num_heads": 4,
                     "num_kv_heads": 2, "qk_norm": True, "qkv_bias": False, "rope_theta": 10000.0,
                     "window_size": cfg.window_size, "moba_chunk_size": 64, "moba_topk": 4},
        }
        with tempfile.TemporaryDirectory() as tmp:
            import json

            keys = sorted(state_dict)
            half = len(keys) // 2
            shards = {"pytorch_model-00001.bin": keys[:half], "pytorch_model-00002.bin": keys[half:]}
            for name, names in shards.items():
                torch.save({k: state_dict[k] for k in names}, os.path.join(tmp, name))
            weight_map = {k: n for n, names in shards.items() for k in names}
            Path(tmp, "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": weight_map}))
            Path(tmp, "config.json").write_text(json.dumps(raw))
            tokens = torch.randint(0, cfg.vocab_size, (40,)).tolist()
            ref_model, report = load_model(tmp, "float32", torch.float32)
            self.assertTrue(report["stored_lm_head_equals_embeddings"])
            ref = ref_model.forward_tokens(tokens, ref_model.new_state())
            for fmt in ("bfloat16", "float16", "int8", "int8-dyn"):
                loaded, _ = load_model(tmp, fmt)
                got = loaded.forward_tokens(tokens, loaded.new_state())
                cos = torch.nn.functional.cosine_similarity(got, ref, dim=0)
                # random tiny weights are a harsh case for per-tensor activation
                # quantisation; real-checkpoint fidelity is measured by bench.py
                self.assertGreater(float(cos), 0.95 if fmt == "int8-dyn" else 0.99, fmt)


if __name__ == "__main__":
    unittest.main()
