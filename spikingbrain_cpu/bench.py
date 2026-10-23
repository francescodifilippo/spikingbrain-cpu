"""Measure load time, memory, prefill and decode speed on the real checkpoint.

Usage: python -m spikingbrain_cpu.bench MODEL_DIR [FORMAT ...]
       python -m spikingbrain_cpu.bench fidelity MODEL_DIR REFERENCE_FORMAT FORMAT ...

Prints one JSON line per measurement. Generated text is printed raw as an
observation; it is not scored.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

from .chat import REQUIRED_TOOL_PREFIX, ChatTemplate, parse_tool_calls
from .engine import Engine, SamplingParams
from .loader import load_model

TOOLS = [
    {"type": "function", "function": {
        "name": "pods_list_in_namespace",
        "description": "List the Kubernetes pods in the specified namespace",
        "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}}, "required": ["namespace"]}}},
    {"type": "function", "function": {
        "name": "events_list",
        "description": "List Kubernetes events (warnings, errors, state changes) in a namespace or cluster-wide",
        "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "query_loki_logs",
        "description": "Run a LogQL query against a Loki datasource and return log lines",
        "parameters": {"type": "object", "properties": {
            "datasourceUid": {"type": "string"}, "logql": {"type": "string"},
            "startRfc3339": {"type": "string"}, "endRfc3339": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["datasourceUid", "logql"]}}},
]


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def emit(**kwargs) -> None:
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def run(model_dir: Path, fmt: str) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template = ChatTemplate(json.loads((model_dir / "tokenizer_config.json").read_text())["chat_template"])
    model, report = load_model(model_dir, fmt)
    emit(step="load", format=fmt, rss_mib=round(rss_mib()), threads=torch.get_num_threads(),
         cpu_capability=torch.backends.cpu.get_cpu_capability(), **report)
    stop = {model.config.eos_token_id, tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")} - {None}
    engine = Engine(model, stop, max_context=24576)

    # raw prefill speed on a long synthetic prompt
    for length in (512, 2048, 6144):
        state = model.new_state()
        ids = (tok.encode("Kubernetes cluster event log line. " * 400).ids * 4)[:length]
        started = time.monotonic()
        model.forward_tokens(ids, state)
        seconds = time.monotonic() - started
        emit(step="prefill", format=fmt, tokens=length, seconds=round(seconds, 2),
             tokens_per_second=round(length / seconds, 2), rss_mib=round(rss_mib()))

    cases = [
        ("chat", [{"role": "system", "content": "You are a technical assistant. Answer in English."},
                  {"role": "user", "content": "Explain in three sentences what a Pod is in Kubernetes."}], None, "auto"),
        ("tool_auto", [{"role": "system", "content": "You inspect a Kubernetes cluster with tools. Read-only."},
                       {"role": "user", "content": "Which pods are running in namespace monitoring?"}], TOOLS, "auto"),
        ("tool_required", [{"role": "system", "content": "You inspect a Kubernetes cluster with tools. Read-only."},
                           {"role": "user", "content": "Find recent warning events in namespace payments."}], TOOLS, "required"),
    ]
    forced_ids = tok.encode(REQUIRED_TOOL_PREFIX, add_special_tokens=False).ids
    for name, messages, tools, choice in cases:
        prompt = tok.encode(template.render(messages, tools), add_special_tokens=False).ids
        forced = forced_ids if choice == "required" else []
        result = engine.generate(prompt, SamplingParams(max_tokens=160, temperature=0.0), forced)
        text = tok.decode(forced + result["tokens"], skip_special_tokens=False)
        names = {t["function"]["name"] for t in tools or []}
        content, calls, recovered = parse_tool_calls(text, names)
        emit(step="generate", format=fmt, case=name, raw_text=text,
             tool_calls=[c["function"] for c in calls], tool_call_recovery=recovered,
             finish=result["finish_reason"],
             rss_mib=round(rss_mib()), **result["timings"])
    del engine, model
    gc.collect()


FIDELITY_TEXT = (
    "Analyse the cluster events. Pod payments-api-7f9c in namespace payments is in "
    "CrashLoopBackOff: Back-off restarting failed container, exit code 137 (OOMKilled). "
    "The node worker-3 reports MemoryPressure=True since 10:42 UTC and the kubelet evicted two pods. "
    "Loki shows repeated 'connection refused' errors from the database client between 10:40 and 10:55. "
    "Prometheus: container_memory_working_set_bytes reached the 512Mi limit. "
) * 4


def fidelity(model_dir: Path, reference: str, candidates: list[str]) -> None:
    """Teacher-forced next-token agreement of each format against a reference format."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template = ChatTemplate(json.loads((model_dir / "tokenizer_config.json").read_text())["chat_template"])
    text = template.render([{"role": "user", "content": "Summarise."}]) + FIDELITY_TEXT
    ids = tok.encode(text, add_special_tokens=False).ids[:512]
    model, report = load_model(model_dir, reference)
    emit(step="fidelity_reference", format=reference, tokens=len(ids), rss_mib=round(rss_mib()), **report)
    ref = torch.log_softmax(model.forward_tokens(ids, model.new_state(), all_logits=True), -1)
    del model
    gc.collect()
    targets = torch.tensor(ids[1:])
    for fmt in candidates:
        model, _ = load_model(model_dir, fmt)
        started = time.monotonic()
        got = torch.log_softmax(model.forward_tokens(ids, model.new_state(), all_logits=True), -1)
        seconds = time.monotonic() - started
        kl = (ref.exp() * (ref - got)).sum(-1)
        emit(step="fidelity", reference=reference, format=fmt, tokens=len(ids),
             top1_agreement=round(float((ref.argmax(-1) == got.argmax(-1)).float().mean()), 4),
             mean_kl=round(float(kl.mean()), 5), max_kl=round(float(kl.max()), 4),
             ref_nll=round(float(-ref[:-1].gather(1, targets[:, None]).mean()), 4),
             nll=round(float(-got[:-1].gather(1, targets[:, None]).mean()), 4),
             prefill_seconds=round(seconds, 2), rss_mib=round(rss_mib()))
        del model, got
        gc.collect()


def probe(model_dir: Path, fmt: str) -> None:
    """Show the most likely first tokens of a tool-using answer (diagnostics only)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template = ChatTemplate(json.loads((model_dir / "tokenizer_config.json").read_text())["chat_template"])
    model, _ = load_model(model_dir, fmt)
    prompts = {
        "en_system": [{"role": "system", "content": "You inspect a Kubernetes cluster with tools. Read-only."},
                      {"role": "user", "content": "Which pods are running in namespace monitoring?"}],
        "no_system": [{"role": "user", "content": "Which pods are running in namespace monitoring?"}],
        "terse_system": [{"role": "system", "content": "Read-only cluster assistant. Use the tools."},
                         {"role": "user", "content": "List the pods in namespace monitoring."}],
    }
    tag = tok.token_to_id("<tool_call>")
    for name, messages in prompts.items():
        ids = tok.encode(template.render(messages, TOOLS), add_special_tokens=False).ids
        probs = torch.softmax(model.forward_tokens(ids, model.new_state()), -1)
        top = torch.topk(probs, 10)
        emit(step="probe", case=name, tool_call_token=tag, tool_call_prob=round(float(probs[tag]), 5),
             tool_call_rank=int((probs > probs[tag]).sum()),
             top=[[tok.id_to_token(int(i)), int(i), round(float(p), 4)] for p, i in zip(top.values, top.indices)])


PROSE = (
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined "
    "movable metal type, oil-based ink and a wooden screw press adapted from wine making. Before that, books "
    "were copied by hand, which made them expensive and rare. Within fifty years printing shops had opened in "
    "more than two hundred European cities, and the price of a book fell sharply. Historians still debate how "
    "quickly literacy rose as a result, because surviving records are incomplete and often biased toward wealthy "
    "households. What is clear is that pamphlets, calendars and religious texts reached audiences that had never "
    "owned a book before. Merchants used printed tables to calculate interest and exchange rates, while sailors "
    "relied on printed charts that could be corrected and reissued. Governments soon realised that the new "
    "technology could spread dissent as easily as instruction, and censorship laws followed. In Venice, publishers "
    "such as Aldus Manutius introduced the italic typeface and small portable editions of classical authors, "
    "which students could carry in a pocket. Paper mills multiplied along rivers, because water wheels powered "
    "the hammers that pulped old rags into fibre. The demand for rags became so great that some towns regulated "
    "their collection. Meanwhile, astronomers exchanged printed star catalogues, and errors that once survived "
    "for centuries in manuscripts could now be noticed and fixed in a later edition. Mathematicians benefited in a "
    "similar way: diagrams were reproduced exactly, so a proof read in Paris looked the same as one read in "
    "Krakow. Musicians, too, adopted printed scores, although engraving music remained difficult and costly. By "
    "the end of the sixteenth century the printed word had changed how people argued, worshipped, traded and "
    "learned, and the slow world of the scriptorium had almost disappeared. Today the same questions return with "
    "digital publishing: who controls distribution, how errors are corrected, and whether cheaper access to "
    "information produces wiser readers or simply louder arguments."
)


def positions(model_dir: Path, fmt: str) -> None:
    """Per-position negative log-likelihood on natural, non-repeated prose (diagnostics only)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model, _ = load_model(model_dir, fmt)
    ids = tok.encode(PROSE, add_special_tokens=False).ids
    for chunk in (512, 48):
        logp = torch.log_softmax(model.forward_tokens(ids, model.new_state(), chunk=chunk, all_logits=True)
                                 if chunk >= len(ids) else _all_logits_chunked(model, ids, chunk), -1)
        nll = -logp[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
        buckets = {f"{lo}-{min(lo + 64, len(ids) - 1)}": round(float(nll[lo:lo + 64].mean()), 3)
                   for lo in range(0, len(ids) - 1, 64)}
        emit(step="positions", format=fmt, prefill_chunk=chunk, tokens=len(ids), nll_by_position=buckets,
             mean_nll=round(float(nll.mean()), 3))


def _all_logits_chunked(model, ids, chunk):
    state = model.new_state()
    parts = [model.forward_tokens(ids[lo:lo + chunk], state, chunk=chunk, all_logits=True)
             for lo in range(0, len(ids), chunk)]
    return torch.cat(parts, 0)


def consistency(model_dir: Path, fmt: str, positions_to_check=(127, 128, 129, 255, 256, 4095, 4096)) -> None:
    """Next-token logits after prefill+one decode step vs a single prefill (diagnostics only)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model, _ = load_model(model_dir, fmt)
    ids = tok.encode((PROSE + " " + FIDELITY_TEXT + " ") * 8, add_special_tokens=False).ids
    for p in positions_to_check:
        state = model.new_state()
        model.forward_tokens(ids[:p], state)
        decoded = model.forward_tokens([ids[p]], state)
        prefilled = model.forward_tokens(ids[:p + 1], model.new_state())
        a, b = torch.log_softmax(prefilled, -1), torch.log_softmax(decoded, -1)
        emit(step="consistency", format=fmt, decoded_position=p,
             kl=float((a.exp() * (a - b)).sum()), same_top1=bool(a.argmax() == b.argmax()),
             max_abs_logit_diff=round(float((prefilled - decoded).abs().max()), 5))


def main() -> None:
    torch.set_num_threads(int(os.environ.get("SB_THREADS", "4")))
    if sys.argv[1] == "consistency":
        consistency(Path(sys.argv[2]), sys.argv[3])
        return
    if sys.argv[1] == "positions":
        positions(Path(sys.argv[2]), sys.argv[3])
        return
    if sys.argv[1] == "probe":
        probe(Path(sys.argv[2]), sys.argv[3])
        return
    if sys.argv[1] == "fidelity":
        fidelity(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
        return
    model_dir = Path(sys.argv[1])
    for fmt in sys.argv[2:] or ["float16"]:
        run(model_dir, fmt)


if __name__ == "__main__":
    main()
