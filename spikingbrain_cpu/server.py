"""Minimal OpenAI-compatible chat completions server for CPU inference.

Endpoints: ``GET /health``, ``GET /v1/models``, ``POST /v1/chat/completions``
(non-streaming). One request is generated at a time; others wait in order.

Configuration (environment variables):
  SB_MODEL_DIR      checkpoint directory (config.json, tokenizer files, shards)
  SB_MODEL_NAME     served model id                     [SpikingBrain-2.0-5B-instruct]
  SB_WEIGHT_FORMAT  float16 | int8 | bfloat16 | float32 | int8-dyn [float16]
  SB_THREADS        torch intra-op threads               [4]
  SB_MAX_CONTEXT    prompt + completion token limit      [24576]
  SB_MAX_TOKENS     upper bound for max_tokens           [2048]
  SB_TOOL_CALL_RECOVERY  1 = accept bare JSON calls to declared tools  [0]
  SB_HOST, SB_PORT  listen address                       [0.0.0.0, 8080]
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .chat import REQUIRED_TOOL_PREFIX, ChatTemplate, parse_tool_calls
from .engine import ContextError, Engine, SamplingParams

log = logging.getLogger("spikingbrain_cpu")
MAX_BODY = 4 * 1024 * 1024


class Service:
    def __init__(self):
        self.name = os.environ.get("SB_MODEL_NAME", "SpikingBrain-2.0-5B-instruct")
        self.model_dir = Path(os.environ.get("SB_MODEL_DIR", "/models/spikingbrain"))
        self.fmt = os.environ.get("SB_WEIGHT_FORMAT", "float16")
        self.max_context = int(os.environ.get("SB_MAX_CONTEXT", "24576"))
        self.max_tokens = int(os.environ.get("SB_MAX_TOKENS", "2048"))
        # Opt-in format adapter for checkpoints that emit a wrong token instead
        # of <tool_call> around a well-formed call (see chat._bare_calls).
        self.recover_bare_calls = os.environ.get("SB_TOOL_CALL_RECOVERY", "0") == "1"
        self.engine: Engine | None = None
        self.error: str | None = None
        self.load_report: dict = {}

    def load(self) -> None:
        try:
            from tokenizers import Tokenizer

            from .loader import load_model

            torch.set_num_threads(int(os.environ.get("SB_THREADS", "4")))
            tok_cfg = json.loads((self.model_dir / "tokenizer_config.json").read_text())
            self.template = ChatTemplate(tok_cfg["chat_template"])
            self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            model, report = load_model(self.model_dir, self.fmt)
            stop = {model.config.eos_token_id}
            for text in ("<|im_end|>", "<|endoftext|>"):
                token = self.tokenizer.token_to_id(text)
                if token is not None:
                    stop.add(token)
            self.forced = self.tokenizer.encode(REQUIRED_TOOL_PREFIX, add_special_tokens=False).ids
            self.load_report = {**report, "threads": torch.get_num_threads(),
                                "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                                "torch": torch.__version__}
            self.engine = Engine(model, stop, self.max_context)
            log.info("model ready %s", json.dumps(self.load_report))
        except Exception as exc:  # reported through /health
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model load failed")

    def complete(self, body: dict) -> dict:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BadRequest("messages must be a non-empty list")
        tools = body.get("tools") or None
        choice = body.get("tool_choice", "auto")
        if choice == "none":
            tools = None
        kwargs = body.get("chat_template_kwargs") or {}
        prompt_text = self.template.render(messages, tools, **kwargs)
        prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False).ids
        requested = body.get("max_tokens") or body.get("max_completion_tokens") or 512
        params = SamplingParams(
            max_tokens=max(1, min(int(requested), self.max_tokens)),
            temperature=float(body.get("temperature", 0.7)),
            top_p=float(body.get("top_p", 0.9)),
            top_k=int(body.get("top_k", 0)),
            seed=body.get("seed"),
        )
        forced = self.forced if tools and choice == "required" else []
        result = self.engine.generate(prompt, params, forced)
        text = self.tokenizer.decode(forced + result["tokens"], skip_special_tokens=False)
        names = {t.get("function", t).get("name") for t in tools or []} - {None}
        content, calls, recovered = parse_tool_calls(text, names if self.recover_bare_calls else None)
        finish = result["finish_reason"]
        if calls and finish == "stop":
            finish = "tool_calls"
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        completion = len(forced) + len(result["tokens"])
        return {
            "id": "chatcmpl-" + secrets.token_hex(12),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": completion,
                      "total_tokens": len(prompt) + completion},
            "timings": result["timings"],
            "tool_call_recovery": recovered,
        }


class BadRequest(ValueError):
    pass


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str, kind: str) -> None:
            self._send(status, {"error": {"code": status, "message": message, "type": kind}})

        def do_GET(self):
            if self.path == "/health":
                if service.engine is not None:
                    self._send(200, {"status": "ok", "model": service.name, **service.load_report})
                elif service.error:
                    self._error(500, service.error, "load_error")
                else:
                    self._error(503, "Loading model", "unavailable_error")
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": service.name, "object": "model", "owned_by": "local"}]})
            else:
                self._error(404, "not found", "not_found_error")

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._error(404, "not found", "not_found_error")
                return
            if service.engine is None:
                self._error(503, "Loading model", "unavailable_error")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._error(413 if length > MAX_BODY else 400, "invalid body size", "invalid_request_error")
                return
            try:
                body = json.loads(self.rfile.read(length))
                if body.get("stream"):
                    raise BadRequest("streaming is not supported")
                self._send(200, service.complete(body))
            except (BadRequest, ContextError, json.JSONDecodeError, TypeError, ValueError) as exc:
                self._error(400, str(exc), "invalid_request_error")
            except Exception as exc:
                log.exception("generation failed")
                self._error(500, f"{type(exc).__name__}: {exc}", "server_error")

    return Handler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    service = Service()
    threading.Thread(target=service.load, daemon=True).start()
    host = os.environ.get("SB_HOST", "0.0.0.0")
    port = int(os.environ.get("SB_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), make_handler(service))
    log.info("listening on %s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
