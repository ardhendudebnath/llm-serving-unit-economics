"""A fake OpenAI-compatible streaming server, for testing the load generator.

The load generator is the instrument every published latency number comes from.
Shipping it untested and discovering a bug on rented GPU time would be the
expensive way to find out it counts the wrong chunk as the first token.

So this stands in for vLLM: same wire format, same SSE framing, same role-only
first delta and trailing usage chunk, but with latency that is *specified*
rather than measured. That inversion is what makes assertions possible -- when
the server is told to take 200 ms to first token, the generator must report
close to 200 ms, and any gap is the instrument's error.

Deliberately stdlib-only and deliberately not a fixture in a test file: it is
also useful by hand for driving the sweep end to end without a GPU.

    python -m tests.mock_server --port 8099 --ttft-ms 100
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self

MODEL_ID = "mock/test-model"


@dataclass
class Behaviour:
    """How the fake server should behave. All times in milliseconds."""

    ttft_ms: float = 50.0
    inter_token_ms: float = 5.0
    tokens_out: int = 20
    #: Requests served at once. Beyond this, requests queue -- which is how a
    #: saturated server is simulated without needing a real one.
    concurrency: int = 8
    #: Fraction of requests answered with HTTP 500, to exercise error paths.
    error_rate: float = 0.0
    prompt_tokens: int = 400


class _Handler(BaseHTTPRequestHandler):
    behaviour: Behaviour
    gate: threading.Semaphore
    served: int
    lock: threading.Lock

    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/v1/models":
            self._json(404, {"error": "not found"})
            return
        self._json(200, {"object": "list", "data": [{"id": MODEL_ID}]})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        with type(self).lock:
            type(self).served += 1
            index = type(self).served

        b = type(self).behaviour
        # Deterministic every-Nth failure rather than random, so a test that
        # asserts an error rate is not flaky. index is 1-based.
        if b.error_rate > 0 and (index % 100) < round(b.error_rate * 100):
            self._json(500, {"error": "synthetic failure"})
            return

        # The semaphore is what makes this behave like a real server under
        # load: past `concurrency`, requests wait, and the generator's recorded
        # latency grows exactly as it would against a saturated vLLM.
        with type(self).gate:
            self._stream(b, int(body.get("max_tokens") or b.tokens_out))

    def _stream(self, b: Behaviour, max_tokens: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # vLLM sends a role-only delta before any content. The generator must
        # NOT count this as the first token -- if it does, every reported TTFT
        # is optimistic by the whole prefill.
        self._chunk({"choices": [{"index": 0, "delta": {"role": "assistant"},
                                  "finish_reason": None}]})

        time.sleep(b.ttft_ms / 1000.0)

        n = max(1, min(max_tokens, b.tokens_out))
        for i in range(n):
            if i:
                time.sleep(b.inter_token_ms / 1000.0)
            self._chunk({"choices": [{"index": 0, "delta": {"content": "tok "},
                                      "finish_reason": None}]})

        self._chunk({"choices": [{"index": 0, "delta": {},
                                  "finish_reason": "length"}]})
        # Trailing usage chunk, as vLLM emits with stream_options.include_usage.
        self._chunk({
            "choices": [],
            "usage": {
                "prompt_tokens": b.prompt_tokens,
                "completion_tokens": n,
                "total_tokens": b.prompt_tokens + n,
            },
        })
        self._sse(b"data: [DONE]\n\n")
        self._end_chunks()

    def _chunk(self, payload: dict) -> None:
        payload = {"id": "mock", "object": "chat.completion.chunk",
                   "model": MODEL_ID, **payload}
        self._sse(f"data: {json.dumps(payload)}\n\n".encode())

    def _sse(self, data: bytes) -> None:
        """One SSE event as one HTTP chunk.

        Flushed immediately: buffering several events together would deliver
        the first content delta at the same instant as the last, and the
        generator's time-to-first-token measurement would be meaningless.
        """
        self._write(f"{len(data):X}\r\n".encode() + data + b"\r\n")

    def _end_chunks(self) -> None:
        self._write(b"0\r\n\r\n")

    def _write(self, raw: bytes) -> None:
        try:
            self.wfile.write(raw)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # The generator abandoning a request mid-stream is a normal thing
            # to simulate, not a server error.
            pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class MockServer:
    """Runs the fake server on a background thread.

        with MockServer(Behaviour(ttft_ms=200)) as server:
            ...drive server.base_url...
    """

    def __init__(self, behaviour: Behaviour | None = None, port: int = 0):
        self.behaviour = behaviour or Behaviour()

        handler = type("BoundHandler", (_Handler,), {
            "behaviour": self.behaviour,
            "gate": threading.Semaphore(self.behaviour.concurrency),
            "served": 0,
            "lock": threading.Lock(),
        })
        self._handler = handler
        # Port 0 lets the OS pick a free one, so parallel test runs cannot
        # collide on a hardcoded port.
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def served(self) -> int:
        return self._handler.served

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ttft-ms", type=float, default=50.0)
    ap.add_argument("--inter-token-ms", type=float, default=5.0)
    ap.add_argument("--tokens-out", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    behaviour = Behaviour(
        ttft_ms=args.ttft_ms, inter_token_ms=args.inter_token_ms,
        tokens_out=args.tokens_out, concurrency=args.concurrency,
    )
    with MockServer(behaviour, port=args.port) as server:
        print(f"  mock vLLM on {server.base_url} "
              f"(ttft {args.ttft_ms:g}ms, {args.concurrency} concurrent)")
        print("  ctrl-c to stop")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print(f"\n  served {server.served} request(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
