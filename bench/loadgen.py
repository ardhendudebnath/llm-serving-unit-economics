"""Open-loop load generator with a Poisson arrival process.

    python -m bench.loadgen --profile long_in --rate 2.0 --duration 120

**Open-loop, not closed-loop, and this is the whole design.** A closed-loop
generator sends its next request only once the previous one comes back, so when
the server slows down the generator slows down with it. It therefore cannot
saturate anything, and the latency it reports is the latency of a server that
was never actually under the load being claimed. This is coordinated omission,
and it is the most common way a serving benchmark reports numbers that are
quietly fiction.

So requests here are fired on a schedule computed in advance, whether or not
earlier ones have returned. When the server falls behind, the queue grows and
the latency distribution shows it -- which is the measurement.

Arrivals are **Poisson**: exponentially distributed gaps at the requested mean
rate. Real request traffic is bursty, and a uniform every-500ms generator
produces a batching pattern no production deployment ever sees. Poisson is not
a perfect model of real traffic either -- it has no diurnal cycle and no
correlated bursts -- and the report says so under Limitations rather than
implying the arrival pattern was solved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from bench import workloads
from bench.metrics import LoadPoint, RequestRecord

try:
    import httpx
except ImportError:  # pragma: no cover - exercised by the import guard in CI
    httpx = None  # type: ignore[assignment]

#: Ceiling on requests in flight. Not a tuning knob -- it is a safety valve so
#: a saturated server cannot exhaust client sockets and make the client look
#: like the server. Hitting it turns the generator partially closed-loop, so
#: `LoadPoint.worst_client_lag_s` will rise and the run is reported as suspect.
DEFAULT_MAX_INFLIGHT = 512

#: Requests are streamed so time-to-first-token can be measured at all. TTFT is
#: what a user perceives as responsiveness, and it moves quite differently from
#: total latency once batching kicks in.
STREAM = True


@dataclass(slots=True)
class RunConfig:
    base_url: str
    model: str
    profile: str
    rate_rps: float
    duration_s: float
    warmup_s: float = 15.0
    seed: int = 0
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    timeout_s: float = 600.0


async def _one_request(
    client: httpx.AsyncClient,
    cfg: RunConfig,
    req: workloads.Request,
    queued_at: float,
    sem: asyncio.Semaphore,
) -> RequestRecord:
    """Fire one streamed request and time it.

    The semaphore is acquired *before* `sent_at` is stamped, so any wait for a
    free slot lands in `coordinated_omission_s` rather than vanishing. A
    generator that hid this would report the server as faster than it is.
    """
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    payload = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        # Greedy. Sampling noise would add variance to output length, and
        # output length drives decode time -- so a sampled run measures the
        # sampler as much as the server.
        "temperature": 0.0,
        "stream": STREAM,
        # vLLM emits a final usage chunk when asked. Without this, token counts
        # would have to be inferred from chunk counts, which is wrong for any
        # tokenizer that emits multi-token chunks.
        "stream_options": {"include_usage": True},
    }

    async with sem:
        sent_at = time.perf_counter()
        first_token_at: float | None = None
        tokens_in = tokens_out = 0
        status = 0
        error: str | None = None

        try:
            async with client.stream(
                "POST",
                f"{cfg.base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                timeout=cfg.timeout_s,
            ) as resp:
                status = resp.status_code
                if status >= 300:
                    body = (await resp.aread())[:200].decode("utf-8", "replace")
                    error = f"http_{status}: {body}"
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        blob = line[6:].strip()
                        if blob == "[DONE]":
                            break
                        try:
                            chunk = json.loads(blob)
                        except json.JSONDecodeError:
                            continue

                        # First chunk carrying actual content, not the first
                        # chunk overall -- vLLM sends a role-only delta first,
                        # and counting that as the first token would report a
                        # TTFT that no user experiences.
                        choices = chunk.get("choices") or []
                        if choices and first_token_at is None:
                            delta = choices[0].get("delta") or {}
                            if delta.get("content"):
                                first_token_at = time.perf_counter()

                        if usage := chunk.get("usage"):
                            tokens_in = usage.get("prompt_tokens", 0) or 0
                            tokens_out = usage.get("completion_tokens", 0) or 0
        except Exception as exc:  # noqa: BLE001 - a failed request is a data point
            error = f"{type(exc).__name__}: {exc}"

        done_at = time.perf_counter()

    return RequestRecord(
        profile=req.profile,
        queued_at=queued_at,
        sent_at=sent_at,
        done_at=done_at,
        first_token_at=first_token_at,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        status=status,
        error=error,
    )


async def run_point(cfg: RunConfig, corpus: list[workloads.Request]) -> LoadPoint:
    """Drive one arrival rate for `duration_s` and collect every record.

    Warmup runs first and is discarded. The first requests against a cold vLLM
    server pay for CUDA graph capture and an empty prefix cache, and folding
    that into the measurement would make low rates look worse than they are.
    """
    if httpx is None:
        raise RuntimeError("the load generator needs httpx: pip install -e '.[load]'")

    rng = random.Random(cfg.seed)
    sem = asyncio.Semaphore(cfg.max_inflight)

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=cfg.max_inflight + 32)
    ) as client:
        if cfg.warmup_s > 0:
            await _drive(client, cfg, corpus, sem, rng, cfg.warmup_s, collect=False)

        start = time.perf_counter()
        records = await _drive(
            client, cfg, corpus, sem, rng, cfg.duration_s, collect=True, start=start
        )
        elapsed = time.perf_counter() - start

    return LoadPoint(
        profile=cfg.profile,
        target_rate_rps=cfg.rate_rps,
        duration_s=elapsed,
        records=records,
    )


async def _drive(
    client: httpx.AsyncClient,
    cfg: RunConfig,
    corpus: list[workloads.Request],
    sem: asyncio.Semaphore,
    rng: random.Random,
    duration_s: float,
    *,
    collect: bool,
    start: float | None = None,
) -> list[RequestRecord]:
    """Schedule Poisson arrivals for `duration_s` and await them all.

    The schedule is advanced by an exponential gap *before* sleeping, so the
    arrival times are a genuine Poisson process rather than a rate-limited
    loop. Crucially the loop never waits on a response -- that is what keeps
    this open-loop.
    """
    base = start if start is not None else time.perf_counter()
    tasks: list[asyncio.Task] = []
    scheduled = 0.0
    index = 0

    while True:
        # Exponential inter-arrival gap: the defining property of a Poisson
        # process. rate_rps is the mean, not the fixed spacing.
        scheduled += rng.expovariate(cfg.rate_rps)
        if scheduled > duration_s:
            break

        now = time.perf_counter() - base
        if scheduled > now:
            await asyncio.sleep(scheduled - now)

        req = corpus[index % len(corpus)]
        index += 1
        tasks.append(
            asyncio.create_task(
                _one_request(client, cfg, req, queued_at=base + scheduled, sem=sem)
            )
        )

    if not tasks:
        return []

    records = await asyncio.gather(*tasks)
    return list(records) if collect else []


def _write(point: LoadPoint, out_dir: Path, cfg: RunConfig) -> Path:
    """Persist the summary, and the raw per-request trace alongside it.

    The trace is git-ignored and the summary is committed. Re-running a sweep
    because a percentile needs recomputing is money set on fire, so everything
    needed to recompute one is written the first time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{stamp}_{cfg.profile}_r{cfg.rate_rps:g}"

    summary = point.as_row() | {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "seed": cfg.seed,
        "warmup_s": cfg.warmup_s,
        "max_inflight": cfg.max_inflight,
        "arrival": "poisson",
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / f"{name}.ndjson").open("w", encoding="utf-8") as fh:
        for r in point.records:
            fh.write(json.dumps({
                "profile": r.profile,
                "queued_at": r.queued_at,
                "sent_at": r.sent_at,
                "first_token_at": r.first_token_at,
                "done_at": r.done_at,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "status": r.status,
                "error": r.error,
            }) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True, choices=sorted(workloads.PROFILES))
    ap.add_argument("--rate", type=float, required=True, help="mean arrivals/sec")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="", help="defaults to the server's only model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results", type=Path)
    args = ap.parse_args()

    corpus = workloads.load(args.profile)
    model = args.model or discover_model(args.base_url)

    cfg = RunConfig(
        base_url=args.base_url,
        model=model,
        profile=args.profile,
        rate_rps=args.rate,
        duration_s=args.duration,
        warmup_s=args.warmup,
        seed=args.seed,
    )

    print(f"  {args.profile} @ {args.rate:g} rps for {args.duration:g}s "
          f"(warmup {args.warmup:g}s, poisson arrivals, open-loop)")
    point = asyncio.run(run_point(cfg, corpus))

    dist = point.total_latency()
    if dist is None:
        print("  every request failed -- nothing to summarise")
        return 1

    path = _write(point, args.out, cfg)
    print(f"    achieved {point.achieved_rps:.2f} rps · "
          f"p50 {dist.p50:.2f}s · p95 {dist.p95:.2f}s · p99 {dist.p99:.2f}s · "
          f"errors {point.error_rate:.1%}")
    if point.worst_client_lag_s() > 0.5:
        print(f"    warning: client lagged its own schedule by "
              f"{point.worst_client_lag_s():.2f}s -- latency here is optimistic")
    print(f"    -> {path}")
    return 0


def discover_model(base_url: str) -> str:
    """Ask the server what it is serving, rather than making the user repeat it."""
    if httpx is None:
        raise RuntimeError("the load generator needs httpx: pip install -e '.[load]'")
    resp = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=30.0)
    resp.raise_for_status()
    data = resp.json().get("data") or []
    if not data:
        raise RuntimeError(f"{base_url} reports no models")
    return data[0]["id"]


if __name__ == "__main__":
    raise SystemExit(main())
