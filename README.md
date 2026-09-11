# Serving an open-weight LLM for GST classification: latency, quality and cost

**What does inference actually cost, and does self-hosting beat the API?**
This repository answers that for one model, one task and one deployment, with
measurements rather than estimates — and publishes the curves.

The task comes from [Project 01](https://github.com/ardhendudebnath/gst-eval-harness):
classifying Indian goods into the GST slab that is *currently* in force, after
two restructures in under two years. That project scores quality. This one adds
latency and cost, so all three sit in the same frame.

> **Status: fp16 measured; int8 and int4 in progress (week 1 of 6).**
>
> The fp16 rung has been served, load-tested on all three workload profiles and
> scored five times on Project 01's harness. Its numbers are below. The int8 and
> int4 rungs, the crossover chart and the headline are still absent rather than
> estimated.
>
> This section will keep saying what is missing until nothing is. Project 01
> holds the same line, and it is the reason its numbers are worth reading.

---

## The headline

*Empty until measured.* It will read like this, with a chart underneath:

> Self-hosting beats the API above **N requests/month** at a p95 target of
> **X ms**, on a *[GPU]* at *[precision]*.

**It may well read the other way**, and that is a real possible outcome rather
than a hedge. The crossover exists only when the API's per-request price is
above the self-hosted marginal cost. A small open-weight model that scores
poorly on this task cannot be compared on price against a frontier API model
that scores well — the honest comparison matches on quality *first*, then
compares cost, and it is entirely possible that nothing self-hostable on one
GPU clears the quality bar. If so, that is the finding and it will be published
as the finding.

---

## Measurements

| | fp16 | int8 | int4 |
|---|---|---|---|
| Latency vs load, three workload profiles | measured | not run | not run |
| Quality on Project 01's harness, five runs | measured | not run | not run |
| GPU utilisation and throttling at each point | measured | not run | not run |
| Cost per 1000 requests, self-hosted vs API | not charted | — | — |

### fp16: capacity at a p95 of 10 s

Qwen3-4B-Instruct-2507 at 16-bit (BF16) on vLLM 0.11.0, one RTX 5070 Ti Laptop
GPU, `max_model_len` 4096, 16 sequences, prefix caching off. Each load point
runs 150 s of Poisson arrivals, with 60 s of idle between points. The knee is
the highest arrival rate whose p95 still meets the SLO.

| Profile | Knee | p95 at the knee | GPU util at the knee | The next rate up |
|---|---:|---:|---:|---|
| `short` | **8 rps** | 1.78 s | 94 % | 16 rps: p95 46.7 s, but the load generator fell 21.9 s behind schedule, so that point measures the client as much as the server |
| `long_in` | **2 rps** | 4.82 s | 90 % | 4 rps: served only 1.65 rps, p95 209.7 s |
| `long_out` | **none** | — | — | even 0.1 rps has a p95 of 22.3 s |

- **`long_in` has no gentle slope.** p95 goes 1.64 → 1.91 → 2.48 → 4.82 s as
  the rate doubles from 0.25 to 2 rps, then 209.7 s at 4 rps. Past the knee,
  the queue takes over completely, so capacity planning has to sit below 2 rps
  rather than near it.
- **`long_out` cannot meet a 10 s total-latency target at any rate.** A lone
  request takes about 20 s, because 768 decode steps on a power-capped card
  cost that at any load. p95 stays flat at 21.9–24.4 s from 0.1 to 0.5 rps and
  reaches 148.9 s at 1 rps. That says more about the target than the server: a
  decode-heavy workload needs a per-token SLO. Read against a 30 s total target,
  the recorded points would put the knee at 0.5 rps. That target was chosen
  after seeing the data, so nothing downstream uses it.
- **The card was power-capped for 84–100 % of busy samples** at every knee
  above. Every throughput figure here is a floor. See Limitations.

### fp16: quality

Five runs of Project 01's harness against the same server, greedy decoding,
28 rows:

| Metric | Mean | Range over five runs |
|---|---:|---:|
| Slab accuracy | 41.4 % | 39.3–42.9 % |
| HSN accuracy | 62.9 % | 60.7–64.3 % |
| Chapter accuracy | 70.0 % | 67.9–71.4 % |
| Abstention accuracy | 100 % | 100 % |
| Stale-slab rate | 5.0 % | 3.6–7.1 % |
| Unparseable | 0 % | 0 % |

Greedy decoding still moved: the runs differ by one row out of 28, which is
3.57 points. vLLM does not guarantee identical output across different batch
compositions. That measured spread is the quality gate's tolerance. For scale,
the frontier reference in Project 01 averages about 54 % on the same rows.

### The three workload profiles

Real request patterns from the domain, not synthetic uniform load. All three
are built from traffic Project 01 collected, and the prompts are rendered by
importing its `harness.prompt` — so a load test replays the exact bytes a
quality run sends.

| Profile | Shape | Median prompt | Output cap | Stands for |
|---|---|---:|---:|---|
| `short` | brief in, brief out | 606 chars | 48 tok | classifying a packaged retail product |
| `long_in` | long in, brief out | 7,177 chars | 48 tok | **Project 01's task** — reading an advance ruling |
| `long_out` | brief in, long out | 446 chars | 768 tok | writing a classification note |

They stress different parts of the server, which is the point of measuring all
three. `long_in` is prefill-bound with a large KV cache per sequence;
`long_out` is decode-bound and is where continuous batching separates most
clearly from static batching; `short` fills the batch with many small
sequences. A test asserts they stay an order of magnitude apart in shape, so
the sweep cannot quietly become one measurement repeated three times.

`short` and `long_out` are sampled from *disjoint* products, so running one
after the other cannot benefit from a prefix cache the other warmed.

---

## What's in the cost figure — and what isn't

Stated before any number exists, so it cannot be written to fit the result.

**In it:** the GPU's hourly cost, meaning amortised purchase price plus
electricity, for every hour of the month. A card costs the same idle as busy.

**Not in it:**

- engineering time — building this took far longer than the GPU it measures
- on-call, and the people to do it
- redundancy: these figures are for **one replica**, which is not a production
  posture. A second for availability doubles the GPU line.
- idle capacity beyond what the peak-to-mean ratio models
- storage, egress, and the control plane
- the cost of being wrong — an API provider absorbs a bad deploy; you do not

**The rate is what an hour of the card costs, not what this project paid.** The
measurements run on a laptop the author already owned, so the cash spent on GPU
time was zero. Reporting the cost as zero would put the crossover at one request
a month and make the chart worthless. Every published figure instead uses the
card's amortised cost: ₹2,49,990 over three years, plus electricity at its 140 W
power limit, priced as available around the clock. That comes to **₹10.63 an
hour**. What the project itself spent is a separate number and lives in
[`docs/cost-log.md`](docs/cost-log.md). Keeping the two apart is deliberate:
one answers "what would this cost to run", the other "what did this cost its
author", and collapsing them answers neither.

`bench/config.py` refuses to produce a cost figure from an unread rate —
`UnpricedError`, not a plausible default.

### Self-hosted cost is a step function

A GPU bills the same idle as saturated, so monthly cost is flat across each
card's capacity and then jumps. The API side is linear through the origin with
no floor. Drawn honestly the self-hosted cost-per-request is a sawtooth: it
falls as a card fills, then jumps when the next one is needed.

A smooth line on that chart means the author divided one number by another and
stopped.

---

## Operations

**Serving.** vLLM, containerised, pinned to an exact tag. Continuous batching
and paged KV cache are the whole reason for using a real serving stack rather
than a Flask wrapper around `model.generate()` — they are what the measurements
are about.

Prefix caching is **off** for measurement runs. The corpora hold 60–200
prompts and a long run replays them; cached replays would report a hit rate
that real traffic — where every advance ruling is a different document — never
reproduces. Measuring what it buys is a separate row, not a free boost.

**Kubernetes.** Deployment, Service, PVC, HPA, and three probes that each do a
different job:

- a **startup probe** with a 10-minute budget, which is what allows the other
  two to be sane rather than needing long initial delays;
- a **readiness probe** that fails fast (3 × 5s) so a broken pod leaves the
  endpoint list quickly, with `timeoutSeconds: 3` — well below any real
  completion, because `/health` does no inference, so failing it means the
  event loop is blocked rather than the server being busy;
- a **liveness probe** that is deliberately the least sensitive (30s × 6 = 3
  minutes). A saturated model server is slow, not dead. Restarting it under
  load turns a traffic spike into an outage: the restart drops in-flight work
  and takes minutes to return.

**The HPA scales on queue depth, not CPU.** A vLLM process sits at ~15 % CPU
while the GPU is saturated and the queue is 200 deep, so a CPU-driven HPA
refuses to scale exactly when the SLO is failing. Its honest limit is written
into the manifest: a new replica needs minutes to load weights, so it cannot
absorb a sudden spike, and against a 10× burst what actually protects the SLO
is bounded queueing plus shedding at the ingress.

**Alerting.** Thresholds are derived, not round — see
[`deploy/observability/rules.yml`](deploy/observability/rules.yml):

| Alert | Threshold | Why that number |
|---|---|---|
| p95 over SLO | 5 min | the SLO the cost curves were computed against |
| Error rate | > 1 %, 5 min | the same ceiling `find_knee()` uses to disqualify a load point, so the alert and the benchmark share one definition of "working" |
| **GPU underutilised** | < 20 %, 30 min | a **cost** alert: below this the per-request figure is dominated by idle time and the API almost certainly wins |
| KV cache | > 95 %, 10 min | leading indicator — preemption shows up as a latency tail before it shows up as errors |
| Prompt-length drift | 1.5× vs 7d ago | compared against the same hour last week so weekday shape does not fire |

---

## The quality gate

A serving change that degrades measured quality does not merge.

**The hard part is not comparing two numbers — it is knowing which differences
mean anything.** Project 01 ran one model over the same 28 examples five times
with identical inputs and got slab accuracies of **53.6, 50.0, 53.6, 50.0,
64.3**: a 14.3-point spread from sampling alone. A "fail on any 1-point
regression" rule would have blocked three of those five runs against any of the
others, and would have been switched off inside a week. A gate nobody trusts is
worse than no gate, because it launders a rubber stamp as a control.

So **tolerance is measured, not chosen.** `gate/record.py` takes repeat runs of
one configuration and banks their observed spread; `gate/compare.py` requires a
candidate to move a metric further than that before calling it a regression,
and **refuses to run at all** when the noise floor is unmeasured, when the
dataset SHA has changed, or when the prompt version has changed. Exit code 2,
not 1 — "cannot judge" is a different outcome from "judged and failed".

A 9-point drop passes this gate. A test says so out loud, and the PR comment
shows the tolerance and where it came from, because a reviewer who cannot see
why a 9-point drop passed will not trust it.

**CI has no GPU, and the pipeline does not pretend otherwise.** A workflow
claiming to run the eval would be green because it never ran. What a free
runner enforces is that a change altering model output *arrives with a scored
run attached* — `gate/guard.py` blocks the merge otherwise — and that the run
clears the baseline. The full build → serve → score loop exists as `gate-live`,
conditioned on a GPU runner actually existing rather than left permanently
pending.

*Screenshot of a blocked PR: pending the first measured run.*

---

## Reproducing

> **Cost warning first.** The measurement runs need a GPU. Everything else —
> container, manifests, dashboards, load generator, cost model, gate — runs on
> CPU and costs nothing. Set a hard billing alert before renting anything.
>
> For the GPU side, [`docs/setup.md`](docs/setup.md) is the runbook: WSL2, the
> Blackwell kernel check to do *before* downloading weights, and the KV-cache
> arithmetic that decides batch size at 12 GB.

```bash
git clone https://github.com/ardhendudebnath/llm-serving-unit-economics
cd llm-serving-unit-economics
python -m pip install -e '.[dev,load,charts]'
python -m pytest tests -q                 # 201 tests, no GPU
python -m pytest tests -q -m "not e2e"    # 187 of them, in about 10 seconds
```

**The whole toolchain runs without a GPU.** `tests/mock_server.py` speaks vLLM's
wire format — same SSE framing, same role-only first delta, same trailing usage
chunk — with latency that is specified rather than measured. Start it and drive
a real sweep against it:

```bash
python -m tests.mock_server --port 8099 --ttft-ms 120
python -m bench.sweep --all-profiles --base-url http://localhost:8099 --slo 2.0
python -m bench.report.build
```

The dashboards run on CPU too:

```bash
podman compose -f deploy/observability/docker-compose.yml up
```

Rebuild the workload corpora from Project 01 (only needed if its data changes):

```bash
python -m bench.build_corpus --harness ../domain-eval-harness
```

One GPU block, end to end — spin up, sweep, collect, destroy:

```bash
# Weights live on the host, not in an engine-managed volume -- see
# docs/setup.md for why that lesson was expensive.
podman run -d --device nvidia.com/gpu=all -p 8000:8000 --shm-size 2g \
  --env-file serving.env -v /opt/llm-models:/models \
  localhost/llm-serving:local
python -m bench.sweep --all-profiles --precision fp16 --slo 10 \
  --duration 150 --warmup 15 --cooldown 60
```

`make serve ENGINE=docker` still works; the Dockerfile and manifests are
engine-agnostic.

The sweep checkpoints after every point, so a preempted spot instance loses one
point rather than the run, and it stops climbing once p95 is 3× over the SLO
rather than paying to measure how badly a server fails.

Score the same server on Project 01's harness — same wire format, so only the
base URL differs:

```bash
NIM_BASE_URL=http://localhost:8000 NIM_MODEL=<served id> \
  python -m harness.run --model open-weight-vllm
```

---

## Limitations

Written before the measurements, so none of it is retrofitted.

- **One GPU, one replica, one region.** No high availability, no failover, and
  the cost figures reflect that. A production posture costs more.
- **Poisson arrivals are not real traffic.** They are far better than uniform
  load, but they have no diurnal cycle and no correlated bursts. The
  `peak_to_mean` parameter exists to model burstiness explicitly and defaults
  to 1.0 — a perfectly flat month, which no service has — because a different
  default would bury an assumption inside a headline.
- **28 golden rows.** Project 01's dataset is small and gazette-derived rather
  than human-labelled; every quality number here inherits that, including the
  gate's baseline.
- **A p99 over a short run is the maximum under another name.** Every
  distribution carries its `n`, and `p99_from_thin_tail` is flagged when
  `n < 100`.
- **The quantisation ladder needs pre-quantised checkpoints.** int8 and int4
  are not runtime flags; if no suitable checkpoint exists for the chosen model,
  that rung is missing rather than faked.
- **12 GB of VRAM bounds the model.** Measurements run on an RTX 5070 Ti
  Laptop GPU — 12,227 MiB, driver 595.79, compute capability 12.0, read from
  `nvidia-smi` on 2026-09-07 rather than inferred from the model name. An 8B
  model needs ~16 GB of fp16 weights before any KV cache, so it does not fit;
  the fp16 rung is the baseline every quality delta is measured against, so the
  model is sized to keep it rather than starting the ladder at int8.
- **The card is power-capped, and it changes the measurement, not just the
  numbers.** Measured: SW Power Cap was active for essentially every sample of
  every load point, with mean SM clock at **29–82 % of the rated 3,090 MHz**.
  It is *power*-limited rather than thermally limited — peak temperature was
  81 °C, well inside spec, and the limit is already raised to its 140 W
  maximum. Every throughput figure here is therefore a **floor**.

  The consequence is worse than slow numbers: **consecutive load points are not
  independent measurements.** Each inherits the power state the previous one
  left behind. Run back to back, the `short` profile produced a p95 of 37.5 s
  at 2 rps; with 60 s of idle between points it produced **1.36 s at the same
  rate**, a 27× difference, and the curve stopped being non-monotonic.

  So sweeps use `--cooldown`, and the reason is not tidiness. A knee is a curve
  *across* rates, and a curve is only meaningful if each point measures the
  same system. The honest caveat is that a cooled point measures burst-from-idle
  performance, which flatters a server that in production never gets to cool
  down — characterising sustained throughput needs a separate long steady-state
  run at a fixed rate, and that is a different measurement from the knee.

  Both versions are kept under `results/sweeps/contaminated/` rather than
  deleted, because the difference between them is itself a finding.
- **vLLM runs in WSL2, and says so itself.** It logs `Using 'pin_memory=False'
  as WSL is detected. This may slow down the performance.` Host-to-device
  transfers therefore cannot use pinned memory, which costs most on the
  prefill-heavy `long_in` profile. Every throughput number here is a **floor**
  relative to the same card on bare-metal Linux, and the gap is unmeasured.
- **This card cannot be rented, which is a real problem for the cost curve.**
  No cloud offers a laptop 5070 Ti, so there is no provider rate to read.
  Pairing locally-measured throughput with some other card's hourly price would
  describe a machine that does not exist. So the card is priced from what it
  cost, in `OwnedHardware` in `bench/config.py`, with every input sourced. Two
  inputs are assumptions and are labelled as such: a three-year life and an
  ₹8/kWh tariff. The tariff barely matters. The whole ₹3–14 range of state
  tariffs moves the hourly rate by ₹1.54.

  **Duty cycle is a trap here.** Cost per *serving* hour moves about 11×
  between serving 24 hours a day and 2 (₹10.63 to ₹115.27). The crossover must
  not use those figures. It already bills the card for every hour of the month
  and shows idle time as low utilisation, so pricing the card at a partial duty
  cycle as well would count the idle hours twice. An early version made exactly
  that mistake. The crossover now uses the around-the-clock rate, and a test
  holds it there.
- **The API side is charged the self-hosted model's token counts.** The
  crossover prices each API request at the input and output tokens vLLM
  counted for Qwen3-4B. The API model's own tokenizer counts the same text
  differently, and neither the size nor the direction of that gap is measured,
  so the API line could sit somewhat higher or lower. Scoring the API model on
  Project 01 would record its real token usage on the same prompts and close
  the gap.

---

## Layout

```
bench/       config (dated prices, amortisation) · metrics (percentiles,
             knee) · cost (step function, crossover) · workloads · loadgen
             (open-loop, Poisson) · sweep (checkpointing) · gpu (VRAM,
             throttle detection) · build_corpus · report/ (charts, build)
gate/        compare (noise-floor tolerance) · record · guard
serving/     Dockerfile · entrypoint.sh
deploy/      k8s/ (deployment, service, hpa, configmap)
             observability/ (prometheus.yml, rules.yml, grafana dashboard,
             docker-compose for the CPU-side stack)
tests/       mock_server.py — a fake vLLM, so the toolchain runs with no GPU
data/        workloads/ — the three replayable corpora
docs/        decision.md · cost-log.md · charts/
results/     sweeps/ · eval/ — committed evidence
```

## Licence

MIT. See [LICENSE](LICENSE).
