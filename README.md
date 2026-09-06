# Serving an open-weight LLM for GST classification: latency, quality and cost

**What does inference actually cost, and does self-hosting beat the API?**
This repository answers that for one model, one task and one deployment, with
measurements rather than estimates — and publishes the curves.

The task comes from [Project 01](https://github.com/ardhendudebnath/gst-eval-harness):
classifying Indian goods into the GST slab that is *currently* in force, after
two restructures in under two years. That project scores quality. This one adds
latency and cost, so all three sit in the same frame.

> **Status: build complete, nothing measured yet (week 1 of 6).**
>
> Every number in this README is absent rather than estimated. There is no
> headline, no crossover point and no quantisation table, because no model has
> been served yet. The tooling below is built, tested and linted; the GPU hours
> have not been spent.
>
> This section will stay here, saying exactly this, until a measurement
> replaces it. Project 01 holds the same line and it is the reason its numbers
> are worth reading.

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

| | Status |
|---|---|
| Latency vs load, by workload profile | not run |
| Quantisation ladder: fp16 / int8 / int4 | not run |
| Quality per precision, scored on Project 01's harness | not run |
| Cost per 1000 requests, self-hosted vs API | not run |
| GPU utilisation at each point | not run |

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

**In it:** GPU rental at a dated market rate, for the hours the deployment
would actually run.

**Not in it:**

- engineering time — building this took far longer than the GPU it measures
- on-call, and the people to do it
- redundancy: these figures are for **one replica**, which is not a production
  posture. A second for availability doubles the GPU line.
- idle capacity beyond what the peak-to-mean ratio models
- storage, egress, and the control plane
- the cost of being wrong — an API provider absorbs a bad deploy; you do not

**The rate is a market rate, not what was paid.** These measurements run on
free-tier GPU hours. Reporting the cost as zero would put the crossover at one
request a month and make the chart worthless, so every published figure uses
what an hour of that GPU class actually costs to rent, read from a provider on
a stated date. What the project itself spent is a separate number and lives in
[`docs/cost-log.md`](docs/cost-log.md). Keeping the two apart is deliberate:
one answers "what would this cost a company", the other "what did this cost its
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

```bash
git clone https://github.com/ardhendudebnath/llm-serving-unit-economics
cd llm-serving-unit-economics
python -m pip install -e '.[dev,load]'
python -m pytest tests -q          # 62 tests, no GPU, no network
```

Rebuild the workload corpora from Project 01 (only needed if its data changes):

```bash
python -m bench.build_corpus --harness ../domain-eval-harness
```

One GPU block, end to end — spin up, sweep, collect, destroy:

```bash
docker run -d --gpus all -p 8000:8000 --shm-size 2g \
  -e MODEL_ID=<model> -e PRECISION=fp16 \
  llm-serving:latest
python -m bench.sweep --all-profiles --precision fp16 --slo 5.0
```

The sweep checkpoints after every point, so a preempted spot instance loses one
point rather than the run, and it stops climbing once p95 is 3× over the SLO
rather than paying to measure how badly a server fails.

Score the same server on Project 01's harness — same wire format, so only the
base URL differs:

```bash
NIM_BASE_URL=http://localhost:8000 python -m harness.run --model open-weight-local
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
- **It is a laptop GPU, and laptop GPUs throttle.** Under sustained load a
  laptop card holds boost clocks briefly and then drops, so a 120-second run at
  a high arrival rate can end up measuring the cooling system. `bench/gpu.py`
  samples clocks, temperature and NVML's throttle-reason bits for exactly the
  span of each load point and flags any point where throttling was active;
  a flagged point's throughput is reported as a **floor**, not as the card's
  sustained rate.
- **This card cannot be rented, which is a real problem for the cost curve.**
  No cloud offers a laptop 5070 Ti, so there is no provider rate to read.
  Pairing locally-measured throughput with some other card's hourly price would
  describe a machine that does not exist, so the honest route is
  `amortised_usd_per_hour()`: purchase price over expected serving hours, plus
  measured power draw at a stated tariff. Every input is printed beside the
  result, and `duty_cycle` is the one that moves it most.

---

## Layout

```
bench/       config (dated prices) · metrics (percentiles, knee) · cost
             (step function, crossover) · workloads · loadgen (open-loop,
             Poisson) · sweep (checkpointing) · build_corpus
gate/        compare (noise-floor tolerance) · record · guard
serving/     Dockerfile · entrypoint.sh
deploy/      k8s/ (deployment, service, hpa, configmap) · observability/
data/        workloads/ — the three replayable corpora
docs/        decision.md · cost-log.md
results/     sweeps/ · eval/ — committed evidence
```

## Licence

MIT. See [LICENSE](LICENSE).
