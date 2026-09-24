# When to self-host vs use an API for GST slab classification

A one-page decision doc. **Partly measured.** The framework was committed
before any measurement existed, so the reasoning could not be
reverse-engineered to fit whatever the curves turned out to say. The numbers
are filled in as they arrive.

---

## The short answer

On price alone, against Claude Haiku 4.5's list price: **self-host above about
47,000 `long_in` requests a month** at a p95 of 10 s, at fp16 or int8. Below
that, use the API, because the laptop costs ₹7,762 a month whether it serves
or not.

That is not yet a decision, because it is not quality-matched. This model
scores 41.4 % slab accuracy, and Haiku has not been scored on the task. If
Haiku scores much higher, the comparison prices two different products, and
question 1 decides it before cost does.

## How to use this

Answer four questions in order. Stop at the first one that decides it.

### 1. Does anything self-hostable clear the quality bar?

**This question comes first, and it can end the discussion.** Comparing a small
open-weight model's cost against a frontier API model's price is not a
comparison — it prices two different products. The rigorous framing is: match
on quality as measured by Project 01's harness, *then* compare cost.

If no model that fits on one GPU reaches the accuracy the task needs, cost is
irrelevant and the answer is "API", at any volume. On a task where the current
frontier reference scores ~54 % slab accuracy with a 14-point run-to-run
spread, this is a live possibility rather than a formality.

> Measured across the ladder: Qwen3-4B-Instruct-2507 scores **41.4 %** slab
> accuracy at fp16 (39.3–42.9 % over five runs), **41.4 %** at int8 (the same
> five-run mean) and **25.0 %** at int4, against about 54 % for the frontier
> reference. Whether a gap of about 13 points is acceptable depends on the
> task, not the server, and it has to be settled before the cost comparison
> means anything. int8 is fp16's equal on quality. int4 is out at any price:
> the quality gate blocks it.

### 2. What is the volume, and where is the crossover?

Self-hosted cost is flat across each GPU's capacity and then steps; API cost is
linear with no floor. They cross once.

Below the crossover the GPU is mostly idle and you are paying for silicon that
is not working. Above it, marginal cost per request approaches the GPU rate
divided by capacity, which is where self-hosting wins decisively.

> Capacity, measured: one card serves `long_in` at **2 rps** within a 10 s
> p95 at every rung: fp16, int8 and int4. If traffic were perfectly flat, that
> is 5.26 M requests a month. The floor cost at full utilisation is **₹1.48
> per 1000 requests**, from the laptop's ₹10.63 amortised hour. At this
> ladder's resolution quantisation buys no capacity on this profile, so it
> cannot lower that floor.
>
> Crossover, against Claude Haiku 4.5's list price ($1 and $5 per million
> input and output tokens, read 2026-06-24): **47,423 `long_in` requests a
> month**, at 0.9 % GPU utilisation. It is the same at every precision,
> because one card covers the crossover volume about a hundred times over.
> Only the card's monthly cost and the API's price per request set it.
> *Not quality-matched:* Haiku has not been scored on this task.

### 3. How bursty is the traffic?

Capacity has to cover the **peak**, not the mean. A 3× peak-to-mean ratio needs
3× the GPUs for the same monthly volume, which triples the self-hosted line and
cuts utilisation to a third.

Burstiness usually does not move the crossover — a crossover exists only when
API price exceeds self-hosted marginal cost, and when it does it lands well
inside the first GPU's capacity. What burstiness does is decide **whether**
self-hosting wins at all: enough of it lifts marginal cost above the API price
and removes the crossover entirely.

> Measured peak-to-mean: *pending — the model defaults to 1.0, a flat month,
> which flatters self-hosting and is stated wherever it is used.*

### 4. What is not in the number?

Even above the crossover, the GPU line is not the whole cost: engineering time,
on-call, redundancy, and the cost of a bad deploy that an API provider would
have absorbed. A crossover at N requests/month with a single replica and no
on-call is a *lower bound* on the volume at which self-hosting makes sense, not
the threshold itself.

---

## What would change this answer

- **A quantised rung that holds quality and adds capacity.** It would take
  both to move the crossover. *Measured, and neither rung does both.* int8
  holds quality (the same five-run mean as fp16), but its `long_in` knee is the
  same 2 rps, so it cuts no cost on the task's own profile. int4 has the same
  knee and scores 16 points lower. The plan predicted that extraction would
  suffer at low bit width, but not the mechanism: int4 abstains on 11 rows that
  fp16 answers, and where it does answer, it agrees with fp16. What
  quantisation does buy is decode speed. `long_out` meets a 10 s target at
  0.25 rps at int8 and 1 rps at int4, where fp16 meets it at no rate.
- **Mixed precision by route.** int8 already removes most of the case for it.
  int8 matches fp16 on the extraction task and more than halves the time of a
  768-token reply, though part of that is the power cap (see the README). So
  on this card int8 is the better single choice. Routing `long_out` to int4
  would buy more decode speed. But the harness scores only the extraction task,
  so whether int4's long answers are good enough is unmeasured, and that route
  would rest on speed alone.
- **A cheaper API tier at the same quality.** The comparison is against a dated
  price from Project 01's registry; providers cut prices.
- **Higher utilisation from co-tenancy.** Serving a second workload on the same
  card raises utilisation and cuts per-request cost for both. Out of scope
  here — this project measures one deployment properly rather than three
  half-finished ones — but it is the first thing to try if the crossover lands
  just out of reach.
