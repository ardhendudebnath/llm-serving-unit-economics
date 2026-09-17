# When to self-host vs use an API for GST slab classification

A one-page decision doc. **Partly measured.** The framework was committed
before any measurement existed, so the reasoning could not be
reverse-engineered to fit whatever the curves turned out to say. The numbers
are filled in as they arrive.

---

## The short answer

*Pending measurement.* It will be conditional and it will cite a number:

> Self-host above **N requests/month** at a p95 of **X s**. Below that, use the
> API — the GPU sits idle and the API wins outright.

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

> Measured at fp16 and int4 (int8 is being rebuilt): Qwen3-4B-Instruct-2507
> scores **41.4 %** slab accuracy at fp16 (39.3–42.9 % over five runs) and
> **25.0 %** at int4, against about 54 % for the frontier reference. Whether
> fp16's gap of about 13 points is acceptable depends on the task, not the
> server. It has to be settled before the cost comparison means anything.
> int4 is out at any price: the quality gate blocks it.

### 2. What is the volume, and where is the crossover?

Self-hosted cost is flat across each GPU's capacity and then steps; API cost is
linear with no floor. They cross once.

Below the crossover the GPU is mostly idle and you are paying for silicon that
is not working. Above it, marginal cost per request approaches the GPU rate
divided by capacity, which is where self-hosting wins decisively.

> Capacity, measured: one card serves `long_in` at **2 rps** within a 10 s
> p95, at fp16 and at int4 alike. If traffic were perfectly flat, that is
> 5.26 M requests a month. The floor cost at full utilisation is **₹1.48 per
> 1000 requests**, from the laptop's ₹10.63 amortised hour. At this ladder's
> resolution int4 buys no capacity on this profile, so it cannot lower that
> floor either.
>
> Crossover: *pending*, until an API model of matched quality is chosen and
> scored. Comparing against an API model that scores 13 points higher would
> price two different products.

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

- **A quantised rung that holds quality.** If int4 cut cost without moving the
  eval score, the crossover would move left. *Measured, and it does not.* On
  the task's own profile, `long_in`, int4's knee is the same 2 rps as fp16's,
  so it cuts no cost, and it scores 16 points lower on slab accuracy. The plan
  predicted that extraction would suffer, but the mechanism was not the one
  expected. int4 abstains on 11 rows that fp16 answers, and where it does
  answer, it agrees with fp16. Its one real gain is on `long_out`, which meets
  a 10 s target at 1 rps where fp16 meets it at no rate.
- **Mixed precision by route.** int4 is faster on decode-heavy `long_out` and
  worse on the extraction task, which is the shape that would justify routing
  by profile. But the harness scores only the extraction task. Whether int4's
  long answers are good enough is unmeasured, so routing `long_out` to int4
  would rest on speed alone.
- **A cheaper API tier at the same quality.** The comparison is against a dated
  price from Project 01's registry; providers cut prices.
- **Higher utilisation from co-tenancy.** Serving a second workload on the same
  card raises utilisation and cuts per-request cost for both. Out of scope
  here — this project measures one deployment properly rather than three
  half-finished ones — but it is the first thing to try if the crossover lands
  just out of reach.
