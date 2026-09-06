# When to self-host vs use an API for GST slab classification

A one-page decision doc. **Currently a framework with the numbers missing** —
it is committed now so the reasoning is fixed before any measurement exists and
cannot be reverse-engineered to fit whatever the curves turn out to say.

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

> Measured: *pending.*

### 2. What is the volume, and where is the crossover?

Self-hosted cost is flat across each GPU's capacity and then steps; API cost is
linear with no floor. They cross once.

Below the crossover the GPU is mostly idle and you are paying for silicon that
is not working. Above it, marginal cost per request approaches the GPU rate
divided by capacity, which is where self-hosting wins decisively.

> Crossover: *pending.*

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

- **A quantised rung that holds quality.** If int4 cuts cost substantially
  without moving eval score, the crossover moves left and self-hosting gets
  more attractive at lower volume. The plan predicts the opposite for this
  task: extraction with exact-match scoring has no partial credit, so it should
  suffer more than open-ended generation. *Unmeasured.*
- **Mixed precision by route.** If quantisation hurts `long_in` but not
  `long_out`, the right answer is neither fp16 nor int4 but both, routed by
  profile. That would be the interesting finding.
- **A cheaper API tier at the same quality.** The comparison is against a dated
  price from Project 01's registry; providers cut prices.
- **Higher utilisation from co-tenancy.** Serving a second workload on the same
  card raises utilisation and cuts per-request cost for both. Out of scope
  here — this project measures one deployment properly rather than three
  half-finished ones — but it is the first thing to try if the crossover lands
  just out of reach.
