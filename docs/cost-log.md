# What this project cost to build

Not the cost curves — those are about what serving a model would cost a
company, at a market rate, and they live in the README. **This page is what
*this project* actually spent**, which is a different number and is kept
separate on purpose.

Budget: **₹5,000.** Tracked here against actual spend.

## Spend to date

| Date | Item | Provider | Hours | ₹ | Running total |
|---|---|---|---:|---:|---:|
| 2026-09-06 | Repo, cost model, load generator, corpora | — | 0 | 0 | 0 |
| 2026-09-07 | Serving container, k8s, alerts, quality gate | — | 0 | 0 | 0 |

**₹0 spent so far**, because nothing built to date needs a GPU. That is the
plan's fourth cost rule working as intended: containers, manifests,
Prometheus rules, the CI gate, the load generator and the whole cost model are
CPU-side plumbing, and every one of them is finished and tested before a single
GPU-hour is bought.

## GPU hours: free tier

This project runs its measurements on free-tier GPU hours rather than rented
ones. Two things follow, and both matter:

1. **The published cost curves do not use ₹0.** They use a dated market rate
   for the GPU class actually used. A crossover chart built on free hardware
   would put the break-even at one request a month and be worthless to anyone
   deciding whether to self-host. See `bench/config.py`, which raises
   `UnpricedError` rather than emitting a figure from an unread rate.
2. **The constraint is real and shapes the work.** Free tiers cap VRAM, session
   length and total hours, which bounds which model can be served and how long
   a sweep can run. The README's Limitations section names the GPU behind every
   number.

| Provider | Card | VRAM | Quota | Used |
|---|---|---:|---|---:|
| *to be confirmed* | | | | |

## What a rented equivalent would have cost

Filled in once the sweeps are done, from measured GPU-hours × the dated market
rate. This is the honest answer to "what would this project have cost without
free credits", and it is worth publishing alongside ₹0 — the free tier is a
constraint this project worked within, not a claim that benchmarking is free.

## Discipline notes

- **Nothing rented idle.** The sweep script is written so a GPU session is spin
  up → run → collect → destroy. It checkpoints after every load point, so an
  interrupted session loses one point rather than the run.
- **The sweep stops early.** Once p95 is 3× over the SLO it stops climbing:
  further points only price how badly a saturated server fails, and no decision
  depends on that.
- **Everything is logged the first time.** VRAM and server-side counters are
  recorded alongside latency even though the headline needs only the knee.
  Re-running a sweep because VRAM was not captured is money set on fire.
