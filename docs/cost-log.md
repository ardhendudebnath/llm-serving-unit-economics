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

## GPU hours: a laptop already owned

The plan assumed free-tier cloud GPU hours. The measurements actually run on
the author's own laptop, an ASUS ROG Strix G16 (G615LR) with an RTX 5070 Ti
Laptop GPU, under WSL2 and Podman. No GPU time was rented. Two things follow,
and both matter:

1. **The published cost curves do not use ₹0.** Hardware already paid for is
   not free to serve on. The crossover prices the card at its amortised cost:
   ₹2,49,990 over three years, plus electricity at the 140 W power limit, as if
   available around the clock. That comes to **₹10.63 an hour**. Every input
   and its source is in `OwnedHardware` in `bench/config.py`. A chart drawn at
   ₹0 would put the break-even at one request a month and be worthless to
   anyone deciding whether to self-host.
2. **The constraint is real and shapes the work.** 12 GB of VRAM, part of it
   reserved by Windows, limited the model to 4B parameters. A laptop power cap
   throttles the card under sustained load. Every sweep point records that
   throttling, and the README's Limitations section names the card behind
   every number.

| Card | VRAM | Power limit | Cloud spend |
|---|---:|---:|---:|
| RTX 5070 Ti Laptop GPU | 12,227 MiB | 140 W | ₹0 |

The only marginal cash cost is electricity: at most 0.14 kWh per GPU-hour,
about ₹1.12 at ₹8/kWh. The measured GPU-hours are totalled below once the
ladder is finished.

## What the GPU time was worth

Filled in once the ladder is finished, from measured GPU-hours × the ₹10.63
amortised hourly rate. This is the honest answer to "what did this project
cost, counting the laptop", and it is worth publishing alongside ₹0 of cloud
spend. Owning the card is a constraint this project worked within, not a claim
that benchmarking is free.

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
