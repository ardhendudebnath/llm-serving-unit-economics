# Picking the model, and the three checkpoints

**Chosen: `Qwen3-4B-Instruct-2507`**, served at three precisions.

Everything below was read from the HuggingFace API on **2026-09-07** rather
than recalled. Project 01 already learned why that matters — it pinned a hosted
model that later answered `410 Gone: reached its end of life`, on the first
live call.

| Rung | Checkpoint | On disk | Format |
|---|---|---:|---|
| fp16 | `Qwen/Qwen3-4B-Instruct-2507` | 8.04 GB | BF16 native |
| int8 | `RedHatAI/Qwen3-4B-Instruct-2507-quantized.w8a8` | 5.19 GB | compressed-tensors, W8A8 |
| int4 | `RedHatAI/Qwen3-4B-Instruct-2507-quantized.w4a16` | 3.43 GB | compressed-tensors, W4A16 |

All three open, none gated. Sizes are summed from each repo's `.safetensors`
files, not derived from a parameter count — a quantised checkpoint carries
scales and packed integer containers that a bytes-per-weight estimate misses.

---

## Why this one

### 1. fp16 has to fit, and fp16 is the tight rung

The full ladder is the point, and fp16 is the baseline every quality delta is
measured against. Starting at int8 would make every number relative to another
quantised run.

The card has **12,227 MiB**. At `--gpu-memory-utilization 0.90` vLLM gets
~10.75 GB for weights *and* KV cache:

| | weights | left for KV | sequences at 4,096 tokens |
|---|---:|---:|---:|
| fp16 | 8.04 GB | 2.71 GB | **~4.5** |
| int8 | 5.19 GB | 5.56 GB | ~9.2 |
| int4 | 3.43 GB | 7.32 GB | ~12.2 |

That fp16 row is the constraint the model was chosen against. Four to five
concurrent sequences is enough for continuous batching to do something
visible; two would not be.

It also previews the ladder's mechanism: **int4 nearly triples concurrency**,
because the VRAM it frees becomes KV cache. Any throughput gain is not just
narrower arithmetic.

### 2. The KV cache is what actually binds

Read from `config.json`: 36 layers, 8 KV heads, head dim 128. So

```
2 (K and V) x 8 x 128 x 2 bytes = 4 KiB per layer
                     x 36 layers = 144 KiB per token
```

0.60 GB for a single 4,096-token sequence. The model advertises **262,144
positions**; reserving for that would be ~37 GB of KV cache for one sequence on
a 12 GB card. `MAX_MODEL_LEN=4096` is not a cautious default, it is the only
workable setting — and the `long_in` corpus peaks near 1,800 tokens, so it
costs nothing real.

### 3. Both quantised rungs come from one publisher, in one format

This is the methodological point, and it eliminated the otherwise-attractive
option below. Mixing an AWQ community checkpoint with a RedHatAI W8A8 would put
a **quantisation-methodology difference inside the one comparison the project
exists to make.** The ladder is supposed to isolate bit width; it cannot do
that if each rung was produced by different people with different calibration
data.

RedHatAI publishes both W8A8 and W4A16 for this exact base model, in
`compressed-tensors`. `tests/test_ladder.py` asserts the publisher and format
match, so a later "convenient" substitution fails the build.

### 4. Non-thinking, on purpose

Qwen3-4B (the original) is a hybrid-thinking model: it emits a reasoning chain
into a separate field that still bills as output tokens. For this task that is
actively wrong — Project 01's prompt asks for four labelled lines, and the
reasoning would inflate decode time until `long_in` measured thinking rather
than extraction. `Instruct-2507` is non-thinking only.

It also settles the `reasoning_style` field left blank in Project 01's
`open-weight-vllm` row: there is no switch to send.

---

## What was rejected

**`Qwen3.5-4B`** — newer and far more popular (7.2M downloads), and RedHatAI
publishes both quantised rungs for it too. Rejected on VRAM: 4.66B parameters
means **9.32 GB** of fp16 weights, leaving only ~1.43 GB for KV cache, or about
**2.4 concurrent sequences** at 4,096 tokens. A benchmark about continuous
batching that can hold two sequences in a batch has almost nothing to measure.
The better model is the wrong instrument here.

**`Qwen3-4B` (hybrid thinking) with first-party AWQ** — `Qwen/Qwen3-4B-AWQ` is
first-party with 195k downloads, which was appealing. But
`RedHatAI/Qwen3-4B-quantized.w8a8` returns **HTTP 401** and no credible W8A8
alternative exists for this base: the search returned only community repos with
2–27 downloads. **There is no int8 rung**, so the ladder cannot be built. The
thinking behaviour would have been a problem anyway.

**8B-class models** — 16 GB of fp16 weights against a 12 GB card. Does not fit
at all, which is what forced the move to 4B in the first place.

---

## Honest caveats

- **The quantised repos have low download counts** — 1,280 and 727. That is
  worth stating rather than glossing. RedHatAI is Red Hat's (formerly Neural
  Magic's) quantisation programme and is the canonical `compressed-tensors`
  publisher, and their higher-traffic siblings for other models are widely
  used, so provenance is strong even where usage is thin. Both were checked to
  exist and be ungated; neither has been *run* yet.
- **The parameter counts read higher for quantised rungs** (4.41B and 4.44B
  against 4.02B). That is not a bigger model — the safetensors metadata counts
  packed int32 containers and per-group scales alongside the weights.
- **The "fp16" rung is served as bfloat16.** The checkpoint is BF16-native, and
  forcing float16 would narrow the exponent range enough to risk overflow in
  attention. The rung keeps the name the plan uses; what runs is 16-bit native,
  and every report says which.
- **Nothing here has been served yet.** These are existence, size and format
  checks. Whether vLLM 0.11.0 loads all three on this card is the next thing to
  find out, and the fp16 rung is the one with the least headroom.
