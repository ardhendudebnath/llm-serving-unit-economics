### Quality gate: **passed**

`gst-4b-int8` against the **fp16** baseline (`gst-4b`) · dataset `8c32ff29e970` (28 rows) · prompt `v1`

| Metric | Baseline | This run | Δ | Tolerance | |
|---|---:|---:|---:|---:|:--:|
| Slab accuracy | 41.4% | 46.4% | +5.0 | ±3.6 | 🟢 |
| HSN-4 accuracy | 62.9% | 64.3% | +1.4 | ±3.6 | ⚪ |
| Chapter accuracy | 70.0% | 71.4% | +1.4 | ±3.6 | ⚪ |
| Abstention accuracy | 100.0% | 100.0% | +0.0 | ±1.0 | ⚪ |
| Abolished slab answered | 5.0% | 3.6% | +1.4 | ±3.6 | ⚪ |
| Abolished slab recited | 5.7% | 3.6% | +2.1 | ±3.6 | ⚪ |
| Unparseable responses | 0 | 0 | +0.0 | ±0.0 | ⚪ |
| Errored calls | 0 | 0 | +0.0 | ±0.0 | ⚪ |

Tolerance is the spread measured across **5 repeat runs** of this same configuration, recorded 2026-09-08 — not a chosen threshold. A change has to move a metric further than the configuration moves on its own before it counts.

> fp16 baseline: Qwen3-4B-Instruct-2507 on RTX 5070 Ti Laptop, vLLM 0.11.0, greedy, max_model_len 4096
