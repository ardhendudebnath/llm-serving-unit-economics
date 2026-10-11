# `gst-4b-int8` against `gst-4b`, row by row

28 rows · dataset `8c32ff29e970` · prompt `v1`

Candidate run `20260917T113409Z_open-weight-vllm_shared`, scored against every baseline run below.

| | rows |
|---|---:|
| same answer | 27 |
| different answer | 1 |
| newly abstained (`gst-4b-int8` said UNANSWERABLE) | 0 |
| newly answered (`gst-4b` said UNANSWERABLE) | 0 |
| both abstained | 0 |
| unreadable on either side | 0 |

**Lost** (`gst-4b` right, `gst-4b-int8` wrong): 0

**Gained** (`gst-4b` wrong, `gst-4b-int8` right): 1 — gst-0068

## Rows whose answer changed

| id | gold | `gst-4b` | `gst-4b-int8` |
|---|---|---|---|
| gst-0068 | 18 | 40 ✗ | 18 ✓ |

## Against every baseline run

| baseline run | same | different | newly abstained | newly answered | lost | gained |
|---|---:|---:|---:|---:|---:|---:|
| 20260908T160730Z_open-weight-vllm_shared | 27 | 1 | 0 | 0 | 0 | 1 |
| 20260908T160930Z_open-weight-vllm_shared | 27 | 1 | 0 | 0 | 0 | 1 |
| 20260908T161130Z_open-weight-vllm_shared | 26 | 2 | 0 | 0 | 0 | 2 |
| 20260908T161330Z_open-weight-vllm_shared | 26 | 2 | 0 | 0 | 0 | 2 |
| 20260908T161533Z_open-weight-vllm_shared | 26 | 2 | 0 | 0 | 0 | 1 |

## Every run of this configuration, against every baseline run

5 candidate runs × 5 baseline runs. Each cell is the range across the baseline runs.

| candidate run | same | different | newly abstained | newly answered | lost | gained |
|---|---:|---:|---:|---:|---:|---:|
| 20260917T113409Z_open-weight-vllm_shared | 26–27 | 1–2 | 0 | 0 | 0 | 1–2 |
| 20260917T113206Z_open-weight-vllm_shared | 27–28 | 0–1 | 0 | 0 | 0 | 0–1 |
| 20260917T113002Z_open-weight-vllm_shared | 26–27 | 1–2 | 0 | 0 | 0–1 | 0 |
| 20260917T112759Z_open-weight-vllm_shared | 26–27 | 1–2 | 0 | 0 | 0–1 | 0 |
| 20260917T112558Z_open-weight-vllm_shared | 26–28 | 0–2 | 0 | 0 | 0–1 | 0 |
