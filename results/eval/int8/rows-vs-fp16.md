# `gst-4b-int8` against `gst-4b`, row by row

28 rows · dataset `8c32ff29e970` · prompt `v1`

| | rows |
|---|---:|
| same answer | 27 |
| different answer | 1 |
| newly abstained (`gst-4b-int8` said UNANSWERABLE) | 0 |
| newly answered (`gst-4b` said UNANSWERABLE) | 0 |
| both abstained | 0 |
| unreadable on either side | 0 |

**Lost** (`gst-4b` right, `gst-4b-int8` wrong): 1 — gst-0062

**Gained** (`gst-4b` wrong, `gst-4b-int8` right): 0

## Rows whose answer changed

| id | gold | `gst-4b` | `gst-4b-int8` |
|---|---|---|---|
| gst-0062 | 18 | 18 ✓ | 28 ✗ |

## Against every baseline run

| baseline run | same | different | newly abstained | newly answered | lost | gained |
|---|---:|---:|---:|---:|---:|---:|
| 20260908T160730Z_open-weight-vllm_shared | 27 | 1 | 0 | 0 | 1 | 0 |
| 20260908T160930Z_open-weight-vllm_shared | 27 | 1 | 0 | 0 | 1 | 0 |
| 20260908T161130Z_open-weight-vllm_shared | 28 | 0 | 0 | 0 | 0 | 0 |
| 20260908T161330Z_open-weight-vllm_shared | 28 | 0 | 0 | 0 | 0 | 0 |
| 20260908T161533Z_open-weight-vllm_shared | 26 | 2 | 0 | 0 | 1 | 0 |
