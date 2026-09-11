# `gst-4b-int4` against `gst-4b`, row by row

28 rows · dataset `8c32ff29e970` · prompt `v1`

| | rows |
|---|---:|
| same answer | 17 |
| different answer | 0 |
| newly abstained (`gst-4b-int4` said UNANSWERABLE) | 11 |
| newly answered (`gst-4b` said UNANSWERABLE) | 0 |
| both abstained | 0 |
| unreadable on either side | 0 |

**Lost** (`gst-4b` right, `gst-4b-int4` wrong): 5 — gst-0005, gst-0062, gst-0066, gst-0087, gst-0124

**Gained** (`gst-4b` wrong, `gst-4b-int4` right): 0

## Rows whose answer changed

| id | gold | `gst-4b` | `gst-4b-int4` |
|---|---|---|---|
| gst-0005 | 18 | 18 ✓ | UNANSWERABLE ✗ |
| gst-0035 | 5 | 18 ✗ | UNANSWERABLE ✗ |
| gst-0048 | 18 | 5 ✗ | UNANSWERABLE ✗ |
| gst-0062 | 18 | 18 ✓ | UNANSWERABLE ✗ |
| gst-0066 | 18 | 18 ✓ | UNANSWERABLE ✗ |
| gst-0068 | 18 | 40 ✗ | UNANSWERABLE ✗ |
| gst-0071 | 5 | 18 ✗ | UNANSWERABLE ✗ |
| gst-0087 | 18 | 18 ✓ | UNANSWERABLE ✗ |
| gst-0088 | 5 | 18 ✗ | UNANSWERABLE ✗ |
| gst-0124 | 18 | 18 ✓ | UNANSWERABLE ✗ |
| gst-0128 | 5 | 18 ✗ | UNANSWERABLE ✗ |

## Against every baseline run

| baseline run | same | different | newly abstained | newly answered | lost | gained |
|---|---:|---:|---:|---:|---:|---:|
| 20260908T160730Z_open-weight-vllm_shared | 17 | 0 | 11 | 0 | 5 | 0 |
| 20260908T160930Z_open-weight-vllm_shared | 17 | 0 | 11 | 0 | 5 | 0 |
| 20260908T161130Z_open-weight-vllm_shared | 17 | 0 | 11 | 0 | 4 | 0 |
| 20260908T161330Z_open-weight-vllm_shared | 17 | 0 | 11 | 0 | 4 | 0 |
| 20260908T161533Z_open-weight-vllm_shared | 16 | 1 | 11 | 0 | 5 | 0 |
