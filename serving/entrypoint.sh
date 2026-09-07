#!/usr/bin/env bash
# Start vLLM with the configuration under test.
#
# Every knob the benchmark sweeps is an environment variable, so one image
# serves every rung of the quantisation ladder. The alternative -- an image per
# precision -- would mean the ladder compared three builds as well as three
# precisions.
set -euo pipefail

MODEL_ID="${MODEL_ID:?MODEL_ID must be set, e.g. Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"

# fp16 | int8 | int4. Mapped rather than passed through, so an unsupported
# value fails here with a readable message instead of deep inside vLLM.
PRECISION="${PRECISION:-fp16}"

# Bounds the KV cache. At long context the cache, not the weights, is what
# runs a card out of memory: it grows with batch size x sequence length, and
# this is the ceiling on the second term. Set it to the longest prompt the
# workload actually contains, not to the model's architectural maximum --
# reserving for 128k context you never send wastes the VRAM that would
# otherwise hold more concurrent sequences.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# Max sequences in a batch. The other half of the KV-cache tradeoff, and the
# main throughput/latency dial: higher fills the GPU better and lengthens the
# tail, lower protects p99 and wastes silicon.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# Fraction of VRAM vLLM may claim for weights plus KV cache. Left below 1.0
# because CUDA context, activations and fragmentation live in the remainder,
# and a server that OOMs mid-sweep costs a whole GPU block.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# OFF by default, and this is a measurement decision rather than a performance
# one. The workload corpora hold 60-200 distinct prompts, so a long run replays
# them; with prefix caching on, those replays would be served from cache and
# the reported throughput would describe a cache hit rate that real traffic --
# where every advance ruling is a different document -- will never reproduce.
# Turn it on deliberately to measure what it buys, and report it as its own row.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"

# Each rung is a different checkpoint, not a different flag: int8 and int4
# weights are quantised ahead of time, and pointing MODEL_ID at an fp16 repo
# with PRECISION=int8 fails at load rather than silently serving fp16.
#
# Both quantised rungs use compressed-tensors, from the same publisher. That is
# deliberate -- see bench/config.py. Serving one rung as AWQ and the other as
# compressed-tensors would put a quantisation-methodology difference inside a
# comparison meant to isolate bit width.
case "$PRECISION" in
  # bfloat16, not float16. The checkpoint is BF16-native, and forcing float16
  # narrows the exponent range enough to risk overflow in attention. The rung
  # is named fp16 because the plan names it that; what is served is 16-bit
  # native, and the report says which.
  fp16) QUANT_ARGS=(--dtype bfloat16) ;;
  int8) QUANT_ARGS=(--quantization compressed-tensors) ;;
  int4) QUANT_ARGS=(--quantization compressed-tensors) ;;
  *)
    echo "PRECISION must be one of fp16, int8, int4 (got '$PRECISION')" >&2
    exit 2
    ;;
esac

PREFIX_ARGS=()
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
  PREFIX_ARGS=(--enable-prefix-caching)
else
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

echo "  model      $MODEL_ID (served as $SERVED_MODEL_NAME)"
echo "  precision  $PRECISION"
echo "  max len    $MAX_MODEL_LEN · max seqs $MAX_NUM_SEQS · mem $GPU_MEMORY_UTILIZATION"
echo "  prefix cache $([ "$ENABLE_PREFIX_CACHING" = "1" ] && echo on || echo off)"

exec python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${QUANT_ARGS[@]}" \
  "${PREFIX_ARGS[@]}" \
  --disable-log-requests \
  "$@"
