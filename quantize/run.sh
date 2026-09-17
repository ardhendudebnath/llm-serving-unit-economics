#!/usr/bin/env bash
# Build the W8A16 int8 checkpoint in a throwaway container. See w8a16.py.
#
#   mkdir -p /root/quantize && cp quantize/run.sh quantize/w8a16.py /root/quantize/
#   /root/quantize/run.sh
#
# Run from a copy under /root, never from the Windows mount -- the same rule as
# the bringup script, for the same reason: bash reads a script incrementally,
# and an edit while it runs resumes execution mid-token.
#
# Round-to-nearest, with no calibration data. The model and the MSE observer fit
# in the WSL VM's 15 GB of RAM; GPTQ's cached calibration activations did not,
# at any of the three sizes tried (w8a16.py records them). One container, so
# llmcompressor is installed once and the build runs straight after.
#
# The container is thrown away, so the pip install cannot touch the serving
# image. llmcompressor 0.7.1 downgrades transformers and accelerate inside it,
# which is fine for building and irrelevant to serving.
set -uo pipefail

MODELS_DIR=${MODELS_DIR:-/opt/llm-models}
IMAGE=localhost/llm-serving:local
NAME=Qwen3-4B-Instruct-2507-quantized.w8a16
OUT=/models/local/$NAME

say() { echo "[$(date -u '+%H:%M:%S')] $*"; }

say "quantisation starting"
# Nothing else should hold the GPU or the VM's memory during a build. One name
# per call: `podman rm -f vllm-serving llm-quantize` left vllm-serving running
# when llm-quantize did not exist, and the first build started beside an int4
# server holding 11 GB of the card.
for c in vllm-serving llm-quantize; do
    podman rm -f "$c" >/dev/null 2>&1 || true
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${used:-0}" -gt 1024 ]; then
    say "QUANT FAILED: the GPU still holds ${used} MiB after stopping containers; refusing to build beside another process"
    exit 1
fi
mkdir -p "$MODELS_DIR/local"
rm -rf "${MODELS_DIR:?}/local/$NAME" "$MODELS_DIR/local/.smoke-w8a16"

podman run --rm --name llm-quantize \
    --device=nvidia.com/gpu=all --shm-size 8g \
    -e HF_HOME=/models -e HF_HUB_DISABLE_XET=1 -e PYTHONUNBUFFERED=1 \
    -v "$MODELS_DIR:/models" -v /root/quantize:/work:ro \
    --entrypoint bash "$IMAGE" -c "
      set -uo pipefail
      pip install --quiet --root-user-action=ignore llmcompressor==0.7.1 || exit 3
      python3 /work/w8a16.py --method rtn --out $OUT
    "
rc=$?

if [ "$rc" -eq 0 ] && [ -f "$MODELS_DIR/local/$NAME/provenance.json" ]; then
    say "QUANT COMPLETE"
    du -sh "$MODELS_DIR/local/$NAME"
else
    say "QUANT FAILED (exit $rc)"
fi
