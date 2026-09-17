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
# One container, so llmcompressor is installed once:
#   1. a smoke build (4 samples, 256 tokens) into a scratch directory, so a
#      broken pipeline fails in minutes rather than an hour in;
#   2. the build at 256 samples x 2,048 tokens;
#   3. only if that fails, one retry at 128 x 2,048, the calibration size the
#      original GPTQ paper used.
#
# **Why not the recipe's 1,024 x 8,192.** It was tried first, on 2026-09-11.
# The WSL VM gets 15 GB of this laptop's 31.4 GB of RAM, and llm-compressor
# holds the whole model on the host plus every calibration sample's
# activations. The recipe's settings were killed (SIGKILL, exit 137) six
# samples into propagating the first layer. A retry at 512 x 4,096 was killed
# twenty-four samples in. provenance.json records the settings that produced
# the weights, so the reduction cannot pass for the recipe.
#
# The container is thrown away, so the pip install cannot touch the serving
# image. llmcompressor 0.7.1 downgrades transformers and accelerate inside it,
# which is fine for building and irrelevant to serving.
set -uo pipefail

MODELS_DIR=${MODELS_DIR:-/opt/llm-models}
IMAGE=localhost/llm-serving:local
NAME=Qwen3-4B-Instruct-2507-quantized.w8a16
OUT=/models/local/$NAME
SMOKE=/models/local/.smoke-w8a16

say() { echo "[$(date -u '+%H:%M:%S')] $*"; }

say "quantisation starting"
# The serving container holds the GPU, and calibration needs all of it. One
# name per call: `podman rm -f vllm-serving llm-quantize` left vllm-serving
# running when llm-quantize did not exist, and the first build started beside
# an int4 server holding 11 GB of the card.
for c in vllm-serving llm-quantize; do
    podman rm -f "$c" >/dev/null 2>&1 || true
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${used:-0}" -gt 1024 ]; then
    say "QUANT FAILED: the GPU still holds ${used} MiB after stopping containers; refusing to calibrate beside another process"
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
      python3 /work/w8a16.py --out $SMOKE --samples 4 --max-seq-len 256 || exit 4
      rm -rf $SMOKE
      echo SMOKE BUILD OK
      python3 /work/w8a16.py --out $OUT --samples 256 --max-seq-len 2048 && exit 0
      echo 'build at 256 x 2048 failed; retrying at 128 samples x 2048 tokens'
      rm -rf $OUT
      python3 /work/w8a16.py --out $OUT --samples 128 --max-seq-len 2048
    "
rc=$?

if [ "$rc" -eq 0 ] && [ -f "$MODELS_DIR/local/$NAME/provenance.json" ]; then
    say "QUANT COMPLETE"
    du -sh "$MODELS_DIR/local/$NAME"
else
    say "QUANT FAILED (exit $rc)"
fi
