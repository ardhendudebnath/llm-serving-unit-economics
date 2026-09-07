# vLLM's official OpenAI-compatible server image.
#
# Pinned to an exact tag, never :latest. The whole repository exists to publish
# numbers attached to a configuration, and a floating base tag means a rerun
# six months from now measures different software under the same commit.
#
# Verified 2026-09-07: this tag runs on sm_120 (Blackwell). torch 2.8.0+cu128,
# sm_120 present in get_arch_list(), and an fp16 matmul launches on the card.
#
# **Fully qualified on purpose.** Podman has no implicit docker.io prefix and
# refuses a short name outright:
#
#   short-name "vllm/vllm-openai:v0.11.0" did not resolve to an alias and no
#   unqualified-search registries are defined
#
# Docker accepts the long form identically, so spelling out the registry is
# what makes this Dockerfile work under both engines rather than one.
FROM docker.io/vllm/vllm-openai:v0.11.0

LABEL org.opencontainers.image.source="https://github.com/ardhendudebnath/llm-serving-unit-economics"
LABEL org.opencontainers.image.description="vLLM serving one open-weight model for GST classification, instrumented for cost measurement"

# Where the weights cache. Mounted as a volume in k8s so a pod restart does not
# re-download several GB -- which matters enormously for the readiness probe
# budget, and is the difference between a 30-second and a 15-minute cold start.
ENV HF_HOME=/models
VOLUME ["/models"]

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

# Liveness only. Readiness is decided by the k8s probe against /health, because
# a container-level healthcheck cannot express the long grace period a model
# server needs on first start. See deploy/k8s/deployment.yaml.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
