# Every target is a thin wrapper over `python -m ...` or one docker command,
# so the repo is equally usable without make -- which matters here, because
# most of it is developed on Windows and measured inside WSL2.

PY ?= python
HARNESS ?= ../domain-eval-harness
BASE_URL ?= http://localhost:8000
SLO ?= 5.0
PRECISION ?= fp16

# Container runtime. Docker Engine (docker-ce) is Apache 2.0 and free for any
# use -- it is Docker *Desktop* that carries a subscription, and this project
# does not use it. So this variable is about portability, not licensing.
#
#   make serve ENGINE=podman
#
# Podman accepts the same Dockerfile and the same run flags, with one
# exception: GPUs come through the Container Device Interface rather than
# --gpus. See GPU_FLAG below and docs/setup.md.
ENGINE ?= docker

ifeq ($(ENGINE),podman)
GPU_FLAG ?= --device nvidia.com/gpu=all
else
GPU_FLAG ?= --gpus all
endif

.PHONY: help test test-fast lint corpus sweep serve mock observability gate baseline charts clean

help:
	@echo "test         everything, including end-to-end HTTP (~70s)"
	@echo "test-fast    skip the end-to-end tests (~0.2s)"
	@echo "lint         ruff"
	@echo ""
	@echo "corpus       rebuild the workload corpora from Project 01"
	@echo "mock         a fake vLLM on :8099, for driving the tooling with no GPU"
	@echo "serve        build and run the real serving container (needs a GPU)"
	@echo "sweep        load sweep across every profile (needs a running server)"
	@echo "charts       render the report charts from results/"
	@echo ""
	@echo "observability  Prometheus + Grafana on :9090 and :3000"
	@echo "baseline     record a gate baseline from repeat eval runs"
	@echo "gate         compare the newest eval run against the baseline"

# ------------------------------------------------------------------ dev ----

test:
	$(PY) -m pytest tests -q

# The end-to-end tests drive real HTTP with real sleeps. Worth running, but not
# on every save.
test-fast:
	$(PY) -m pytest tests -q -m "not e2e"

lint:
	$(PY) -m ruff check .

corpus:
	$(PY) -m bench.build_corpus --harness $(HARNESS)

# ------------------------------------------------------------- measure -----

# No GPU needed. Speaks the same wire format as vLLM, so the load generator,
# the sweep and the charts can all be exercised end to end before renting or
# booting anything.
mock:
	$(PY) -m tests.mock_server --port 8099

serve:
	$(ENGINE) build -t llm-serving:local serving/
	$(ENGINE) run --rm -it $(GPU_FLAG) -p 8000:8000 --shm-size 2g \
	  --env-file serving.env -v vllm-models:/models \
	  llm-serving:local

# One GPU block: this is the whole measurement. Checkpoints after every point,
# so an interruption loses one point rather than the run.
sweep:
	$(PY) -m bench.sweep --all-profiles --precision $(PRECISION) \
	  --slo $(SLO) --base-url $(BASE_URL)

charts:
	$(PY) -m bench.report.build

# -------------------------------------------------------------- operate ----

observability:
	docker compose -f deploy/observability/docker-compose.yml up

# --------------------------------------------------------------- gating ----

# Needs at least three runs of one configuration -- two give one gap, which is
# not a spread. See gate/compare.py for why the tolerance has to be measured.
baseline:
	$(PY) -m gate.record --runs "results/eval/*.json" --precision $(PRECISION)

gate:
	$(PY) -m gate.compare --candidate $$(ls -1 results/eval/*.json | sort | tail -n1)

clean:
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
