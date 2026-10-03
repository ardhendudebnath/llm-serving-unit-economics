# Every target is a thin wrapper over `python -m ...` or one docker command,
# so the repo is equally usable without make -- which matters here, because
# most of it is developed on Windows and measured inside WSL2.

PY ?= python
HARNESS ?= ../domain-eval-harness
BASE_URL ?= http://localhost:8000
PRECISION ?= fp16

# The SLO every published knee was measured against. Changing it here changes
# what a new sweep means, and a test holds the alert rules and the dashboard to
# the same number.
SLO ?= 10.0

# Sweep shape, as measured. The cooldown is not politeness: on this
# power-capped card, points run back to back are not independent measurements,
# and the first fp16 sweep was wrong by 27x because of it. See README,
# Limitations.
DURATION ?= 150
WARMUP ?= 15
COOLDOWN ?= 60

# k3s ships its own kubectl:
#
#   make k8s-up KUBECTL="k3s kubectl"
KUBECTL ?= kubectl

# Printed on the crossover chart. The comparison is priced against an API model
# that has not been scored on this task, and the chart says so on its face.
# Remove this only after scoring that model -- not to tidy the picture.
CAVEAT ?= Not quality-matched: Claude Haiku 4.5 has not been scored on this task.\nThis model scores 41.4 % slab accuracy; the API model's accuracy here is unknown.\nThe API is charged this model's token counts, not its own.

# Container runtime. Podman: daemonless, rootless-capable, and no licensing
# question of any kind.
#
#   make serve ENGINE=docker    # still works, nothing here is podman-only
#
# Podman takes the same Dockerfile and the same run flags with one exception:
# GPUs arrive through the Container Device Interface rather than --gpus, so the
# flag is switched below rather than hardcoded. See docs/setup.md.
ENGINE ?= podman

ifeq ($(ENGINE),docker)
GPU_FLAG ?= --gpus all
else
GPU_FLAG ?= --device nvidia.com/gpu=all
endif

# Weights live on the host, not in an engine-managed volume. That was the
# lesson from migrating off Docker: 7.5 GB sat inside a Docker volume and would
# have been destroyed with it. A bind mount is portable across podman, docker
# and the k3s PersistentVolume alike.
MODELS_DIR ?= /opt/llm-models

.PHONY: help test test-fast lint corpus sweep serve mock observability \
        observability-wsl gate baseline charts clean \
        k8s-up k8s-down k8s-status k8s-logs

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
	@echo "k8s-up       apply the manifests, device plugin included"
	@echo "k8s-down     remove them"
	@echo "k8s-status   pods, services, claims, and the pod's events"
	@echo "k8s-logs     follow the server's logs"
	@echo ""
	@echo "observability      Prometheus + Grafana on :9090 and :3000, via compose"
	@echo "observability-wsl  the same, with plain podman, for WSL2"
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
	  --env-file serving.env -v $(MODELS_DIR):/models \
	  llm-serving:local

# ------------------------------------------------------------------ k3s ----

# Apply the manifests to the local single-node cluster. This is the deployment
# the project actually measures from Stage 4 onward; `serve` above is the
# quicker loop for one-off checks.
# The device plugin in this directory is what makes nvidia.com/gpu allocatable;
# without it the pod sits Pending forever. See deploy/k8s/nvidia-device-plugin.yaml.
k8s-up:
	$(KUBECTL) apply -f deploy/k8s/

k8s-down:
	$(KUBECTL) delete -f deploy/k8s/ --ignore-not-found

k8s-status:
	$(KUBECTL) get pods,svc,pvc -o wide
	$(KUBECTL) describe pod -l app=vllm-server | sed -n '/Events:/,$$p'

k8s-logs:
	$(KUBECTL) logs -l app=vllm-server --tail=50 -f

# One GPU block: this is the whole measurement. Checkpoints after every point,
# so an interruption loses one point rather than the run.
sweep:
	$(PY) -m bench.sweep --all-profiles --precision $(PRECISION) \
	  --slo $(SLO) --base-url $(BASE_URL) \
	  --duration $(DURATION) --warmup $(WARMUP) --cooldown $(COOLDOWN)

charts:
	$(PY) -m bench.report.build --caveat "$(CAVEAT)"

# -------------------------------------------------------------- operate ----

observability:
	$(ENGINE) compose -f deploy/observability/docker-compose.yml up

# The same stack where no compose provider is installed, as on the WSL2 box
# this project is measured on. Also wires Prometheus to the pod in k3s and to
# the nvidia-smi stand-in for DCGM, which cannot run under WSL2.
observability-wsl:
	bash deploy/observability/wsl-up.sh

# --------------------------------------------------------------- gating ----

# Needs at least three runs of one configuration -- two give one gap, which is
# not a spread. See gate/compare.py for why the tolerance has to be measured.
baseline:
	$(PY) -m gate.record --runs "results/eval/*.json" --precision $(PRECISION)

gate:
	$(PY) -m gate.compare --candidate $$(ls -1 results/eval/*.json | sort | tail -n1)

clean:
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
