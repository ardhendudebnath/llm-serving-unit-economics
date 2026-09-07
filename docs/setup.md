# Getting from Windows to a serving GPU

The measurements run on a local **RTX 5070 Ti Laptop GPU**. Everything else in
this repository already runs on CPU; this page is the one-off setup that makes
the measurement runs possible.

Read the sizing section before pulling any weights. 12 GB is the constraint
that decides the whole project, and getting it wrong costs a download rather
than a rental — but it still costs an evening.

---

## 0. What the hardware actually is

Read from `nvidia-smi` on 2026-09-07, not inferred from the model name:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti **Laptop** GPU |
| VRAM | **12,227 MiB** (~12 GB) |
| Driver | 595.79 |
| CUDA | 13.2 |
| Compute capability | **12.0** (Blackwell, `sm_120`) |
| Max SM clock | 3,090 MHz |

Two consequences worth knowing before anything else:

- **It is not the 16 GB desktop 5070 Ti.** An 8B model needs ~16 GB of fp16
  weights before any KV cache, so 8B is out.
- **It is a laptop card, so it throttles.** `bench/gpu.py` already samples
  clocks and NVML throttle bits for the span of each load point and flags any
  point where performance was limited. Expect flags on long high-rate runs;
  they are honest, not a bug.

---

## 1. WSL2

vLLM is Linux-only, so Windows needs WSL2. This needs admin and a reboot.

**Done on 2026-09-07:** WSL **2.7.13** (kernel 6.18.33.2, WSLg 1.0.73.2) is
installed and `VirtualMachinePlatform` is enabled. Hardware virtualisation was
already on in firmware, which is the usual blocker.

```powershell
wsl --install --no-launch
```

`--no-launch` is not optional in an unattended run. A bare `wsl --install`
installs Ubuntu *and launches it*, which blocks on an interactive "enter a new
UNIX username" prompt — in an elevated window with nobody watching, it hangs.

**A reboot is required before anything further works.** The registry carries
`...\Component Based Servicing\RebootPending`, and until it clears, a
distribution cannot register: `wsl --install Ubuntu-24.04` returns success and
then `wsl -l -v` still reports no distributions.

### It has to be a *Restart*, not a shutdown

Windows Fast Startup (`HiberbootEnabled=1`, the default) means **"Shut down"
followed by pressing the power button is not a reboot.** It hibernates the
kernel session and resumes it, so pending servicing operations never complete —
`RebootPending` stays set, `LastBootUpTime` never moves, and WSL still reports:

```
WSL2 is unable to start since virtualization is not enabled on this machine.
```

which sends you hunting through BIOS settings for a problem that is not there.

Use **Restart**, which always performs a full boot, or from a terminal:

```powershell
shutdown /r /t 0
```

**Do not trust `VirtualizationFirmwareEnabled` when diagnosing this.** It reads
`False` whenever Windows is itself running under a hypervisor — which it is, if
Virtualization-based Security is on. The reliable check is:

```powershell
systeminfo | Select-String "Hyper-V Requirements"
```

`A hypervisor has been detected` means firmware virtualisation is **on** and the
problem is elsewhere.

### After the reboot

Ubuntu 24.04 — chosen because it is what NVIDIA's CUDA base images target, so
the container toolchain lines up rather than needing to be argued with:

```powershell
wsl --install Ubuntu-24.04 --no-launch
```

Then confirm the GPU is visible **inside** WSL. This is the step that actually
matters — WSL2 passes the GPU through via the Windows driver, and **no separate
Linux NVIDIA driver should be installed.** Installing one inside WSL is the
classic way to break passthrough:

```bash
wsl -d Ubuntu-24.04 -e nvidia-smi
```

**Verified working, 2026-09-07:**

| | |
|---|---|
| Distro | Ubuntu 24.04.4 LTS |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 |
| GPU | RTX 5070 Ti Laptop, 12,227 MiB, driver 595.79, compute cap 12.0 |
| Passthrough | `libcuda.so` present under `/usr/lib/wsl/lib/` |
| Host | 24 cores, 15 GB RAM, 955 GB free |

A distribution installed with `--no-launch` has no UNIX user configured, so it
runs as **root** until one is created. That is fine for the checks above and
for Docker, but create a normal user before doing real work:

```bash
wsl -d Ubuntu-24.04
```

The first launch prompts for a username and password interactively — which is
why it is a manual step rather than part of any script here.

If `nvidia-smi` does not print the card, stop — nothing below will work, and
the usual cause is an out-of-date Windows NVIDIA driver rather than anything
in WSL.

---

## 2. Docker with GPU access

Docker CE **inside WSL** plus NVIDIA's container toolkit — not Docker Desktop.
One less Windows-side install, and the daemon lives in the same place as
everything else that touches the GPU.

**Enable systemd first.** WSL2 does not run it by default, and Docker's
packaging ships a systemd unit rather than a SysV init script, so without this
`dockerd` has to be started by hand every session:

```bash
# /etc/wsl.conf, then `wsl --shutdown` and reopen
[boot]
systemd=true
```

Then, as root (`wsl -d Ubuntu-24.04 --user root` — no password needed, which is
how privileged setup happens without typing one into a script):

- add Docker's apt repo with its key under `/etc/apt/keyrings`, install
  `docker-ce docker-ce-cli containerd.io`
- add NVIDIA's repo, install `nvidia-container-toolkit`
- **`nvidia-ctk runtime configure --runtime=docker`**, then restart Docker.
  Installing the toolkit is not enough on its own — without this step
  `docker run --gpus all` fails with *"could not select device driver"*.
- `usermod -aG docker <user>`, then `wsl --shutdown` before the group applies

Verify with a container unrelated to this project, so a failure is unambiguous:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

**Verified working, 2026-09-07:** Docker **29.8.0**, NVIDIA Container Toolkit
**1.20.0**, and the container reported
`NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227 MiB, 595.79, 12.0`.

---

## 3. Verify Blackwell before committing to anything

**This is the real technical risk in the project and it is worth ten minutes.**
`sm_120` is new. vLLM needs kernels compiled for it, which means a build on
CUDA 12.8+ with a PyTorch that supports Blackwell. The pinned tag in
`serving/Dockerfile` follows vLLM's release convention and **has not been
pulled or run** — there is no GPU on the machine this repo was written on.

**`--entrypoint python3` is required.** The vLLM image sets its ENTRYPOINT to
the OpenAI API server, so `docker run IMAGE python3 -c "..."` appends those
arguments to *the server* and dies with `api_server.py: error: unrecognized
arguments: -c`. That looks like a broken image and is nothing of the kind.

```bash
docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:v0.11.0 -c "
import torch
print(torch.__version__, torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
"
```

**Reading the capability alone is not enough.** `get_device_capability()`
returning `(12, 0)` only says the driver reports the card; it says nothing
about whether kernels were compiled for it. Two things settle it: `sm_120`
appearing in `get_arch_list()`, and a kernel actually launching:

```bash
docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:v0.11.0 -c "
import torch
a = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
print('ok', (a @ a).float().abs().sum().item())
"
```

**Verified working, 2026-09-07** — `vllm/vllm-openai:v0.11.0` runs on this card:

| | |
|---|---|
| torch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Device | RTX 5070 Ti Laptop GPU, capability (12, 0) |
| Compiled arches | `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 **sm_120**` |
| Kernel launch | fp16 matmul on device — OK |
| vLLM | 0.11.0 |

So the pin in `serving/Dockerfile` stands. Had it failed with `no kernel image
is available for execution on the device`, the fix would have been to move that
pin forward to a release built on CUDA 12.8 or later and record which one
worked — the version is part of what the measurements mean.

---

## 4. Sizing the model to 12 GB

The full **fp16 → int8 → int4** ladder is the point of the project, and fp16 is
the baseline every quality delta is measured against. So the model has to fit
at fp16, not merely at int4.

Budget, with ~1 GB for the CUDA context and activations:

| Model size | fp16 weights | Left for KV cache | Verdict |
|---|---:|---:|---|
| 8B | ~16 GB | — | **does not fit at all** |
| 7B | ~14 GB | — | does not fit |
| 4B | ~8 GB | ~3 GB | **fits** |
| 3B | ~6 GB | ~5 GB | fits comfortably |

### KV cache, which is what actually decides the batch size

For a 4B-class model with grouped-query attention — roughly 36 layers, 8 KV
heads, head dimension 128 — one token of KV cache costs:

```
2 (K and V) × 8 heads × 128 dim × 2 bytes = 4 KB per layer
                          × 36 layers      = ~147 KB per token
```

So the context length you reserve for is expensive:

| `MAX_MODEL_LEN` | Per sequence | Sequences in ~3 GB |
|---:|---:|---:|
| 8192 | ~1.2 GB | 2 |
| 4096 | ~0.6 GB | 5 |
| 2048 | ~0.3 GB | 10 |

**The `long_in` corpus has a median prompt of 7,177 characters, roughly 1,800
tokens.** Reserving 8192 therefore buys nothing the traffic uses and costs more
than half the available concurrency. Set `MAX_MODEL_LEN=4096` — enough for the
longest real prompt with room for the answer, and it roughly doubles how many
sequences fit in a batch.

This is the tradeoff the plan calls interview material, and here it is concrete
rather than abstract: at 12 GB, max sequence length and max batch size are
directly trading against each other in the same 3 GB.

### Quantised checkpoints

int8 and int4 are **not runtime flags**. vLLM loads pre-quantised weights, so
each rung needs its own checkpoint:

- **fp16** — the base repo.
- **int8** — a `compressed-tensors` W8A8 checkpoint.
- **int4** — an AWQ checkpoint.

Confirm all three exist for the chosen model *before* starting, and record the
exact repo ids. If a rung has no checkpoint, that rung is **missing from the
results**, not approximated — the same rule the rest of this repo follows about
numbers nobody measured.

---

## 5. Run it

```bash
docker build -t llm-serving:local serving/

docker run -d --name vllm --gpus all -p 8000:8000 --shm-size 2g \
  -e MODEL_ID=<the fp16 repo> \
  -e PRECISION=fp16 \
  -e MAX_MODEL_LEN=4096 \
  -e MAX_NUM_SEQS=16 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -v "$HOME/.cache/huggingface:/models" \
  llm-serving:local

curl -fsS http://localhost:8000/health && curl -s http://localhost:8000/v1/models
```

Then the measurement, which is one command:

```bash
python -m bench.sweep --all-profiles --precision fp16 --slo 5.0
```

And the quality half, against the same server — Project 01 needs no changes
beyond a base URL, because vLLM speaks the same wire format:

```bash
cd ../domain-eval-harness
NIM_BASE_URL=http://localhost:8000 NIM_MODEL=<served id> \
  python -m harness.run --model open-weight-vllm
```

---

## 6. Before the ladder: measure the noise floor

The deploy gate refuses to work without one, deliberately — see
`gate/compare.py`. Run the **same** configuration at least three times and let
`gate/record.py` bank the spread:

```bash
for i in 1 2 3 4 5; do
  NIM_BASE_URL=http://localhost:8000 NIM_MODEL=<served id> \
    python -m harness.run --model open-weight-vllm
done
cp results/*.json ../llm-serving-unit-economics/results/eval/

cd ../llm-serving-unit-economics
python -m gate.record --runs "results/eval/*.json" --precision fp16
```

Self-hosted greedy decoding should be far more stable than the API runs that
produced Project 01's 14.3-point spread, but it will not be *zero*: vLLM's
batching is non-deterministic in the sense that matters here, because a
different batch composition changes floating-point reduction order, which
occasionally changes a token. Measure it rather than assuming it away — the
whole gate rests on that number.

---

## Order of work

1. WSL2, GPU passthrough, Docker — **§1–2**
2. Blackwell kernel check — **§3**, before downloading weights
3. fp16 baseline + noise floor — **§5–6**
4. Sweeps at fp16 across all three profiles
5. int8, then int4: sweep and score each
6. Read a GPU rate, or derive one with `amortised_usd_per_hour()`, and render
   the charts

Steps 1–3 are the risky ones. Everything after them is running scripts that
already exist and are already tested.
