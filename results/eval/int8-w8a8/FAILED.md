# int8 as W8A8: cannot run on this card

Tried on 2026-09-11 as the first quantised rung, and abandoned. Kept because
the reason is a finding about the hardware, not a mistake to hide.

| | |
|---|---|
| Checkpoint | `RedHatAI/Qwen3-4B-Instruct-2507-quantized.w8a8`: int8 weights per channel, int8 activations per token (dynamic), compressed-tensors |
| Server | vLLM 0.11.0 (`docker.io/vllm/vllm-openai:v0.11.0`) |
| GPU | RTX 5070 Ti Laptop GPU, compute capability 12.0 |

The weights loaded. The engine then died on the first compiled forward pass,
about 2.5 minutes after the container started:

    torch.ops._C.cutlass_scaled_mm.default(buf8, buf0, arg4_1, buf3, arg5_1, None)
    RuntimeError: Int8 not supported for this architecture
    RuntimeError: Engine core initialization failed. See root cause above.

## Why there is no workaround in this vLLM

Read from the installed vLLM 0.11.0 source rather than recalled:

- int8 activations go through `scaled_mm`, and the only kernel vLLM lists for
  it on CUDA is `CutlassScaledMMLinearKernel`. The Triton fallback is listed
  for ROCm only, so disabling CUTLASS with `VLLM_DISABLED_KERNELS` would leave
  no kernel at all.
- The CUTLASS kernel's `is_supported` only checks that the platform is CUDA.
  vLLM therefore selects it, and the failure surfaces at runtime rather than
  when the config is read.
- Weight-only kernels do cover this card. Marlin reports `uint4b8`,
  `uint8b128`, `float8_e4m3fn` and `float4_e2m1f` at capability 12.0.

## What the 8-bit rung became

**W8A16**: int8 weights, 16-bit activations, served through Marlin. No
publisher offers a W8A16 of this model, so `quantize/run.sh` builds it with
RedHatAI's own W4A16 recipe, changing only the bit width. Both quantised rungs
are then weight-only GPTQ at group size 128 on the same calibration set, and
differ in bit width alone. That isolates bit width more cleanly than W8A8
against W4A16 would have.

The cost is that the ladder no longer measures int8 *activation*
quantisation, which is where a W8A8 checkpoint's prefill speed-up would come
from. On this card and this vLLM, that speed-up is not available to measure.
