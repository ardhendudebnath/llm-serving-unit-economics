"""Build the int8 rung: RedHatAI's W4A16 recipe with the bit width set to 8.

Runs inside a throwaway container from the serving image, via quantize/run.sh:

    python3 w8a16.py --out /models/local/Qwen3-4B-Instruct-2507-quantized.w8a16

**Why the 8-bit rung is built here.** The ladder was planned on RedHatAI's
W8A8 checkpoint, and W8A8 cannot run on this card. vLLM 0.11.0's only int8
activation kernel on CUDA is CUTLASS, which rejects compute capability 12.0 at
the first forward pass (results/eval/int8-w8a8/FAILED.md). Weight-only int8
runs through Marlin, which does support it there, so the rung becomes W8A16.

No one publishes a W8A16 of this model from the same source as the int4 rung,
and a community AWQ or GPTQModel checkpoint would put a methodology difference
inside a comparison meant to isolate bit width. So the checkpoint is built with
the recipe RedHatAI shipped inside their W4A16 snapshot, changing only num_bits
from 4 to 8. Both quantised rungs are then weight-only GPTQ at group size 128,
symmetric, static act-order, MSE observer, on the same calibration set, and
they differ in bit width alone.

The only knobs are the two a 12 GB card might force down: calibration samples
and sequence length. Whatever was actually used is written to provenance.json
beside the weights, so a reduced run cannot pass for the recipe's.

run.sh pins llmcompressor to 0.7.1, the release built against
compressed-tensors 0.11.0, which is the version vLLM 0.11.0 reads.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

#: Read from recipe.yaml in RedHatAI/Qwen3-4B-Instruct-2507-quantized.w4a16,
#: snapshot 5d4fde88. recipe.yaml is what llm-compressor recorded when the
#: checkpoint was built, so where it disagrees with the model card's snippet
#: (dampening_frac 0.05 against 0.01) it is the one followed.
RECIPE = {
    "num_bits": 8,  # the one change: 4 in the original
    "group_size": 128,
    "symmetric": True,
    "strategy": "group",
    "observer": "mse",
    "actorder": "static",
    "block_size": 128,
    "dampening_frac": 0.05,
    "ignore": ["lm_head"],
}

#: From the same model card's creation snippet.
CALIBRATION = {
    "dataset": "neuralmagic/LLM_compression_calibration",
    "num_calibration_samples": 1024,
    "max_seq_length": 8192,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=CALIBRATION["num_calibration_samples"])
    ap.add_argument("--max-seq-len", type=int, default=CALIBRATION["max_seq_length"])
    args = ap.parse_args()

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Loaded on CPU. llm-compressor's sequential pipeline onloads one decoder
    # layer at a time for calibration; the whole model at once, plus
    # activations at 8,192 tokens, would not fit in 12 GB.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")

    ds = load_dataset(CALIBRATION["dataset"], split="train")
    ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(
        ex["messages"], add_generation_prompt=False, tokenize=False,
    )})

    recipe = GPTQModifier(
        targets=["Linear"],
        ignore=RECIPE["ignore"],
        config_groups={"group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": RECIPE["num_bits"],
                "type": "int",
                "symmetric": RECIPE["symmetric"],
                "strategy": RECIPE["strategy"],
                "group_size": RECIPE["group_size"],
                "dynamic": False,
                "actorder": RECIPE["actorder"],
                "observer": RECIPE["observer"],
            },
        }},
        block_size=RECIPE["block_size"],
        dampening_frac=RECIPE["dampening_frac"],
    )

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tokenizer.save_pretrained(args.out)

    provenance = {
        "built_from": args.model,
        "recipe_source": (
            "RedHatAI/Qwen3-4B-Instruct-2507-quantized.w4a16 recipe.yaml @ 5d4fde88, "
            "num_bits changed from 4 to 8"
        ),
        "recipe": RECIPE,
        "calibration": {
            **CALIBRATION,
            "num_calibration_samples": args.samples,
            "max_seq_length": args.max_seq_len,
        },
        "calibration_reduced_from_recipe": (
            args.samples != CALIBRATION["num_calibration_samples"]
            or args.max_seq_len != CALIBRATION["max_seq_length"]
        ),
        "versions": {
            pkg: version(pkg)
            for pkg in ("llmcompressor", "compressed-tensors", "transformers", "torch")
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_mib": (round(torch.cuda.max_memory_allocated() / 2**20)
                         if torch.cuda.is_available() else None),
        "seconds": round(time.time() - started),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Written last: run.sh and the rung chain treat its presence as proof the
    # build finished, so it must not exist beside half-saved weights.
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
