"""Build the int8 rung: W8A16 in the same weight format as RedHatAI's W4A16.

Runs inside a throwaway container from the serving image, via quantize/run.sh:

    python3 w8a16.py --out /models/local/Qwen3-4B-Instruct-2507-quantized.w8a16
    python3 w8a16.py --method gptq --out ... [--samples N --max-seq-len M]

**Why the 8-bit rung is built here.** The ladder was planned on RedHatAI's
W8A8 checkpoint, and W8A8 cannot run on this card. vLLM 0.11.0's only int8
activation kernel on CUDA is CUTLASS, which rejects compute capability 12.0 at
the first forward pass (results/eval/int8-w8a8/FAILED.md). Weight-only int8
runs through Marlin, which does support it there, so the rung becomes W8A16.
No one publishes a W8A16 of this model from the same source as the int4 rung.

**Round-to-nearest, not GPTQ.** The first plan was RedHatAI's own W4A16
recipe at 8 bits: GPTQ, with 1,024 calibration samples of up to 8,192 tokens.
On this laptop GPTQ does not fit. llm-compressor holds the model on the host
plus every sample's cached activations, and the WSL VM gets 15 GB of RAM.
Three sizes were tried on 2026-09-11 and 2026-09-17, and each ran out of
memory while propagating the first layer:

    1,024 x 8,192   killed (SIGKILL) after 6 samples
      512 x 4,096   killed after 24 samples
      256 x 2,048   VM at 0 GB available and unresponsive after 97 samples

So the default builds round-to-nearest (RTN), with no calibration data. It
uses every weight setting RedHatAI's W4A16 uses -- int8 in place of int4,
symmetric, group size 128, MSE observer -- without GPTQ's Hessian-based error
correction.

That is a method difference between the rungs as well as a bit-width
difference, and it cuts one way. RTN is the weaker method. If this int8 rung
holds fp16's quality, GPTQ at 8 bits would hold it too, and "8 bits holds,
4 bits does not" stands. If it loses quality, the result is ambiguous between
bits and method, and the report has to say so.

`--method gptq` is kept, for a machine with the memory to run it.
provenance.json records the method and every setting actually used.

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
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

RECIPE_SOURCE = "RedHatAI/Qwen3-4B-Instruct-2507-quantized.w4a16 recipe.yaml @ 5d4fde88"

#: The weight format, read from that recipe with num_bits changed from 4 to 8.
#: Shared by both methods, so the checkpoint format is identical either way.
WEIGHTS = {
    "num_bits": 8,
    "type": "int",
    "symmetric": True,
    "strategy": "group",
    "group_size": 128,
    "dynamic": False,
    "observer": "mse",
}
IGNORE = ["lm_head"]

#: GPTQ-only settings from the same recipe. recipe.yaml is what llm-compressor
#: recorded when the checkpoint was built, so where it disagrees with the model
#: card's snippet (dampening_frac 0.05 against 0.01) it is the one followed.
GPTQ = {"actorder": "static", "block_size": 128, "dampening_frac": 0.05}

#: From the same model card's creation snippet. Used by --method gptq only.
CALIBRATION = {
    "dataset": "neuralmagic/LLM_compression_calibration",
    "num_calibration_samples": 1024,
    "max_seq_length": 8192,
}


def build_recipe(method: str) -> QuantizationModifier | GPTQModifier:
    if method == "rtn":
        return QuantizationModifier(
            targets=["Linear"],
            ignore=IGNORE,
            config_groups={"group_0": {"targets": ["Linear"], "weights": dict(WEIGHTS)}},
        )
    return GPTQModifier(
        targets=["Linear"],
        ignore=IGNORE,
        config_groups={"group_0": {
            "targets": ["Linear"],
            "weights": {**WEIGHTS, "actorder": GPTQ["actorder"]},
        }},
        block_size=GPTQ["block_size"],
        dampening_frac=GPTQ["dampening_frac"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", choices=["rtn", "gptq"], default="rtn")
    ap.add_argument("--samples", type=int, default=CALIBRATION["num_calibration_samples"],
                    help="gptq only")
    ap.add_argument("--max-seq-len", type=int, default=CALIBRATION["max_seq_length"],
                    help="gptq only")
    args = ap.parse_args()

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Loaded on the host. RTN needs no forward pass at all. GPTQ's sequential
    # pipeline onloads one decoder layer at a time, because the whole model plus
    # calibration activations would not fit on a 12 GB card.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")

    if args.method == "rtn":
        oneshot(model=model, recipe=build_recipe("rtn"))
        calibration = None
    else:
        ds = load_dataset(CALIBRATION["dataset"], split="train")
        ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(
            ex["messages"], add_generation_prompt=False, tokenize=False,
        )})
        oneshot(
            model=model,
            tokenizer=tokenizer,
            dataset=ds,
            recipe=build_recipe("gptq"),
            max_seq_length=args.max_seq_len,
            num_calibration_samples=args.samples,
        )
        calibration = {
            **CALIBRATION,
            "num_calibration_samples": args.samples,
            "max_seq_length": args.max_seq_len,
            "reduced_from_recipe": (
                args.samples != CALIBRATION["num_calibration_samples"]
                or args.max_seq_len != CALIBRATION["max_seq_length"]
            ),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tokenizer.save_pretrained(args.out)

    provenance = {
        "built_from": args.model,
        "method": args.method,
        "weights": WEIGHTS,
        "ignore": IGNORE,
        "weights_source": f"{RECIPE_SOURCE}, num_bits changed from 4 to 8",
        "gptq": GPTQ if args.method == "gptq" else None,
        "calibration": calibration,
        "note": (
            "round-to-nearest: the recipe's weight settings without GPTQ's error "
            "correction, because GPTQ calibration did not fit in the WSL VM's 15 GB"
            if args.method == "rtn" else "GPTQ with the recipe's settings"
        ),
        "versions": {
            pkg: version(pkg)
            for pkg in ("llmcompressor", "compressed-tensors", "transformers", "torch")
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
