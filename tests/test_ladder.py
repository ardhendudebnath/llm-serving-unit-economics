"""The quantisation ladder must fit the card, and compare the right thing."""

from __future__ import annotations

import pytest

from bench.config import (
    HEAD_DIM,
    LADDER,
    N_KV_HEADS,
    N_LAYERS,
    concurrent_sequences,
    get_gpu,
    get_precision,
    kv_cache_gb,
)

CARD = "rtx5070ti-laptop"
CONTEXT = 4096


# ------------------------------------------------------------- fits the card --

def test_every_rung_fits_the_12gb_card_with_room_for_a_kv_cache():
    # The whole model choice turns on this. Weights alone fitting is not
    # enough -- a rung that leaves no room for a KV cache cannot serve a
    # single request.
    vram = get_gpu(CARD).vram_gb
    for key, rung in LADDER.items():
        assert rung.weights_gb < vram, f"{key} weights exceed the card"
        assert concurrent_sequences(key, CONTEXT, CARD) >= 2.0, (
            f"{key} leaves room for fewer than 2 concurrent sequences at "
            f"{CONTEXT} tokens -- there would be no batching to measure"
        )


def test_the_fp16_baseline_is_the_tight_one():
    # Worth pinning: fp16 is what decides whether the model was small enough,
    # and it is the rung with the least headroom by construction.
    seqs = {k: concurrent_sequences(k, CONTEXT, CARD) for k in LADDER}
    assert seqs["fp16"] == min(seqs.values())
    # ~4-5 sequences: enough for continuous batching to do something visible.
    assert 3.5 < seqs["fp16"] < 6.0


def test_int4_buys_substantially_more_concurrency_than_fp16():
    # The mechanism behind the expected throughput gain: smaller weights leave
    # more VRAM for KV cache, so more sequences run at once. If this were not
    # true the ladder would only be measuring arithmetic width.
    assert (concurrent_sequences("int4", CONTEXT, CARD)
            > 2.0 * concurrent_sequences("fp16", CONTEXT, CARD))


def test_weights_shrink_monotonically_down_the_ladder():
    sizes = [LADDER[k].weights_gb for k in ("fp16", "int8", "int4")]
    assert sizes == sorted(sizes, reverse=True)


# ----------------------------------------------------------- KV arithmetic ---

def test_kv_cache_per_token_matches_the_published_architecture():
    # 2 (K and V) x 8 kv heads x 128 head_dim x 2 bytes x 36 layers = 144 KiB.
    # Read from config.json on 2026-09-07, not assumed.
    per_token_bytes = kv_cache_gb(1) * 1e9
    assert per_token_bytes == pytest.approx(2 * N_KV_HEADS * HEAD_DIM * 2 * N_LAYERS)
    assert per_token_bytes / 1024 == pytest.approx(144.0)


def test_a_4096_token_sequence_costs_about_0_6gb():
    assert kv_cache_gb(4096) == pytest.approx(0.60, abs=0.02)


def test_context_length_and_batch_size_trade_directly():
    # The tradeoff the plan calls interview material. Doubling context halves
    # the sequences that fit, because it is the same VRAM either way.
    at_4k = concurrent_sequences("fp16", 4096, CARD)
    at_8k = concurrent_sequences("fp16", 8192, CARD)
    assert at_8k == pytest.approx(at_4k / 2, rel=1e-6)


def test_reserving_the_full_context_window_would_fit_nothing():
    # The model advertises 262,144 positions. Reserving for that is 37 GB of
    # KV cache for one sequence on a 12 GB card.
    assert kv_cache_gb(262_144) > 30
    assert concurrent_sequences("fp16", 262_144, CARD) < 0.2


# ------------------------------------------------- comparing the right thing --

def test_both_quantised_rungs_share_a_publisher_and_a_format():
    """Otherwise the ladder measures the quantiser, not the bit width.

    Mixing an AWQ community checkpoint with a RedHatAI W8A8 would put a
    methodology difference inside the one comparison the project exists to
    make.
    """
    int8, int4 = get_precision("int8"), get_precision("int4")
    assert int8.repo.split("/")[0] == int4.repo.split("/")[0] == "RedHatAI"
    assert int8.vllm_quantization == int4.vllm_quantization == "compressed-tensors"


def test_every_rung_names_its_own_checkpoint():
    # int8 and int4 are pre-quantised weights, not runtime flags. Three rungs
    # sharing one repo would mean two of them were not what they claim.
    repos = [r.repo for r in LADDER.values()]
    assert len(set(repos)) == len(repos)
    assert all(repos)


def test_the_whole_ladder_is_one_model_family():
    # A ladder across different base models would compare models, not
    # precisions.
    assert all("Qwen3-4B-Instruct-2507" in r.repo for r in LADDER.values())


def test_the_baseline_is_served_as_bfloat16_not_float16():
    """The checkpoint is BF16-native.

    Forcing float16 narrows the exponent range and risks overflow in
    attention. It is a real numerical difference, not a naming quibble, and
    the rung being called "fp16" is what makes it easy to get wrong.
    """
    assert get_precision("fp16").dtype == "bfloat16"
    assert get_precision("fp16").vllm_quantization is None
