"""CPU tests for the V61 timing-campaign harness: no GPU, no FA2 kernels.

Covers the schema/manifest helpers (cell grid, fixed-token inputs, source hashes,
checkpoint identity, cell-result validation, summary entries) and the paired-bootstrap
ratio function on synthetic numbers. The only model-level test runs the SDPA reference
arm on a tiny CPU Qwen2 — it never touches CUDA, and skips when the tiny-model runtime
is unavailable.
"""

from __future__ import annotations

import hashlib
import statistics

import pytest
import torch

from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from scripts.run_segmented_fa2_timing import (
    BOOTSTRAP_DRAWS,
    DOC_COUNTS,
    DOC_LENGTHS,
    GATE_THRESHOLD,
    QUERY_LENGTH,
    REPEATS,
    ROOT,
    SCHEMA_VERSION,
    SEED,
    _summary_entry,
    build_cell_specs,
    cell_inputs,
    collect_checkpoint_identity,
    collect_source_hashes,
    gate_overall,
    gate_pass,
    paired_bootstrap_ratio,
    run_arm_timings,
    validate_cell_result,
)


def test_schema_version_is_pinned():
    assert SCHEMA_VERSION == "v61_timing_v1"
    assert BOOTSTRAP_DRAWS == 10_000
    assert SEED == 47
    assert REPEATS == 10
    assert GATE_THRESHOLD == 1.05


def test_bootstrap_exact_double_speedup_passes_gate():
    stats = paired_bootstrap_ratio([1.0] * 10, [0.5] * 10)
    assert stats["draws"] == 10_000 and stats["seed"] == 47
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["ci95_low"] == pytest.approx(2.0)
    assert stats["ci95_high"] == pytest.approx(2.0)
    assert stats["repeats"] == [2.0] * 10
    assert gate_pass(stats) is True


def test_bootstrap_slow_candidate_fails_gate():
    stats = paired_bootstrap_ratio([1.0] * 10, [2.0] * 10)
    assert stats["mean"] == pytest.approx(0.5)
    assert gate_pass(stats) is False


def test_bootstrap_noisy_synthetic_bounds_and_gate_consistency():
    generator = torch.Generator().manual_seed(SEED)
    reference = 1.0 + 0.05 * torch.randn(REPEATS, generator=generator)
    candidate = reference / 1.2 + 0.01 * torch.randn(REPEATS, generator=generator)
    stats = paired_bootstrap_ratio(
        reference.tolist(), candidate.tolist(), draws=1000, seed=SEED
    )
    assert stats["ci95_low"] <= stats["mean"] <= stats["ci95_high"]
    assert 1.05 < stats["mean"] < 1.35
    assert gate_pass(stats) == (stats["ci95_low"] > GATE_THRESHOLD)


def test_bootstrap_deterministic_for_fixed_seed():
    reference = [1.0, 1.1, 0.95, 1.2, 1.05, 0.9, 1.15, 1.0, 1.3, 0.85]
    candidate = [0.9] * 10
    assert paired_bootstrap_ratio(reference, candidate, seed=SEED) == paired_bootstrap_ratio(
        reference, candidate, seed=SEED
    )


def test_bootstrap_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_ratio([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_ratio([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap_ratio([1.0], [0.0])
    with pytest.raises(ValueError):
        paired_bootstrap_ratio([1.0], [-1.0])
    with pytest.raises(ValueError):
        paired_bootstrap_ratio([-1.0], [1.0])


def test_cell_specs_cover_full_grid():
    specs = build_cell_specs()
    pairs = [(spec["n_docs"], spec["doc_length"]) for spec in specs]
    assert sorted(pairs) == sorted(
        (n_docs, doc_length) for n_docs in DOC_COUNTS for doc_length in DOC_LENGTHS
    )
    for spec in specs:
        assert spec["query_length"] == QUERY_LENGTH
        assert spec["doc_lengths"] == [spec["doc_length"]] * spec["n_docs"]


def test_cell_inputs_deterministic_and_mask_aligned():
    first = cell_inputs(2, 128)
    second = cell_inputs(2, 128)
    assert first == second  # seed-47 fixed tokens per cell

    total = 2 * 128 + QUERY_LENGTH
    assert len(first["tokens"]) == total
    assert len(first["positions"]) == total
    assert first["positions"] == build_positions([128, 128], QUERY_LENGTH, 0, "restart").tolist()
    assert min(first["tokens"]) >= 100
    assert max(first["tokens"]) < 30_000

    allow = build_mask([128, 128], QUERY_LENGTH, 0, "ddi_qf")
    additive = to_additive(allow)
    assert allow.shape == (total, total)
    assert additive.shape == (total, total)


def test_collect_source_hashes_covers_ddi_and_runner():
    hashes = collect_source_hashes()
    assert "ddi/masks.py" in hashes
    assert "ddi/segmented_fa2.py" in hashes
    assert "scripts/run_segmented_fa2_timing.py" in hashes
    expected = hashlib.sha256((ROOT / "ddi" / "segmented_fa2.py").read_bytes()).hexdigest()
    assert hashes["ddi/segmented_fa2.py"] == expected


def test_collect_checkpoint_identity(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen2"}')
    identity = collect_checkpoint_identity(tmp_path)
    assert identity["config_sha256"] == hashlib.sha256(b'{"model_type": "qwen2"}').hexdigest()
    assert identity["weights"] == []
    assert identity["weights_bytes"] == 0
    with pytest.raises(FileNotFoundError):
        collect_checkpoint_identity(tmp_path / "missing")


def _synthetic_ok_cell():
    reference = [1.0 + 0.01 * index for index in range(REPEATS)]
    candidate = [0.8] * REPEATS
    ratio = paired_bootstrap_ratio(reference, candidate)
    return {
        "status": "ok",
        "n_docs": 2,
        "doc_length": 128,
        "doc_lengths": [128, 128],
        "query_length": QUERY_LENGTH,
        "total_tokens": 2 * 128 + QUERY_LENGTH,
        "tokens_sha256": "0" * 64,
        "positions_sha256": "0" * 64,
        "reference": {
            "backend": "sdpa",
            "repeats_seconds": reference,
            "median_seconds": statistics.median(reference),
        },
        "candidate": {
            "backend": "segmented_fa2",
            "repeats_seconds": candidate,
            "median_seconds": statistics.median(candidate),
        },
        "ratio": ratio,
        "gate": {"threshold": GATE_THRESHOLD, "pass": gate_pass(ratio)},
    }


def _synthetic_error_cell():
    cell = _synthetic_ok_cell()
    cell["status"] = "error"
    cell["error"] = "RuntimeError: out of memory"
    cell["gate"] = {**cell["gate"], "pass": False}
    return cell


def test_validate_cell_result_accepts_synthetic_ok_cell():
    validate_cell_result(_synthetic_ok_cell())  # must not raise


def test_validate_cell_result_rejects_broken_cells():
    cell = _synthetic_ok_cell()
    with pytest.raises(ValueError):
        validate_cell_result({key: value for key, value in cell.items() if key != "ratio"})
    with pytest.raises(ValueError):
        validate_cell_result(
            {**cell, "reference": {**cell["reference"], "repeats_seconds": []}}
        )
    with pytest.raises(ValueError):
        validate_cell_result(
            {
                **cell,
                "ratio": {**cell["ratio"], "ci95_low": cell["ratio"]["ci95_high"] + 1.0},
            }
        )
    with pytest.raises(ValueError):
        validate_cell_result(
            {**cell, "gate": {**cell["gate"], "pass": not cell["gate"]["pass"]}}
        )
    with pytest.raises(ValueError):
        validate_cell_result({**cell, "total_tokens": 1})


def test_summary_entry_ok_and_error():
    ok = _summary_entry(_synthetic_ok_cell())
    assert ok["status"] == "ok"
    assert ok["gate_pass"] is True
    assert ok["ratio_ci95"][0] <= ok["ratio_mean"] <= ok["ratio_ci95"][1]
    assert "error" not in ok

    error = _summary_entry(_synthetic_error_cell())
    assert error["status"] == "error"
    assert error["gate_pass"] is False
    assert error["error"] == "RuntimeError: out of memory"
    assert "ratio_mean" not in error


def test_gate_overall_requires_every_cell():
    ok = _synthetic_ok_cell()
    assert gate_overall([ok]) is True
    assert gate_overall([ok, ok]) is True
    assert gate_overall([ok, _synthetic_error_cell()]) is False
    assert gate_overall([]) is False
    failing = {**ok, "gate": {**ok["gate"], "pass": False}}
    assert gate_overall([failing]) is False


def test_reference_arm_prefill_smoke_on_cpu_tiny_model():
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except Exception as error:  # Local optional vision stack can break lazy Qwen imports.
        pytest.skip(f"tiny Qwen runtime unavailable: {error}")

    from ddi.model import patch_model

    torch.manual_seed(SEED)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    model = patch_model(Qwen2ForCausalLM(config).eval())

    generator = torch.Generator().manual_seed(SEED)
    doc_length = 16
    doc_lengths = [doc_length, doc_length]
    tokens = torch.randint(1, 90, (sum(doc_lengths) + QUERY_LENGTH,), generator=generator)
    ids = tokens.unsqueeze(0)
    positions = build_positions(doc_lengths, QUERY_LENGTH, 0, "restart").unsqueeze(0)
    additive = to_additive(
        build_mask(doc_lengths, QUERY_LENGTH, 0, "ddi_qf"), dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    seconds = run_arm_timings(model, ids, additive, positions, warmups=0, repeats=1)
    assert len(seconds) == 1
    assert seconds[0] > 0.0
