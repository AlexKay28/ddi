"""CPU end-to-end tests for the V60 development-matrix runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

try:
    from scripts.run_canonical_matrix import _decode_arm, main
except ModuleNotFoundError as error:
    # This sandbox has an incompatible optional stack; the parent environment runs the
    # real tiny-Qwen tests below.
    if "PreTrainedModel" not in str(error):
        raise
    pytest.skip(f"canonical matrix runtime unavailable: {error}", allow_module_level=True)


def _model(backend: str):
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from ddi.model import patch_model
    except Exception as error:  # Local optional vision stack can make Qwen lazy import unavailable.
        pytest.skip(f"tiny Qwen runtime unavailable: {error}")
    torch.manual_seed(58)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    config._attn_implementation = backend
    return patch_model(Qwen2ForCausalLM(config).eval())


def _loader():
    def load_models(_checkpoint, _device):
        return _model("sdpa"), _model("eager")

    return load_models


def _write_inputs(path: Path) -> Path:
    payload = {
        "seed": 58,
        "docs": [list(range(2, 10)), list(range(10, 18))],
        "query_ids": list(range(18, 22)),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _argv(output: Path, inputs: Path) -> list[str]:
    return [
        "--checkpoint",
        "tiny-checkpoint",
        "--inputs",
        str(inputs),
        "--output",
        str(output),
        "--decode-length",
        "3",
        "--repeats",
        "2",
        "--device",
        "cpu",
    ]


def test_decode_arm_records_divergence_and_stays_teacher_forced():
    seen: list[list[int]] = []
    proposals = iter([6, 9])

    def step(prefix: list[int]) -> int:
        seen.append(list(prefix))
        return next(proposals)

    decode_ids, divergence = _decode_arm(step, [5, 6, 7])
    assert decode_ids == [5, 6, 9]
    assert divergence == 2
    assert seen == [[5], [5, 6]]


def test_tiny_matrix_end_to_end(tmp_path):
    torch.set_num_threads(1)
    inputs = _write_inputs(tmp_path / "inputs.json")
    output = tmp_path / "out"
    main(_argv(output, inputs), load_models=_loader())

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 60
    assert manifest["data_role"] == "development"
    assert manifest["runtime"]["gpu_name"] is None
    assert manifest["runtime"]["python"] and manifest["runtime"]["torch"]
    assert manifest["runtime"]["transformers"]
    assert manifest["checkpoint"].endswith("tiny-checkpoint")
    assert manifest["provenance"]["git_base_sha256"]
    assert manifest["provenance"]["git_diff_head_sha256"]
    source_files = manifest["provenance"]["source_files"]
    assert "scripts/run_canonical_matrix.py" in source_files
    assert "ddi/cache.py" in source_files
    assert "ddi/canonical_cache.py" in source_files
    assert set(manifest["arms"]) == {"A", "B", "C", "D"}
    assert all(len(arm["arm_identity"]) == 64 for arm in manifest["arms"].values())
    assert manifest["arms"]["C"]["cache"] == "persisted"

    lines = (output / "cells.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    cell = json.loads(lines[0])
    assert cell["seed"] == 58 and cell["request_index"] == 0
    assert cell["doc_lengths"] == [8, 8] and cell["query_len"] == 4
    arms = cell["arms"]
    assert set(arms) == {"A", "B", "C", "D"}
    assert set(arms["C"]) == {"cold", "all_hit"}

    a, b, d = arms["A"], arms["B"], arms["D"]
    c_cold, c_hit = arms["C"]["cold"], arms["C"]["all_hit"]
    assert b["logits_sha256"] == c_cold["logits_sha256"] == c_hit["logits_sha256"]
    assert b["decode_ids"] == c_cold["decode_ids"] == c_hit["decode_ids"]
    assert all(cell["b_vs_c_equal"].values())
    assert isinstance(cell["a_vs_b_equal_logits"], bool)

    assert a["decode_ids"] == cell["teacher_token_ids"]
    assert a["decode_divergence_step"] is None
    assert len(cell["teacher_token_ids"]) == 3
    assert len(d["decode_ids"]) == 3
    assert d["decode_divergence_step"] is None or isinstance(d["decode_divergence_step"], int)
    for record in (a, b, d, c_cold, c_hit):
        assert len(record["ttft_seconds"]) == 2
        assert record["ttft_median_seconds"] >= 0.0
        assert record["peak_allocated_bytes"] is None
        assert record["peak_reserved_bytes"] is None
        assert record["decode_ids"][0] == a["prefill_argmax"]
        assert len(record["arm_identity"]) == 64
    for record in (a, b, d):
        assert record["cache_bytes"] == 0
        assert record["hit_state"] == [False, False]
    assert c_cold["hit_state"] == [False, False]
    assert c_cold["cache_bytes"] > 0
    assert c_hit["hit_state"] == [True, True]
    assert c_hit["cache_bytes"] > 0
    assert (output / "canonical_cache" / "all_hit").is_dir()

    result = json.loads((output / "result.json").read_text())
    manifest_sha = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    assert result["manifest_sha256"] == manifest_sha
    assert result["data_role"] == "development"
    assert set(result["aggregate"]["ttft_median_seconds"]) == {
        "A", "B", "C_cold", "C_all_hit", "D",
    }
    assert result["aggregate"]["b_vs_c_equal_all"] is True
    counts = result["aggregate"]["argmax_agreement"]["B"]
    assert counts["decode_steps"] == 3
    assert counts["decode_step_matches"] <= counts["decode_steps"]
    assert result["requests"][0]["a_vs_b_equal_logits"] == (
        a["logits_sha256"] == b["logits_sha256"]
    )

    with pytest.raises(FileExistsError):
        main(_argv(output, inputs), load_models=_loader())


def test_invalid_inputs_rejected_before_model_load(tmp_path):
    inputs = tmp_path / "bad.json"
    inputs.write_text(json.dumps({"seed": 58, "docs": [], "query_ids": [1]}))

    def failing_loader(_checkpoint, _device):
        raise AssertionError("loader must not be called for invalid inputs")

    with pytest.raises(ValueError, match="docs"):
        main(
            [
                "--checkpoint", "x",
                "--inputs", str(inputs),
                "--output", str(tmp_path / "out"),
                "--device", "cpu",
            ],
            load_models=failing_loader,
        )
    assert not (tmp_path / "out").exists()


def test_pad_token_inputs_rejected(tmp_path):
    torch.set_num_threads(1)
    inputs = tmp_path / "pad.json"
    inputs.write_text(json.dumps({"seed": 58, "docs": [[0, 1]], "query_ids": [2]}))
    with pytest.raises(ValueError, match="pad"):
        main(
            [
                "--checkpoint", "x",
                "--inputs", str(inputs),
                "--output", str(tmp_path / "out"),
                "--device", "cpu",
            ],
            load_models=_loader(),
        )


def test_cli_rejects_bad_flags(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--checkpoint", "x",
                "--inputs", "x",
                "--output", str(tmp_path / "out"),
                "--decode-length", "0",
                "--device", "cpu",
            ],
        )
