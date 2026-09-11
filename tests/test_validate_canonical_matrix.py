"""CPU tests for the V60 matrix provenance and attribution schema."""

from __future__ import annotations

import copy
import hashlib

import pytest

from scripts.validate_canonical_matrix import (
    ARMS,
    SCHEMA_VERSION,
    arm_identity,
    validate_comparison,
    validate_manifest,
    validate_result,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "data_role": "confirmation_frozen",
        "provenance": {
            "source_revision": "arc:deadbeef",
            "source_tree_sha256": digest("source"),
            "runtime_sha256": digest("runtime"),
            "checkpoint_sha256": digest("checkpoint"),
            "tokenizer_sha256": digest("tokenizer"),
            "hardware_sha256": digest("hardware"),
            "frozen_manifest_label": "heldout-v60-01",
        },
        "arms": {
            arm: {**identity, "dtype": "torch.bfloat16", "layout": "restart"}
            for arm, identity in ARMS.items()
        },
        "requests": [
            {
                "request_id": "request-1",
                "confirmation_data_label": "heldout-v60-01/request-1",
                "document_token_sha256": [digest("doc-1"), digest("doc-2")],
                "query_token_sha256": digest("query"),
                "input_sha256": digest("input"),
            }
        ],
    }


def result(item: dict, arm: str = "C") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": validate_manifest(item),
        "request_id": "request-1",
        "arm": arm,
        "arm_identity": arm_identity(arm, item["arms"][arm]),
        "hit_state": [True, True] if arm == "C" else [False, False],
        "natural_output": {"token_ids_sha256": digest("natural"), "token_count": 4},
        "fixed_output": {"token_ids_sha256": digest("fixed"), "token_count": 4},
        "timing_ms": [1.0, 1.1, 1.2, 1.3, 1.4],
        "data_label": "heldout-v60-01/request-1",
    }


def test_manifest_accepts_complete_immutable_provenance() -> None:
    assert len(validate_manifest(manifest())) == 64


def test_manifest_rejects_reused_confirmation_label() -> None:
    item = manifest()
    duplicate = copy.deepcopy(item["requests"][0])
    duplicate["request_id"] = "request-2"
    item["requests"].append(duplicate)
    with pytest.raises(ValueError, match="reused"):
        validate_manifest(item)


def test_arm_identity_rejects_backend_substitution() -> None:
    item = manifest()
    item["arms"]["B"]["backend"] = "sdpa"
    with pytest.raises(ValueError, match="B identity"):
        validate_manifest(item)


def test_result_requires_outputs_hits_raw_repeats_and_matching_label() -> None:
    item = manifest()
    row = result(item)
    validate_result(row, item)
    row["timing_ms"] = [1.0]
    with pytest.raises(ValueError, match="five raw repeats"):
        validate_result(row, item)


def test_result_rejects_reused_or_missing_confirmation_label() -> None:
    item = manifest()
    row = result(item)
    row["data_label"] = "development_observed_V58"
    with pytest.raises(ValueError, match="label"):
        validate_result(row, item)


def test_only_bc_cache_reuse_comparison_is_admissible() -> None:
    item = manifest()
    comparison = {
        "left_arm": "B",
        "right_arm": "C",
        "kind": "cache_reuse",
        "data_labels": ["heldout-v60-01/request-1"],
    }
    validate_comparison(comparison, item)
    comparison.update(left_arm="A", kind="performance")
    with pytest.raises(ValueError, match="C/B reuse is not C/A performance"):
        validate_comparison(comparison, item)
