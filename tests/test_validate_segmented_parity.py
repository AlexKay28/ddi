from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.validate_segmented_parity import (
    SCHEMA_VERSION,
    VARIANTS,
    _digest,
    _mask_digest,
    validate_manifest,
    validate_result,
)


def _cell(variant: str, n_docs: int) -> dict:
    lengths = [2] * n_docs
    total = sum(lengths) + 3
    tokens = list(range(total))
    positions = list(range(10, 10 + total))
    return {
        "variant": variant,
        "n_docs": n_docs,
        "doc_lengths": lengths,
        "query_length": 3,
        "decode_length": 1,
        "tokens": {"ids": tokens, "sha256": _digest(tokens)},
        "positions": {"ids": positions, "sha256": _digest(positions)},
        "mask": {
            "implementation": "ddi.masks.build_mask",
            "sha256": _mask_digest(variant, lengths, 3, 1),
        },
    }


def _manifest() -> dict:
    provenance = {
        "backend": "sdpa",
        "implementation_revision": "source-a",
        "source_revision": "arc-a",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": "v61-fixed-inputs",
        "reference": provenance,
        "candidate": {**provenance, "backend": "flex"},
        "runtime": {"python": "3.12.0", "torch": "2.5.1", "cuda": "12.1", "device": "A100"},
        "cells": [_cell(variant, n_docs) for n_docs in (1, 2) for variant in VARIANTS],
    }


def _result(manifest: dict) -> dict:
    cells = []
    for cell in manifest["cells"]:
        cells.append(
            {
                "variant": cell["variant"],
                "n_docs": cell["n_docs"],
                "tokens_sha256": cell["tokens"]["sha256"],
                "positions_sha256": cell["positions"]["sha256"],
                "mask_sha256": cell["mask"]["sha256"],
                "prefill": {
                    "reference_finite": True,
                    "candidate_finite": True,
                    "cosine_similarity": 1.0,
                    "token_argmax_agreement": 1.0,
                    "reference_cache_length": sum(cell["doc_lengths"]) + cell["query_length"],
                    "candidate_cache_length": sum(cell["doc_lengths"]) + cell["query_length"],
                },
                "decode": {
                    "reference_finite": True,
                    "candidate_finite": True,
                    "reference_argmax": 7,
                    "candidate_argmax": 7,
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "reference": manifest["reference"],
        "candidate": manifest["candidate"],
        "runtime": manifest["runtime"],
        "cells": cells,
    }


def test_accepts_complete_pinned_six_cell_result() -> None:
    manifest = _manifest()
    validate_manifest(manifest)
    validate_result(manifest, _result(manifest))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda manifest, result: result.pop("candidate"), "candidate"),
        (lambda manifest, result: result.pop("runtime"), "runtime"),
        (lambda manifest, result: result["cells"][0].pop("tokens_sha256"), "tokens_sha256"),
        (lambda manifest, result: result["cells"][0].pop("mask_sha256"), "mask_sha256"),
        (
            lambda manifest, result: result["cells"][0]["prefill"].update(candidate_finite=False),
            "non-finite",
        ),
        (
            lambda manifest, result: result["cells"][0]["prefill"].update(cosine_similarity=0.998),
            "cosine",
        ),
        (
            lambda manifest, result: result["cells"][0]["prefill"].update(
                token_argmax_agreement=0.99
            ),
            "argmax",
        ),
        (
            lambda manifest, result: result["cells"][0]["prefill"].update(
                reference_cache_length=1, candidate_cache_length=1
            ),
            "cache lengths",
        ),
        (
            lambda manifest, result: result["cells"][0]["decode"].update(candidate_argmax=8),
            "decode argmax",
        ),
    ],
)
def test_rejects_missing_provenance_and_failed_hard_gates(mutate, match: str) -> None:
    manifest = _manifest()
    result = _result(manifest)
    mutate(manifest, result)
    with pytest.raises(ValueError, match=match):
        validate_result(manifest, result)


def test_rejects_manifest_without_exact_matrix_or_pinned_token_digest() -> None:
    manifest = _manifest()
    manifest["cells"].pop()
    with pytest.raises(ValueError, match="six cells"):
        validate_manifest(manifest)
    manifest = _manifest()
    manifest["cells"][0]["tokens"]["ids"][0] = 99
    with pytest.raises(ValueError, match="does not match ids"):
        validate_manifest(manifest)


def test_rejects_runtime_provenance_drift() -> None:
    manifest = _manifest()
    result = deepcopy(_result(manifest))
    result["runtime"]["torch"] = "2.6.0"
    with pytest.raises(ValueError, match="runtime provenance"):
        validate_result(manifest, result)
