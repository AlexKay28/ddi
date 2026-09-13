"""Validate the V61 pinned full-model parity manifest and runner result.

This is deliberately CPU-only plumbing.  It neither imports a model nor runs
attention: a future GPU runner owns producing the JSON result that this module
accepts.  In particular, this validator must not be used to reinterpret the
historical V57 Flex failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "v61_pinned_segmented_parity_v1"
VARIANTS = ("full", "ddi_qf", "blind")
DOC_COUNTS = (1, 2)
COSINE_GATE = 0.999


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _object(value: Any, path: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    actual = set(value)
    if actual != keys:
        _fail(path, f"must contain exactly {sorted(keys)}, got {sorted(actual)}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        _fail(path, "must be a finite number")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _provenance(value: Any, path: str) -> None:
    fields = _object(value, path, {"backend", "implementation_revision", "source_revision"})
    for name, field in fields.items():
        _string(field, f"{path}.{name}")


def _runtime(value: Any, path: str) -> None:
    fields = _object(value, path, {"python", "torch", "cuda", "device"})
    for name, field in fields.items():
        _string(field, f"{path}.{name}")


def _pinned_sequence(value: Any, path: str, length: int) -> str:
    fields = _object(value, path, {"ids", "sha256"})
    ids = fields["ids"]
    if not isinstance(ids, list) or len(ids) != length:
        _fail(f"{path}.ids", f"must be a list with {length} entries")
    for index, token in enumerate(ids):
        _integer(token, f"{path}.ids[{index}]")
    digest = _string(fields["sha256"], f"{path}.sha256")
    if digest != _digest(ids):
        _fail(f"{path}.sha256", "does not match ids")
    return digest


def _mask_digest(
    variant: str, doc_lengths: list[int], query_length: int, decode_length: int
) -> str:
    # This stable semantic fingerprint binds runner output to build_mask inputs.
    return _digest(
        {
            "implementation": "ddi.masks.build_mask",
            "variant": variant,
            "doc_lengths": doc_lengths,
            "query_length": query_length,
            "decode_length": decode_length,
        }
    )


def _manifest_cell(value: Any, path: str) -> tuple[str, int, str, str, str, int]:
    fields = _object(
        value,
        path,
        {
            "variant",
            "n_docs",
            "doc_lengths",
            "query_length",
            "decode_length",
            "tokens",
            "positions",
            "mask",
        },
    )
    variant = fields["variant"]
    if variant not in VARIANTS:
        _fail(f"{path}.variant", f"must be one of {VARIANTS}")
    n_docs = _integer(fields["n_docs"], f"{path}.n_docs", minimum=1)
    lengths = fields["doc_lengths"]
    if not isinstance(lengths, list) or len(lengths) != n_docs:
        _fail(f"{path}.doc_lengths", "must contain one positive length per document")
    for index, length in enumerate(lengths):
        _integer(length, f"{path}.doc_lengths[{index}]", minimum=1)
    query_length = _integer(fields["query_length"], f"{path}.query_length", minimum=1)
    decode_length = _integer(fields["decode_length"], f"{path}.decode_length", minimum=1)
    total = sum(lengths) + query_length
    token_digest = _pinned_sequence(fields["tokens"], f"{path}.tokens", total)
    position_digest = _pinned_sequence(fields["positions"], f"{path}.positions", total)
    mask = _object(fields["mask"], f"{path}.mask", {"implementation", "sha256"})
    if mask["implementation"] != "ddi.masks.build_mask":
        _fail(f"{path}.mask.implementation", "must be ddi.masks.build_mask")
    mask_digest = _string(mask["sha256"], f"{path}.mask.sha256")
    if mask_digest != _mask_digest(variant, lengths, query_length, decode_length):
        _fail(f"{path}.mask.sha256", "does not match the pinned layout")
    return variant, n_docs, token_digest, position_digest, mask_digest, total


def validate_manifest(manifest: Any) -> None:
    """Require the exact V61 six-cell matrix and all runner-relevant pins."""
    fields = _object(
        manifest,
        "manifest",
        {"schema_version", "manifest_id", "reference", "candidate", "runtime", "cells"},
    )
    if fields["schema_version"] != SCHEMA_VERSION:
        _fail("manifest.schema_version", f"must be {SCHEMA_VERSION}")
    _string(fields["manifest_id"], "manifest.manifest_id")
    _provenance(fields["reference"], "manifest.reference")
    if fields["reference"]["backend"] != "sdpa":
        _fail("manifest.reference.backend", "must be sdpa")
    _provenance(fields["candidate"], "manifest.candidate")
    _runtime(fields["runtime"], "manifest.runtime")
    cells = fields["cells"]
    if not isinstance(cells, list) or len(cells) != len(VARIANTS) * len(DOC_COUNTS):
        _fail("manifest.cells", "must contain exactly six cells")
    observed = set()
    for index, cell in enumerate(cells):
        variant, n_docs, *_ = _manifest_cell(cell, f"manifest.cells[{index}]")
        observed.add((variant, n_docs))
    required = {(variant, count) for variant in VARIANTS for count in DOC_COUNTS}
    if observed != required:
        _fail("manifest.cells", "must cover full/ddi_qf/blind for N=1 and N=2 exactly once")


def _result_cell(value: Any, path: str, expected: tuple[str, int, str, str, str, int]) -> None:
    fields = _object(
        value,
        path,
        {
            "variant",
            "n_docs",
            "tokens_sha256",
            "positions_sha256",
            "mask_sha256",
            "prefill",
            "decode",
        },
    )
    actual = (
        fields["variant"],
        fields["n_docs"],
        fields["tokens_sha256"],
        fields["positions_sha256"],
        fields["mask_sha256"],
    )
    if actual != expected[:5]:
        _fail(path, "does not match its pinned manifest cell or provenance")
    prefill = _object(
        fields["prefill"],
        f"{path}.prefill",
        {
            "reference_finite",
            "candidate_finite",
            "cosine_similarity",
            "token_argmax_agreement",
            "reference_cache_length",
            "candidate_cache_length",
        },
    )
    if not _boolean(prefill["reference_finite"], f"{path}.prefill.reference_finite"):
        _fail(path, "reference prefill logits are non-finite")
    if not _boolean(prefill["candidate_finite"], f"{path}.prefill.candidate_finite"):
        _fail(path, "candidate prefill logits are non-finite")
    if _number(prefill["cosine_similarity"], f"{path}.prefill.cosine_similarity") < COSINE_GATE:
        _fail(path, f"prefill cosine similarity must be >= {COSINE_GATE}")
    if _number(prefill["token_argmax_agreement"], f"{path}.prefill.token_argmax_agreement") != 1.0:
        _fail(path, "prefill token argmax agreement must be 1.0")
    reference_length = _integer(
        prefill["reference_cache_length"], f"{path}.prefill.reference_cache_length"
    )
    candidate_length = _integer(
        prefill["candidate_cache_length"], f"{path}.prefill.candidate_cache_length"
    )
    if reference_length != candidate_length or reference_length != expected[5]:
        _fail(path, "prefill cache lengths must agree with the pinned input length")
    decode = _object(
        fields["decode"],
        f"{path}.decode",
        {"reference_finite", "candidate_finite", "reference_argmax", "candidate_argmax"},
    )
    if not _boolean(decode["reference_finite"], f"{path}.decode.reference_finite"):
        _fail(path, "reference decode logits are non-finite")
    if not _boolean(decode["candidate_finite"], f"{path}.decode.candidate_finite"):
        _fail(path, "candidate decode logits are non-finite")
    if _integer(decode["reference_argmax"], f"{path}.decode.reference_argmax") != _integer(
        decode["candidate_argmax"], f"{path}.decode.candidate_argmax"
    ):
        _fail(path, "decode argmax differs")


def validate_result(manifest: Any, result: Any) -> None:
    """Validate a GPU runner result against a separately validated manifest."""
    validate_manifest(manifest)
    fields = _object(
        result,
        "result",
        {"schema_version", "manifest_id", "reference", "candidate", "runtime", "cells"},
    )
    if fields["schema_version"] != SCHEMA_VERSION:
        _fail("result.schema_version", f"must be {SCHEMA_VERSION}")
    if fields["manifest_id"] != manifest["manifest_id"]:
        _fail("result.manifest_id", "does not match manifest")
    _provenance(fields["reference"], "result.reference")
    _provenance(fields["candidate"], "result.candidate")
    _runtime(fields["runtime"], "result.runtime")
    if fields["reference"] != manifest["reference"] or fields["candidate"] != manifest["candidate"]:
        _fail("result", "backend provenance does not match manifest")
    if fields["runtime"] != manifest["runtime"]:
        _fail("result", "runtime provenance does not match manifest")
    expected_cells = [_manifest_cell(cell, "manifest.cells") for cell in manifest["cells"]]
    cells = fields["cells"]
    if not isinstance(cells, list) or len(cells) != len(expected_cells):
        _fail("result.cells", "must contain one result for every manifest cell")
    for index, (cell, expected) in enumerate(zip(cells, expected_cells, strict=True)):
        _result_cell(cell, f"result.cells[{index}]", expected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate V61 pinned parity JSON without running a model"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("result", type=Path, nargs="?")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.result is None:
        validate_manifest(manifest)
        print("manifest valid")
    else:
        validate_result(manifest, json.loads(args.result.read_text()))
        print("result valid")


if __name__ == "__main__":
    main()
