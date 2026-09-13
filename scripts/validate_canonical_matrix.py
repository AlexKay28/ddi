"""CPU-only V60 A/B/C/D matrix manifest and result-schema validator.

This module deliberately does not load a model, write artifacts, or execute a
benchmark.  A GPU runner must supply a frozen manifest and records that pass
these provenance and attribution checks before interpreting its observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 60
ARMS = {
    "A": {"backend": "sdpa", "execution": "original_monolithic_ddi_qf", "cache": "none"},
    "B": {"backend": "eager", "execution": "canonical_per_document_ddi_qf", "cache": "disabled"},
    "C": {"backend": "eager", "execution": "canonical_per_document_ddi_qf", "cache": "persisted"},
    "D": {"backend": "sdpa", "execution": "full_causal_reference", "cache": "none"},
}
HEX = set("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fail(message: str) -> None:
    raise ValueError(message)


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    if set(value) != keys:
        _fail(f"{name} keys must be exactly {sorted(keys)}")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - HEX:
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value


def arm_identity(arm: str, identity: Mapping[str, Any]) -> str:
    """Return the immutable identity of one explicitly specified matrix arm."""
    if arm not in ARMS:
        _fail("unknown arm")
    item = _object(identity, "arm identity")
    _require_keys(item, {"backend", "execution", "cache", "dtype", "layout"}, "arm identity")
    expected = ARMS[arm]
    if any(item[key] != expected[key] for key in expected):
        _fail(f"{arm} identity does not match its fixed backend/execution contract")
    _text(item["dtype"], "dtype")
    _text(item["layout"], "layout")
    return hashlib.sha256(_canonical({"arm": arm, **item})).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> str:
    """Validate an immutable confirmation manifest and return its content digest."""
    item = _object(manifest, "manifest")
    _require_keys(
        item,
        {"schema_version", "data_role", "provenance", "arms", "requests"},
        "manifest",
    )
    if item["schema_version"] != SCHEMA_VERSION or item["data_role"] != "confirmation_frozen":
        _fail("manifest must be V60 confirmation_frozen data")
    provenance = _object(item["provenance"], "provenance")
    _require_keys(
        provenance,
        {
            "source_revision", "source_tree_sha256", "runtime_sha256", "checkpoint_sha256",
            "tokenizer_sha256", "hardware_sha256", "frozen_manifest_label",
        },
        "provenance",
    )
    for key in set(provenance) - {"source_revision", "frozen_manifest_label"}:
        _sha(provenance[key], f"provenance.{key}")
    _text(provenance["source_revision"], "provenance.source_revision")
    _text(provenance["frozen_manifest_label"], "provenance.frozen_manifest_label")
    arms = _object(item["arms"], "arms")
    if set(arms) != set(ARMS):
        _fail("manifest must include exactly A, B, C, and D")
    for arm, identity in arms.items():
        arm_identity(arm, _object(identity, f"arms.{arm}"))
    requests = item["requests"]
    if not isinstance(requests, list) or not requests:
        _fail("manifest.requests must be a non-empty list")
    labels: set[str] = set()
    request_ids: set[str] = set()
    for request in requests:
        row = _object(request, "request")
        _require_keys(
            row,
            {
                "request_id",
                "confirmation_data_label",
                "document_token_sha256",
                "query_token_sha256",
                "input_sha256",
            },
            "request",
        )
        request_id = _text(row["request_id"], "request.request_id")
        label = _text(row["confirmation_data_label"], "request.confirmation_data_label")
        if request_id in request_ids or label in labels:
            _fail("request IDs and confirmation data labels must not be reused")
        request_ids.add(request_id)
        labels.add(label)
        if not isinstance(row["document_token_sha256"], list) or not row["document_token_sha256"]:
            _fail("request.document_token_sha256 must be a non-empty list")
        hashes = [*row["document_token_sha256"], row["query_token_sha256"], row["input_sha256"]]
        for digest in hashes:
            _sha(digest, "per-request token/input hash")
    return hashlib.sha256(_canonical(item)).hexdigest()


def validate_result(result: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Validate one per-request arm record without interpreting model outputs."""
    manifest_digest = validate_manifest(manifest)
    row = _object(result, "result")
    _require_keys(
        row,
        {
            "schema_version", "manifest_sha256", "request_id", "arm", "arm_identity",
            "hit_state", "natural_output", "fixed_output", "timing_ms", "data_label",
        },
        "result",
    )
    if row["schema_version"] != SCHEMA_VERSION:
        _fail("result schema version mismatch")
    if row["manifest_sha256"] != manifest_digest:
        _fail("result does not bind to this immutable manifest")
    arm = row["arm"]
    if arm not in ARMS:
        _fail("result arm is unknown")
    requests = {request["request_id"]: request for request in manifest["requests"]}
    if row["request_id"] not in requests:
        _fail("result request is absent from manifest")
    if row["data_label"] != requests[row["request_id"]]["confirmation_data_label"]:
        _fail("result confirmation data label is missing or reused")
    if row["arm_identity"] != arm_identity(arm, manifest["arms"][arm]):
        _fail("result arm identity mismatch")
    hits = row["hit_state"]
    if not isinstance(hits, list) or not hits or any(type(hit) is not bool for hit in hits):
        _fail("hit_state must be a non-empty boolean list")
    if arm != "C" and any(hits):
        _fail("only C may report persisted-cache hits")
    if arm == "C" and not all(hits):
        _fail("C result must explicitly record all persisted-cache hits")
    for field in ("natural_output", "fixed_output"):
        output = _object(row[field], field)
        _require_keys(output, {"token_ids_sha256", "token_count"}, field)
        _sha(output["token_ids_sha256"], f"{field}.token_ids_sha256")
        if type(output["token_count"]) is not int or output["token_count"] < 0:
            _fail(f"{field}.token_count must be nonnegative")
    timings = row["timing_ms"]
    if not isinstance(timings, list) or len(timings) < 5:
        _fail("timing_ms must retain at least five raw repeats")
    if any(type(value) not in {int, float} or value < 0 for value in timings):
        _fail("timing_ms must contain nonnegative numbers")


def validate_comparison(comparison: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Reject performance claims that cross the A/B backend boundary."""
    validate_manifest(manifest)
    row = _object(comparison, "comparison")
    _require_keys(row, {"left_arm", "right_arm", "kind", "data_labels"}, "comparison")
    left, right, kind = row["left_arm"], row["right_arm"], row["kind"]
    if left not in ARMS or right not in ARMS or left == right:
        _fail("comparison must name two different known arms")
    if not isinstance(row["data_labels"], list) or not row["data_labels"]:
        _fail("comparison must retain confirmation data labels")
    labels = {request["confirmation_data_label"] for request in manifest["requests"]}
    if len(row["data_labels"]) != len(set(row["data_labels"])) or set(row["data_labels"]) - labels:
        _fail("comparison has missing or reused confirmation data labels")
    pair = frozenset((left, right))
    if kind == "cache_reuse" and pair == frozenset(("B", "C")):
        return
    if kind == "performance":
        _fail(
            "performance comparisons are unavailable: A/B backends differ; "
            "C/B reuse is not C/A performance"
        )
    _fail("unsupported comparison kind/arm pair")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=argparse.FileType("r"))
    parser.add_argument("--result", type=argparse.FileType("r"))
    parser.add_argument("--comparison", type=argparse.FileType("r"))
    args = parser.parse_args()
    manifest = json.load(args.manifest)
    digest = validate_manifest(manifest)
    if args.result:
        validate_result(json.load(args.result), manifest)
    if args.comparison:
        validate_comparison(json.load(args.comparison), manifest)
    print(json.dumps({"manifest_sha256": digest, "valid": True}))


if __name__ == "__main__":
    main()
