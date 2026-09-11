"""Fail-closed analysis of the V70 confirmation300 evaluation artifact.

Reads the confirmation evaluation JSON, revalidates every frozen preregistration
binding (schema, split, record IDs against ``experiments/v70_confirmation300.json``,
cache/persistence evidence, embedded gates), recomputes the preregistered C-B and
C-D gates with ``scripts/v70_harness.py`` helpers, and writes ``verdict.json`` /
``verdict.md`` / ``SHA256SUMS`` into a new ``results/v70_confirmation_analysis_<UTC>``
directory.  Any validation failure aborts before any output is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

# This host can carry a mismatched optional torchvision wheel; transformers only needs
# one enum from it, so reproduce the isolation used by tests/test_v70_harness.py.
try:
    import torchvision  # noqa: F401
except (ModuleNotFoundError, RuntimeError):
    from enum import Enum
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    for _name in tuple(sys.modules):
        if _name == "torchvision" or _name.startswith("torchvision."):
            del sys.modules[_name]

    class InterpolationMode(Enum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BOX = 2
        BILINEAR = 3
        HAMMING = 4
        BICUBIC = 5
        LANCZOS = 6

    _torchvision = ModuleType("torchvision")
    _torchvision.__spec__ = ModuleSpec("torchvision", loader=None)
    _transforms = ModuleType("torchvision.transforms")
    _transforms.InterpolationMode = InterpolationMode
    _torchvision.transforms = _transforms
    sys.modules["torchvision"] = _torchvision
    sys.modules["torchvision.transforms"] = _transforms

from scripts.v70_harness import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    NONINFERIORITY_MARGIN,
    paired_quality_gate,
)

ANALYSIS_SCHEMA = "v70-confirmation-verdict-v1"
DEFAULT_INPUT = ROOT / "results/v70_confirmation300_k5.json"
DEFAULT_MANIFEST = ROOT / "experiments/v70_confirmation300.json"
DEFAULT_DIVERGENT = ROOT / "results/v68_anatomy_20260910T131036Z/differing_records.jsonl"

EXPECTED_SCHEMA = "v70-evaluation-v2"
EXPECTED_SPLIT = "confirmation300"
EXPECTED_DATA_ROLE = "confirmation_primary"
EXPECTED_RECORD_COUNT = 300
EXPECTED_K = 5
EXPECTED_ORIGINAL_CONFIRMATION100_COUNT = 100
EXPECTED_CONFIRMATION_SHA256 = (
    "ba7263ae886d929edb63595bfb693e1579269c1814969dffb934d55cf13051d1"
)
EXPECTED_ORIGINAL_CONFIRMATION100_SHA256 = (
    "529f0109eefbf2ef42ede00be146d95cf19876e35859bc1358fbceb1b1bf04d7"
)
EXPECTED_SELECTED_CHECKPOINT_SHA256 = (
    "04fe4dcb3c70f0ef75f0cdfbf1f99164a90ab54df0833cb4b14cc56bd723ba87"
)

ARMS = ("A", "B", "C", "D")
GATE_PAIRS = ("C-B", "C-D")
GATE_METRICS = ("em", "f1")
DIVERGENT_EXPECTED_COUNT = 22
RECOVERY_RULE = "C em > 0 or C f1 >= 0.1"
REQUIRED_CACHE_EVIDENCE = {
    "persisted_before_evaluation": True,
    "reopened_store": True,
    "hit_only": True,
    "keys_unchanged": True,
    "b_c_kv_equal_after_detached_cpu_normalization": True,
}
VALIDATION_CHECKS = (
    "schema",
    "split",
    "data_role",
    "k_is_selected_k5",
    "fixed_marker_conditioning",
    "record_count",
    "row_structure",
    "rows_match_provenance_ids_in_order",
    "provenance_ids_match_manifest_in_order",
    "manifest_internal_hashes",
    "frozen_confirmation_sha",
    "frozen_original_confirmation100_sha",
    "provenance_id_hashes",
    "split_manifest_sha",
    "frozen_selected_checkpoint",
    "cache_hit_only_reopened_evidence",
    "embedded_gates_present",
    "bootstrap_helper_constants_frozen",
)


class AnalysisError(RuntimeError):
    """A fail-closed validation or consistency failure; no outputs are written."""


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_id_list(record_ids: list[str], *, sorted_ids: bool) -> str:
    values = sorted(record_ids) if sorted_ids else list(record_ids)
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return ROOT / f"results/v70_confirmation_analysis_{stamp}"


def _assert_frozen_helper_constants() -> None:
    frozen = (47, 10_000, -0.02)
    actual = (BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, NONINFERIORITY_MARGIN)
    if actual != frozen:
        raise AnalysisError(
            f"scripts/v70_harness.py bootstrap constants {actual!r} drifted from the "
            f"frozen preregistration {frozen!r}"
        )


def _load_json(path: Path | str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise AnalysisError(f"{label} not found: {path}") from error
    except json.JSONDecodeError as error:
        raise AnalysisError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} is not a JSON object: {path}")
    return value


def _load_divergent_ids(path: Path | str) -> list[str]:
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError as error:
        raise AnalysisError(f"divergent-records file not found: {path}") from error
    ids = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnalysisError(f"divergent line {number} is not valid JSON") from error
        record_id = record.get("record_id")
        if not isinstance(record_id, str):
            raise AnalysisError(f"divergent line {number} has no string record_id")
        ids.append(record_id)
    return ids


def _validate_row(index: int, row: Any) -> None:
    if not isinstance(row, dict):
        raise AnalysisError(f"row {index} is not an object")
    if not isinstance(row.get("record_id"), str):
        raise AnalysisError(f"row {index} has no string record_id")
    if not isinstance(row.get("gold"), list):
        raise AnalysisError(f"row {index} has no gold list")
    arms = row.get("arms")
    if not isinstance(arms, dict):
        raise AnalysisError(f"row {index} has no arms object")
    for arm in ARMS:
        entry = arms.get(arm)
        if not isinstance(entry, dict):
            raise AnalysisError(f"row {index} arm {arm} is missing")
        for metric in ("em", "f1", "presence"):
            value = entry.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AnalysisError(f"row {index} arm {arm} metric {metric} is not numeric")
        if not isinstance(entry.get("prediction"), str):
            raise AnalysisError(f"row {index} arm {arm} has no prediction string")
        if not isinstance(entry.get("generated_ids"), list):
            raise AnalysisError(f"row {index} arm {arm} has no generated_ids list")


def validate_and_extract(
    data: dict[str, Any],
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    expected_confirmation_sha: str,
    expected_original100_sha: str,
    expected_checkpoint_sha: str,
) -> dict[str, Any]:
    if data.get("schema") != EXPECTED_SCHEMA:
        raise AnalysisError(f"input schema {data.get('schema')!r} != {EXPECTED_SCHEMA!r}")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        raise AnalysisError("input has no provenance object")
    split = provenance.get("split", data.get("split"))
    if split != EXPECTED_SPLIT:
        raise AnalysisError(f"input split {split!r} != {EXPECTED_SPLIT!r}")
    if data.get("data_role") != EXPECTED_DATA_ROLE:
        raise AnalysisError(f"input data_role {data.get('data_role')!r} != {EXPECTED_DATA_ROLE!r}")
    if data.get("k") != EXPECTED_K:
        raise AnalysisError(f"input k {data.get('k')!r} != selected arm k {EXPECTED_K}")
    if data.get("fixed_marker_conditioning") is not True:
        raise AnalysisError("input fixed_marker_conditioning is not true")
    if data.get("record_count") != EXPECTED_RECORD_COUNT:
        raise AnalysisError(f"input record_count {data.get('record_count')!r} != 300")

    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_RECORD_COUNT:
        raise AnalysisError("input rows is not a 300-element list")
    for index, row in enumerate(rows):
        _validate_row(index, row)

    confirmation = manifest.get("confirmation")
    if not isinstance(confirmation, dict):
        raise AnalysisError("manifest has no confirmation object")
    manifest_ids = confirmation.get("record_ids")
    if (
        not isinstance(manifest_ids, list)
        or len(manifest_ids) != EXPECTED_RECORD_COUNT
        or len(set(manifest_ids)) != EXPECTED_RECORD_COUNT
        or not all(isinstance(value, str) for value in manifest_ids)
    ):
        raise AnalysisError("frozen manifest must contain 300 unique string record IDs")
    if confirmation.get("count") not in (None, EXPECTED_RECORD_COUNT):
        raise AnalysisError("manifest confirmation.count != 300")
    if confirmation.get("original_confirmation100_count") not in (
        None,
        EXPECTED_ORIGINAL_CONFIRMATION100_COUNT,
    ):
        raise AnalysisError("manifest original_confirmation100_count != 100")

    provenance_ids = provenance.get("record_ids")
    if not isinstance(provenance_ids, list) or provenance_ids != manifest_ids:
        first = None
        if isinstance(provenance_ids, list):
            for index, (left, right) in enumerate(
                zip(provenance_ids, manifest_ids, strict=False)
            ):
                if left != right:
                    first = index
                    break
        raise AnalysisError(
            "provenance.record_ids do not match experiments/v70_confirmation300.json "
            f"exactly in order (first divergence at index {first})"
        )
    row_ids = [row["record_id"] for row in rows]
    if row_ids != manifest_ids:
        raise AnalysisError(
            "rows are not in the exact frozen manifest order "
            "(record_id sequence != confirmation300 manifest order)"
        )

    full_hash = sha256_id_list(manifest_ids, sorted_ids=True)
    if full_hash != confirmation.get("sha256"):
        raise AnalysisError("manifest IDs do not hash to manifest confirmation.sha256")
    if full_hash != expected_confirmation_sha:
        raise AnalysisError("manifest confirmation ID hash != frozen preregistration SHA")
    subset_ids = manifest_ids[:EXPECTED_ORIGINAL_CONFIRMATION100_COUNT]
    if len(set(subset_ids)) != EXPECTED_ORIGINAL_CONFIRMATION100_COUNT:
        raise AnalysisError("manifest first 100 IDs are not unique")
    subset_hash = sha256_id_list(subset_ids, sorted_ids=True)
    if subset_hash != confirmation.get("original_confirmation100_sha256"):
        raise AnalysisError("manifest first-100 IDs do not hash to original_confirmation100_sha256")
    if subset_hash != expected_original100_sha:
        raise AnalysisError(
            "manifest original-confirmation100 hash != frozen preregistration SHA"
        )

    if provenance.get("id_list_sha256") != sha256_id_list(manifest_ids, sorted_ids=False):
        raise AnalysisError("provenance.id_list_sha256 does not match provenance.record_ids")
    if provenance.get("id_set_sha256") != sha256_id_list(manifest_ids, sorted_ids=True):
        raise AnalysisError("provenance.id_set_sha256 does not match provenance.record_ids")
    if provenance.get("split_manifest_sha256") != sha256_file(manifest_path):
        raise AnalysisError(
            f"provenance.split_manifest_sha256 does not match {manifest_path}"
        )

    if provenance.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise AnalysisError(
            f"input checkpoint aggregate {provenance.get('checkpoint_sha256')!r} is not the "
            f"selected k5 checkpoint {expected_checkpoint_sha!r}"
        )

    cache = data.get("cache_validation")
    if not isinstance(cache, dict):
        raise AnalysisError("input has no cache_validation object")
    for key, expected in REQUIRED_CACHE_EVIDENCE.items():
        if cache.get(key) is not expected:
            raise AnalysisError(f"cache_validation.{key} is not {expected!r}")
    if cache.get("encode_on_c_miss") != "fail_closed":
        raise AnalysisError("cache_validation.encode_on_c_miss != 'fail_closed'")
    before = cache.get("entry_count_before")
    after = cache.get("entry_count_after")
    if (
        isinstance(before, bool)
        or isinstance(after, bool)
        or not isinstance(before, int)
        or not isinstance(after, int)
        or before < 1
        or before != after
    ):
        raise AnalysisError(
            "cache_validation entry_count_before/after must be equal positive integers"
        )

    gates = data.get("gates")
    if not isinstance(gates, dict) or any(
        not isinstance(gates.get(pair), dict) for pair in GATE_PAIRS
    ):
        raise AnalysisError("input has no embedded C-B/C-D gates object")

    return {
        "rows": rows,
        "manifest_ids": list(manifest_ids),
        "subset_ids": subset_ids,
        "subset_sha256": subset_hash,
        "cache_validation": cache,
        "embedded_gates": gates,
    }


def arm_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        means[arm] = {
            metric: sum(row["arms"][arm][metric] for row in rows) / len(rows)
            for metric in ("em", "f1", "presence")
        }
    return means


def recompute_gates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "C-B": paired_quality_gate(rows, candidate="C", reference="B"),
        "C-D": paired_quality_gate(rows, candidate="C", reference="D"),
    }


def gates_match(recomputed: dict[str, Any], embedded: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in ("candidate", "reference", "margin", "pass"):
        if recomputed.get(key) != embedded.get(key):
            issues.append(
                f"{key}: recomputed {recomputed.get(key)!r} != embedded {embedded.get(key)!r}"
            )
    embedded_metrics = embedded.get("metrics")
    if not isinstance(embedded_metrics, dict):
        return issues + ["metrics: embedded gate has no metrics object"]
    for metric in GATE_METRICS:
        mine = recomputed["metrics"][metric]
        theirs = embedded_metrics.get(metric)
        if not isinstance(theirs, dict):
            issues.append(f"metrics.{metric}: embedded gate has no {metric} object")
            continue
        for field in ("mean", "ci95", "n", "seed", "resamples", "method"):
            if mine[field] != theirs.get(field):
                issues.append(
                    f"metrics.{metric}.{field}: recomputed {mine[field]!r} != "
                    f"embedded {theirs.get(field)!r}"
                )
    return issues


def divergent_report(
    rows_by_id: dict[str, dict[str, Any]],
    manifest_ids: list[str],
    divergent_ids: list[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for record_id in divergent_ids:
        row = rows_by_id.get(record_id)
        if row is None:
            missing.append(record_id)
            continue
        cached = row["arms"]["C"]
        full_reference = row["arms"]["D"]
        records.append({
            "record_id": record_id,
            "confirmation_position_1based": manifest_ids.index(record_id) + 1,
            "gold": row["gold"],
            "C": {"em": cached["em"], "f1": cached["f1"], "prediction": cached["prediction"]},
            "D": {
                "em": full_reference["em"],
                "f1": full_reference["f1"],
                "prediction": full_reference["prediction"],
            },
            "delta_em_C_minus_D": cached["em"] - full_reference["em"],
            "delta_f1_C_minus_D": cached["f1"] - full_reference["f1"],
            "recovered": bool(cached["em"] > 0.0 or cached["f1"] >= 0.1),
        })
    return {
        "not_selection": True,
        "rule": RECOVERY_RULE,
        "prereg_statement": (
            "V68 confirmation labels or predictions cannot be used for selection; "
            "the divergent population is reported post hoc only."
        ),
        "divergent_file_record_count": len(divergent_ids),
        "expected_count": DIVERGENT_EXPECTED_COUNT,
        "present_count": len(records),
        "missing_ids": missing,
        "recovered_count": sum(1 for record in records if record["recovered"]),
        "records": records,
    }


def build_conclusion(
    gates: dict[str, dict[str, Any]], overall_pass: bool, cache_evidence_ok: bool
) -> str:
    margin = NONINFERIORITY_MARGIN
    bounds = []
    for pair in GATE_PAIRS:
        parts = []
        verdicts = []
        for metric in GATE_METRICS:
            analysis = gates[pair]["metrics"][metric]
            lower = analysis["ci95"][0]
            ok = lower >= margin
            verdicts.append(ok)
            parts.append(f"{metric.upper()} lower bound {lower:.6f} {'>=' if ok else '<'} {margin}")
        gate_pass = all(verdicts)
        state = "passes" if gate_pass else "FAILS"
        bounds.append(f"gate {pair} {state} ({'; '.join(parts)})")
    failed = [
        f"{pair} {metric.upper()} lower bound {gates[pair]['metrics'][metric]['ci95'][0]:.6f}"
        for pair in GATE_PAIRS
        for metric in GATE_METRICS
        if gates[pair]["metrics"][metric]["ci95"][0] < margin
    ]
    if overall_pass:
        outcome = (
            "Both preregistered primary confirmation gates (C-B and C-D) passed: all four "
            "metric lower bounds are at or above the -0.02 margin."
        )
    else:
        outcome = (
            "The preregistered primary confirmation gates did not all pass. Failing bounds: "
            + "; ".join(failed)
            + "."
        )
    persistence = (
        "The persistence evidence required by the preregistration is present: C was read "
        "hit-only from a reopened persisted store, encoding failed closed on any miss, keys "
        "and entry counts were unchanged, and B/C K/V matched after detached CPU "
        "normalization."
        if cache_evidence_ok
        else "The persistence evidence required by the preregistration is NOT present."
    )
    return (
        "Under the frozen preregistration (experiments/v70_preregistration.md, amendments 1-3) "
        "the primary confirmation gates are paired C-B and C-D non-inferiority on EM and token "
        f"F1 over the {EXPECTED_RECORD_COUNT}-record confirmation300 split, with margin {margin}, "
        f"{BOOTSTRAP_RESAMPLES} paired resamples and seed {BOOTSTRAP_SEED}; arms A and B are "
        "diagnostic only. "
        + "; ".join(bounds)
        + ". "
        + outcome
        + " The n=100 original-confirmation-subset column (confirmation positions 1-100) is "
        "reported for V67 comparability and is not a preregistered gate. The V68 divergent-22 "
        "report is post hoc only: per the frozen preregistration, V68 confirmation labels or "
        "predictions cannot be used for selection, and the divergent population is reported, "
        "not selected on. "
        + persistence
        + " This verdict states only which preregistered gates passed or failed and adds no "
        "interpretation beyond the preregistration."
    )


def build_verdict(
    *,
    data: dict[str, Any],
    input_path: Path,
    input_sha: str,
    manifest_path: Path,
    manifest_sha: str,
    divergent_path: Path,
    divergent_sha: str,
    facts: dict[str, Any],
    gates: dict[str, dict[str, Any]],
    subset: dict[str, Any],
    divergent: dict[str, Any],
) -> dict[str, Any]:
    provenance = data["provenance"]
    overall_pass = all(gates[pair]["pass"] for pair in GATE_PAIRS)
    return {
        "schema": ANALYSIS_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "path": str(input_path),
            "sha256": input_sha,
            "schema": data.get("schema"),
            "split": data.get("split"),
            "data_role": data.get("data_role"),
            "k": data.get("k"),
            "max_new_tokens": data.get("max_new_tokens"),
            "record_count": data.get("record_count"),
            "timestamp_utc": data.get("timestamp_utc"),
            "elapsed_seconds": data.get("elapsed_seconds"),
        },
        "validation": {check: True for check in VALIDATION_CHECKS},
        "gate_margin": NONINFERIORITY_MARGIN,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "method": "paired-percentile-mean-delta",
            "helper": "scripts/v70_harness.paired_quality_gate",
        },
        "arm_means": arm_means(facts["rows"]),
        "gates": gates,
        "embedded_gate_match": {pair: True for pair in GATE_PAIRS},
        "overall_gate_pass": overall_pass,
        "original_confirmation100_subset": subset,
        "v68_divergent22_post_hoc": divergent,
        "cache_validation": facts["cache_validation"],
        "conclusion": build_conclusion(gates, overall_pass, True),
        "provenance": {
            "input_path": str(input_path),
            "input_sha256": input_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "divergent_path": str(divergent_path),
            "divergent_sha256": divergent_sha,
            "analyzer_script": "scripts/v70_analyze_confirmation.py",
            "analyzer_script_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint": {
                key: provenance.get(key)
                for key in (
                    "checkpoint",
                    "checkpoint_sha256",
                    "config_sha256",
                    "tokenizer_sha256",
                    "checkpoint_file_sha256",
                )
            },
            "ids": {
                "record_count": EXPECTED_RECORD_COUNT,
                "id_list_sha256": provenance.get("id_list_sha256"),
                "id_set_sha256": provenance.get("id_set_sha256"),
                "split_manifest": provenance.get("split_manifest"),
                "split_manifest_sha256": provenance.get("split_manifest_sha256"),
            },
            "dataset_source": provenance.get("dataset_source"),
            "source_sha256": provenance.get("source_sha256", {}),
        },
    }


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _num(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(verdict: dict[str, Any]) -> str:
    lines: list[str] = []
    provenance = verdict["provenance"]
    input_info = verdict["input"]
    lines.append("# V70 Confirmation300 Verdict")
    lines.append("")
    lines.append(f"Generated: {verdict['generated_utc']}")
    lines.append(
        f"Input: `{input_info['path']}` (sha256 `{input_info['sha256']}`), "
        f"split `{input_info['split']}`, data_role `{input_info['data_role']}`, "
        f"k={input_info['k']}, max_new_tokens={input_info['max_new_tokens']}, "
        f"records={input_info['record_count']}"
    )
    checkpoint = provenance["checkpoint"]
    lines.append(
        f"Selected checkpoint: `{checkpoint['checkpoint']}` "
        f"(aggregate sha256 `{checkpoint['checkpoint_sha256']}`)"
    )
    lines.append("")
    lines.append("## Fail-Closed Validation")
    lines.append("")
    lines.append(
        "All preregistration binding checks passed before this verdict was written: "
        + ", ".join(sorted(verdict["validation"]))
        + "."
    )
    lines.append("")
    lines.append("## Arm Means (n=300)")
    lines.append("")
    lines.append("| Arm | EM | F1 | Presence |")
    lines.append("|---|---:|---:|---:|")
    for arm in ARMS:
        means = verdict["arm_means"][arm]
        lines.append(
            f"| {arm} | {_num(means['em'])} | {_num(means['f1'])} | {_num(means['presence'])} |"
        )
    lines.append("")
    lines.append(
        f"## Preregistered Primary Gates (n=300, paired bootstrap "
        f"{verdict['bootstrap']['resamples']} resamples, seed {verdict['bootstrap']['seed']}, "
        f"margin {verdict['gate_margin']})"
    )
    lines.append("")
    lines.append("| Gate | Metric | Mean delta | 95% CI | Lower bound >= margin | Gate pass |")
    lines.append("|---|---|---:|---|---|---|")
    for pair in GATE_PAIRS:
        gate = verdict["gates"][pair]
        for metric in GATE_METRICS:
            analysis = gate["metrics"][metric]
            lower = analysis["ci95"][0]
            lines.append(
                f"| {pair} | {metric} | {_num(analysis['mean'])} "
                f"| [{_num(analysis['ci95'][0])}, {_num(analysis['ci95'][1])}] "
                f"| {'yes' if lower >= verdict['gate_margin'] else 'NO'} "
                f"| {'yes' if gate['pass'] else 'NO'} |"
            )
    lines.append("")
    lines.append(
        "Embedded gates in the input match the recomputation exactly: "
        + ", ".join(f"{pair} yes" for pair in GATE_PAIRS)
        + "."
    )
    lines.append("")

    subset = verdict["original_confirmation100_subset"]
    lines.append("## V67-Comparability Subset: Original Confirmation100 (n=100)")
    lines.append("")
    lines.append(
        "Descriptive column only; not a preregistered gate. Subset is confirmation "
        f"positions 1-100 of the frozen manifest (sorted-ID sha256 `{subset['sha256']}`)."
    )
    lines.append("")
    lines.append("| Arm | EM | F1 | Presence |")
    lines.append("|---|---:|---:|---:|")
    for arm in ARMS:
        means = subset["arm_means"][arm]
        lines.append(
            f"| {arm} | {_num(means['em'])} | {_num(means['f1'])} | {_num(means['presence'])} |"
        )
    lines.append("")
    lines.append("| Gate | Metric | Mean delta | 95% CI | Lower bound >= margin | Gate pass |")
    lines.append("|---|---|---:|---|---|---|")
    for pair in GATE_PAIRS:
        gate = subset["gates"][pair]
        for metric in GATE_METRICS:
            analysis = gate["metrics"][metric]
            lower = analysis["ci95"][0]
            lines.append(
                f"| {pair} | {metric} | {_num(analysis['mean'])} "
                f"| [{_num(analysis['ci95'][0])}, {_num(analysis['ci95'][1])}] "
                f"| {'yes' if lower >= verdict['gate_margin'] else 'NO'} "
                f"| {'yes' if gate['pass'] else 'NO'} |"
            )
    lines.append("")

    divergent = verdict["v68_divergent22_post_hoc"]
    lines.append("## V68 Divergent-22 Post-Hoc Report (NOT selection)")
    lines.append("")
    lines.append(f"Post hoc only: {divergent['prereg_statement']}")
    lines.append("")
    lines.append(
        f"Divergent IDs in file: {divergent['divergent_file_record_count']} "
        f"(expected {divergent['expected_count']}); present in confirmation300: "
        f"{divergent['present_count']}; missing: "
        f"{', '.join(divergent['missing_ids']) if divergent['missing_ids'] else 'none'}; "
        f"recovered by rule ({divergent['rule']}): {divergent['recovered_count']}."
    )
    lines.append("")
    lines.append(
        "| Record | Pos | C EM | C F1 | D EM | D F1 | dEM(C-D) | dF1(C-D) | Recovered "
        "| C prediction | D prediction |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for record in divergent["records"]:
        lines.append(
            f"| {record['record_id']} | {record['confirmation_position_1based']} "
            f"| {_num(record['C']['em'])} | {_num(record['C']['f1'])} "
            f"| {_num(record['D']['em'])} | {_num(record['D']['f1'])} "
            f"| {_num(record['delta_em_C_minus_D'])} | {_num(record['delta_f1_C_minus_D'])} "
            f"| {'yes' if record['recovered'] else 'no'} "
            f"| {_escape(record['C']['prediction'])} | {_escape(record['D']['prediction'])} |"
        )
    lines.append("")

    lines.append("## Cache (Persistence) Evidence")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for key, value in sorted(verdict["cache_validation"].items()):
        lines.append(f"| {key} | {_escape(value)} |")
    lines.append("")

    lines.append("## Provenance And Hashes")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    rows = [
        (
            "Analyzer script",
            f"`{provenance['analyzer_script']}` sha256 `{provenance['analyzer_script_sha256']}`",
        ),
        ("Input artifact", f"sha256 `{provenance['input_sha256']}`"),
        (
            "Frozen manifest",
            f"`{provenance['manifest_path']}` sha256 `{provenance['manifest_sha256']}`",
        ),
        (
            "Divergent file",
            f"`{provenance['divergent_path']}` sha256 `{provenance['divergent_sha256']}`",
        ),
        ("Checkpoint config sha256", f"`{checkpoint.get('config_sha256')}`"),
        ("Checkpoint tokenizer sha256", f"`{checkpoint.get('tokenizer_sha256')}`"),
        ("ID list sha256", f"`{provenance['ids']['id_list_sha256']}`"),
        ("ID set sha256", f"`{provenance['ids']['id_set_sha256']}`"),
        (
            "Split manifest",
            f"`{provenance['ids']['split_manifest']}` "
            f"sha256 `{provenance['ids']['split_manifest_sha256']}`",
        ),
        ("Dataset source", _escape(provenance["dataset_source"])),
    ]
    for name, value in rows:
        lines.append(f"| {name} | {value} |")
    for source, sha in sorted((provenance.get("source_sha256") or {}).items()):
        lines.append(f"| Source `{source}` | sha256 `{sha}` |")
    lines.append("")
    lines.append("## Preregistration-Faithful Conclusion")
    lines.append("")
    lines.append(verdict["conclusion"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(verdict: dict[str, Any], markdown: str, output_dir: Path) -> dict[str, str]:
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError as error:
        raise AnalysisError(
            f"refusing to overwrite existing analysis directory: {output_dir}"
        ) from error
    markdown_path = output_dir / "verdict.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    verdict["provenance"]["outputs"] = {"verdict_md_sha256": sha256_file(markdown_path)}
    json_path = output_dir / "verdict.json"
    json_path.write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksums = (
        f"{sha256_file(json_path)}  verdict.json\n{sha256_file(markdown_path)}  verdict.md\n"
    )
    (output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return {
        "verdict_json_sha256": sha256_file(json_path),
        "verdict_md_sha256": sha256_file(markdown_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT), help="confirmation evaluation JSON"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="analysis output directory (auto-UTC by default)",
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST), help="frozen confirmation300 manifest"
    )
    parser.add_argument(
        "--divergent", default=str(DEFAULT_DIVERGENT), help="V68 differing_records.jsonl"
    )
    parser.add_argument("--expected-confirmation-sha", default=EXPECTED_CONFIRMATION_SHA256)
    parser.add_argument(
        "--expected-original100-sha", default=EXPECTED_ORIGINAL_CONFIRMATION100_SHA256
    )
    parser.add_argument(
        "--expected-checkpoint-sha", default=EXPECTED_SELECTED_CHECKPOINT_SHA256
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, str]]:
    _assert_frozen_helper_constants()
    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    divergent_path = Path(args.divergent)
    data = _load_json(input_path, "confirmation evaluation artifact")
    manifest = _load_json(manifest_path, "confirmation manifest")
    divergent_ids = _load_divergent_ids(divergent_path)

    facts = validate_and_extract(
        data,
        manifest,
        manifest_path=manifest_path,
        expected_confirmation_sha=args.expected_confirmation_sha,
        expected_original100_sha=args.expected_original100_sha,
        expected_checkpoint_sha=args.expected_checkpoint_sha,
    )
    rows = facts["rows"]
    rows_by_id = {row["record_id"]: row for row in rows}

    gates = recompute_gates(rows)
    issues = {
        pair: problems
        for pair in GATE_PAIRS
        if (problems := gates_match(gates[pair], facts["embedded_gates"][pair]))
    }
    if issues:
        raise AnalysisError(
            "recomputed gates do not match the embedded gates: " + json.dumps(issues)
        )

    subset_rows = [rows_by_id[record_id] for record_id in facts["subset_ids"]]
    subset = {
        "count": len(subset_rows),
        "sha256": facts["subset_sha256"],
        "arm_means": arm_means(subset_rows),
        "gates": recompute_gates(subset_rows),
        "note": "V67 comparability column; not a preregistered gate",
    }
    divergent = divergent_report(rows_by_id, facts["manifest_ids"], divergent_ids)

    verdict = build_verdict(
        data=data,
        input_path=input_path,
        input_sha=sha256_file(input_path),
        manifest_path=manifest_path,
        manifest_sha=sha256_file(manifest_path),
        divergent_path=divergent_path,
        divergent_sha=sha256_file(divergent_path),
        facts=facts,
        gates=gates,
        subset=subset,
        divergent=divergent,
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    hashes = write_outputs(verdict, render_markdown(verdict), output_dir)
    return output_dir, verdict, hashes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_dir, verdict, hashes = run(args)
    except AnalysisError as error:
        print(f"V70 confirmation analysis FAILED (fail closed): {error}", file=sys.stderr)
        return 2
    print(f"analysis directory: {output_dir}")
    print(f"verdict.json sha256: {hashes['verdict_json_sha256']}")
    print(f"verdict.md sha256: {hashes['verdict_md_sha256']}")
    for pair in GATE_PAIRS:
        print(f"gate {pair} pass: {verdict['gates'][pair]['pass']}")
    print(f"overall preregistered gate pass: {verdict['overall_gate_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
