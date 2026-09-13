"""Fail-closed CPU tests for scripts/v70_analyze_confirmation.py.

The synthetic confirmation artifact uses the real frozen confirmation300 manifest IDs
(read-only) with hand-designed per-row metrics, so every gate, subset, and divergent
outcome is known in advance.  Embedded gates are produced by the same frozen
``scripts.v70_harness.paired_quality_gate`` helper the evaluation run uses, which is
exactly the agreement the analyzer must verify.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.v70_analyze_confirmation import (
    EXPECTED_CONFIRMATION_SHA256,
    EXPECTED_ORIGINAL_CONFIRMATION100_SHA256,
    EXPECTED_SELECTED_CHECKPOINT_SHA256,
    default_output_dir,
    gates_match,
    main,
    recompute_gates,
    sha256_file,
    sha256_id_list,
)
from scripts.v70_harness import paired_quality_gate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/v70_confirmation300.json"
DIVERGENT_PATH = ROOT / "results/v68_anatomy_20260910T131036Z/differing_records.jsonl"

MANIFEST_IDS: list[str] = json.loads(MANIFEST_PATH.read_text())["confirmation"]["record_ids"]
DIVERGENT_IDS: list[str] = [
    json.loads(line)["record_id"]
    for line in DIVERGENT_PATH.read_text().splitlines()
    if line.strip()
]


def _arm(em: float, f1: float, prediction: str = "gold answer") -> dict[str, Any]:
    return {
        "generated_ids": [1, 2, 3],
        "prediction": prediction,
        "em": em,
        "f1": f1,
        "presence": 1.0 if em > 0 else 0.0,
    }


def _recovered_rule(em: float, f1: float) -> bool:
    return em > 0.0 or f1 >= 0.1


def build_rows() -> list[dict[str, Any]]:
    """300 rows with known gate outcomes: C == B everywhere, C beats D by design.

    Divergent-22 overrides exercise the post-hoc report: the first three divergent IDs
    (in file order) get C metrics that are recovered-by-EM, recovered-by-F1, and
    not-recovered; every other divergent row scores zero and is not recovered.
    """
    first_divergent = {record_id: index for index, record_id in enumerate(DIVERGENT_IDS[:3])}
    rows = []
    for index, record_id in enumerate(MANIFEST_IDS):
        c_em = 1.0 if index % 5 == 0 else 0.0
        c_f1 = c_em
        d_em = 1.0 if (index % 5 == 0 and index % 10 != 0) else 0.0
        d_f1 = d_em
        if record_id in DIVERGENT_IDS:
            d_em, d_f1 = 0.0, 0.0
            if record_id in first_divergent:
                position = first_divergent[record_id]
                if position == 0:
                    c_em, c_f1 = 1.0, 1.0  # recovered via EM
                elif position == 1:
                    c_em, c_f1 = 0.0, 0.2  # recovered via F1
                else:
                    c_em, c_f1 = 0.0, 0.05  # not recovered
            else:
                c_em, c_f1 = 0.0, 0.0  # not recovered
        rows.append({
            "record_id": record_id,
            "gold": ["gold answer"],
            "arms": {
                "A": _arm(c_em, c_f1),
                "B": _arm(c_em, c_f1),
                "C": _arm(c_em, c_f1),
                "D": _arm(d_em, d_f1),
            },
        })
    return rows


def build_artifact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "v70-evaluation-v2",
        "timestamp_utc": "2026-09-11T00:00:00+00:00",
        "split": "confirmation300",
        "data_role": "confirmation_primary",
        "record_count": len(rows),
        "k": 5,
        "max_new_tokens": 64,
        "elapsed_seconds": 12.5,
        "fixed_marker_conditioning": True,
        "cache_validation": {
            "persisted_before_evaluation": True,
            "reopened_store": True,
            "encode_on_c_miss": "fail_closed",
            "hit_only": True,
            "entry_count_before": 1000,
            "entry_count_after": 1000,
            "keys_unchanged": True,
            "b_c_kv_equal_after_detached_cpu_normalization": True,
        },
        "rows": rows,
        "gates": {
            "C-B": paired_quality_gate(rows, candidate="C", reference="B"),
            "C-D": paired_quality_gate(rows, candidate="C", reference="D"),
        },
        "provenance": {
            "checkpoint": "results/v70_arms_valid_seed47/checkpoint_k5",
            "checkpoint_sha256": EXPECTED_SELECTED_CHECKPOINT_SHA256,
            "checkpoint_file_sha256": {"config.json": "not-a-real-hash"},
            "config_sha256": "not-a-real-hash",
            "tokenizer_sha256": "not-a-real-hash",
            "record_ids": list(MANIFEST_IDS),
            "id_list_sha256": sha256_id_list(MANIFEST_IDS, sorted_ids=False),
            "id_set_sha256": sha256_id_list(MANIFEST_IDS, sorted_ids=True),
            "split_manifest": "experiments/v70_confirmation300.json",
            "split_manifest_sha256": sha256_file(MANIFEST_PATH),
            "dataset_source": "ddi.data.load('hotpot_qa', 'train', 0)",
            "source_sha256": {},
        },
    }


def _means(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return sum(row["arms"][arm][metric] for row in rows) / len(rows)


def write_input(tmp_path: Path, artifact: dict[str, Any]) -> Path:
    path = tmp_path / "v70_confirmation300_k5.json"
    path.write_text(json.dumps(artifact))
    return path


def run_analyzer(
    tmp_path: Path, artifact: dict[str, Any], *extra: str
) -> tuple[int, Path, Path]:
    input_path = write_input(tmp_path, artifact)
    output_dir = tmp_path / "analysis_out"
    code = main([
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        *extra,
    ])
    return code, input_path, output_dir


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return build_rows()


@pytest.fixture(scope="module")
def base_artifact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return build_artifact(rows)


def test_manifest_frozen_hashes_unchanged() -> None:
    confirmation = json.loads(MANIFEST_PATH.read_text())["confirmation"]
    assert sha256_id_list(MANIFEST_IDS, sorted_ids=True) == EXPECTED_CONFIRMATION_SHA256
    assert confirmation["sha256"] == EXPECTED_CONFIRMATION_SHA256
    assert confirmation["original_confirmation100_sha256"] == (
        EXPECTED_ORIGINAL_CONFIRMATION100_SHA256
    )
    assert len(DIVERGENT_IDS) == 22
    assert all(record_id in MANIFEST_IDS for record_id in DIVERGENT_IDS)


def test_sha256_id_list_sorted_vs_unordered() -> None:
    values = ["b", "a", "c"]
    assert sha256_id_list(values, sorted_ids=False) != sha256_id_list(values, sorted_ids=True)
    assert sha256_id_list(values, sorted_ids=True) == hashlib.sha256(b"a\nb\nc").hexdigest()
    assert sha256_id_list(values, sorted_ids=False) == hashlib.sha256(b"b\na\nc").hexdigest()


def test_gate_recomputation_agreement(rows: list[dict[str, Any]]) -> None:
    artifact = build_artifact(rows)
    recomputed = recompute_gates(rows)
    for pair in ("C-B", "C-D"):
        assert gates_match(recomputed[pair], artifact["gates"][pair]) == []
    assert recomputed["C-B"]["pass"] is True
    assert recomputed["C-D"]["pass"] is True
    tampered = deepcopy(artifact["gates"]["C-D"])
    tampered["metrics"]["em"]["ci95"] = [-0.5, -0.1]
    assert gates_match(recomputed["C-D"], tampered) != []


def test_default_output_dir_has_utc_stamp() -> None:
    name = default_output_dir().name
    assert name.startswith("v70_confirmation_analysis_")
    stamp = name.removeprefix("v70_confirmation_analysis_")
    assert len(stamp) == 16 and stamp.endswith("Z") and stamp[8] == "T"


def test_happy_path_emits_verdict_files(
    tmp_path: Path, rows: list[dict[str, Any]], base_artifact: dict[str, Any]
) -> None:
    code, input_path, output_dir = run_analyzer(tmp_path, base_artifact)
    assert code == 0
    assert (output_dir / "verdict.json").is_file()
    assert (output_dir / "verdict.md").is_file()
    assert (output_dir / "SHA256SUMS").is_file()

    verdict = json.loads((output_dir / "verdict.json").read_text())
    assert verdict["overall_gate_pass"] is True
    assert all(verdict["embedded_gate_match"].values())
    assert all(verdict["validation"].values())

    for arm in ("A", "B", "C", "D"):
        for metric in ("em", "f1", "presence"):
            assert verdict["arm_means"][arm][metric] == pytest.approx(
                _means(rows, arm, metric)
            )

    subset = verdict["original_confirmation100_subset"]
    assert subset["count"] == 100
    assert subset["sha256"] == EXPECTED_ORIGINAL_CONFIRMATION100_SHA256
    for arm in ("A", "B", "C", "D"):
        for metric in ("em", "f1", "presence"):
            assert subset["arm_means"][arm][metric] == pytest.approx(
                _means(rows[:100], arm, metric)
            )
    assert subset["gates"]["C-B"]["pass"] is True
    assert subset["gates"]["C-D"]["pass"] is True
    # Subset gates must be recomputed over exactly the first 100 manifest rows.
    for pair in ("C-B", "C-D"):
        candidate, reference = pair.split("-")
        assert gates_match(
            subset["gates"][pair],
            paired_quality_gate(rows[:100], candidate=candidate, reference=reference),
        ) == []

    divergent = verdict["v68_divergent22_post_hoc"]
    assert divergent["not_selection"] is True
    assert divergent["divergent_file_record_count"] == 22
    assert divergent["present_count"] == 22
    assert divergent["missing_ids"] == []
    assert divergent["recovered_count"] == 2
    recovered_ids = {
        record["record_id"] for record in divergent["records"] if record["recovered"]
    }
    assert recovered_ids == set(DIVERGENT_IDS[:2])
    for record in divergent["records"]:
        arms = next(row["arms"] for row in rows if row["record_id"] == record["record_id"])
        assert record["recovered"] == _recovered_rule(arms["C"]["em"], arms["C"]["f1"])
        assert record["delta_em_C_minus_D"] == arms["C"]["em"] - arms["D"]["em"]

    provenance = verdict["provenance"]
    assert provenance["input_sha256"] == sha256_file(input_path)
    assert provenance["analyzer_script_sha256"] == sha256_file(
        ROOT / "scripts/v70_analyze_confirmation.py"
    )
    assert provenance["checkpoint"]["checkpoint_sha256"] == (
        EXPECTED_SELECTED_CHECKPOINT_SHA256
    )

    outputs = provenance["outputs"]
    assert outputs["verdict_md_sha256"] == sha256_file(output_dir / "verdict.md")
    sums = (output_dir / "SHA256SUMS").read_text().splitlines()
    assert sums[0] == f"{sha256_file(output_dir / 'verdict.json')}  verdict.json"
    assert sums[1] == f"{outputs['verdict_md_sha256']}  verdict.md"

    markdown = (output_dir / "verdict.md").read_text()
    assert "Preregistration-Faithful Conclusion" in markdown
    assert "Both preregistered primary confirmation gates (C-B and C-D) passed" in markdown
    assert "post hoc" in markdown


def test_missing_input_fails_closed(tmp_path: Path) -> None:
    output_dir = tmp_path / "analysis_out"
    code = main(["--input", str(tmp_path / "absent.json"), "--output-dir", str(output_dir)])
    assert code == 2
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("mutate", "extra"),
    [
        pytest.param(lambda a: a.update(schema="v70-evaluation-v1"), (), id="schema"),
        pytest.param(lambda a: a.update(record_count=299), (), id="record_count"),
        pytest.param(lambda a: a.update(split="confirmation299"), (), id="split"),
        pytest.param(
            lambda a: a["cache_validation"].update(hit_only=False), (), id="cache_hit_only"
        ),
        pytest.param(
            lambda a: a["provenance"].update(checkpoint_sha256="0" * 64),
            (),
            id="checkpoint_sha",
        ),
        pytest.param(
            lambda a: a["provenance"]["record_ids"].reverse(), (), id="ids_reordered"
        ),
        pytest.param(
            lambda a: a["rows"].reverse(), (), id="rows_reordered"
        ),
        pytest.param(
            lambda a: a["gates"]["C-D"]["metrics"]["em"]["ci95"].__setitem__(0, -0.9),
            (),
            id="embedded_gate_tampered",
        ),
        pytest.param(
            lambda a: None,
            ("--expected-confirmation-sha", "f" * 64),
            id="frozen_confirmation_sha",
        ),
    ],
)
def test_validation_failures_write_nothing(
    tmp_path: Path,
    base_artifact: dict[str, Any],
    mutate: Any,
    extra: tuple[str, ...],
) -> None:
    artifact = deepcopy(base_artifact)
    mutate(artifact)
    code, _, output_dir = run_analyzer(tmp_path, artifact, *extra)
    assert code == 2
    assert not output_dir.exists()


def test_tampered_manifest_copy_fails_closed(tmp_path: Path, base_artifact: dict[str, Any]) -> None:
    manifest_copy = tmp_path / "manifest_copy.json"
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["confirmation"]["record_ids"][7] = "0" * 24
    manifest_copy.write_text(json.dumps(manifest))
    code, _, output_dir = run_analyzer(
        tmp_path, base_artifact, "--manifest", str(manifest_copy)
    )
    assert code == 2
    assert not output_dir.exists()


def test_divergent_id_absent_from_rows_is_reported_not_fatal(
    tmp_path: Path, base_artifact: dict[str, Any]
) -> None:
    divergent_file = tmp_path / "differing_records.jsonl"
    lines = [json.dumps({"record_id": DIVERGENT_IDS[0]}), json.dumps({"record_id": "absent1"})]
    divergent_file.write_text("\n".join(lines) + "\n")
    code, _, output_dir = run_analyzer(
        tmp_path, base_artifact, "--divergent", str(divergent_file)
    )
    assert code == 0
    divergent = json.loads((output_dir / "verdict.json").read_text())[
        "v68_divergent22_post_hoc"
    ]
    assert divergent["divergent_file_record_count"] == 2
    assert divergent["present_count"] == 1
    assert divergent["missing_ids"] == ["absent1"]
    assert divergent["recovered_count"] == 1
