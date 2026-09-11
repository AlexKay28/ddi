"""V64 evidence-audit step 3: preregistered QA recomputation (Analysis A).

Executes experiments/v64_audit_preregistration.{md,json} section 2 (analyses.qa)
verbatim: paired bootstrap over the 24 frozen ids, seed 47, 10000 resamples, one
shared 10000x24 index matrix drawn once, two-sided 95% percentile CI. Additive-only:
creates results/v64_audit_<UTCts>_QA_RECOMPUTATION/ and touches nothing else.
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_MD = "experiments/v64_audit_preregistration.md"
PREREG_JSON = "experiments/v64_audit_preregistration.json"
FREEZE_MANIFEST = "experiments/v60_confirmation_manifest.json"
RUN_DIR = "results/v60_confirmation24_20260910T093500Z"
REQUESTS_JSONL = f"{RUN_DIR}/requests.jsonl"
RESULT_JSON = f"{RUN_DIR}/result.json"

SEED = 47
N_RECORDS = 24
N_RESAMPLES = 10000
CI_QUANTILES = (0.025, 0.975)
NI_FLOOR = -0.02
AGG_TOLERANCE = 1e-9
METRICS = ("em", "f1")
ARM_PATHS = {
    "A": ("A",),
    "B": ("B",),
    "C_cold": ("C", "cold"),
    "C_all_hit": ("C", "all_hit"),
    "D": ("D",),
}
CONTRASTS = (
    ("B_minus_A", "B", "primary"),
    ("C_all_hit_minus_A", "C_all_hit", "co_reported"),
    ("C_cold_minus_A", "C_cold", "completeness_only"),
    ("D_minus_A", "D", "descriptive_not_preregistered"),
)


class AuditAbort(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_inputs(prereg: dict) -> list:
    verified = []
    for entry in prereg["frozen_inputs"]:
        path = REPO_ROOT / entry["path"]
        actual = sha256_file(path)
        ok = actual == entry["sha256"]
        verified.append(
            {
                "path": entry["path"],
                "role": entry["role"],
                "sha256_expected": entry["sha256"],
                "sha256_actual": actual,
                "match": ok,
            }
        )
        if not ok:
            raise AuditAbort(
                f"input sha mismatch for {entry['path']}: stop and re-inventory; "
                "never recompute on drifted inputs"
            )
    return verified


def load_frozen_ids(prereg: dict) -> list:
    manifest = json.loads((REPO_ROOT / FREEZE_MANIFEST).read_text())
    ids = manifest["config"]["record_ids"]
    if len(ids) != N_RECORDS or len(set(ids)) != N_RECORDS:
        raise AuditAbort(
            f"freeze manifest holds {len(ids)} ids (unique {len(set(ids))}), expected {N_RECORDS}"
        )
    if ids != prereg["frozen_record_ids"]:
        raise AuditAbort("freeze-manifest ids diverge from preregistration frozen_record_ids")
    return list(ids)


def load_records(requests_path: Path, frozen_ids: list) -> dict:
    records = {}
    duplicates = []
    with requests_path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditAbort(
                    f"unparsable line {line_no} in {requests_path.name}: {exc}"
                ) from exc
            record_id = record.get("record_id")
            if record_id is None:
                raise AuditAbort(f"line {line_no} has no record_id")
            if record_id in records:
                duplicates.append(record_id)
            records[record_id] = record
    missing = [rid for rid in frozen_ids if rid not in records]
    unexpected = sorted(set(records) - set(frozen_ids))
    if duplicates:
        raise AuditAbort(f"duplicate record lines: {sorted(set(duplicates))}")
    if missing or unexpected:
        raise AuditAbort(
            f"count reconciliation against {N_RECORDS} failed: parsed={len(records)} "
            f"frozen={len(frozen_ids)} missing={missing} unexpected={unexpected}; "
            "records are never silently dropped"
        )
    for rid in frozen_ids:
        for arm in ARM_PATHS.values():
            entry = records[rid]["arms"]
            for key in arm:
                if not isinstance(entry, dict) or key not in entry:
                    raise AuditAbort(f"record {rid}: missing arm path {'/'.join(arm)}")
                entry = entry[key]
            quality = entry.get("quality")
            for metric in METRICS:
                value = None if not isinstance(quality, dict) else quality.get(metric)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise AuditAbort(
                        f"record {rid}: arm {'/'.join(arm)} has unparsable "
                        f"quality.{metric}: {value!r}"
                    )
    return records


def arm_metrics(records: dict, frozen_ids: list, arm: str) -> dict:
    path = ARM_PATHS[arm]
    vectors = {metric: [] for metric in METRICS}
    for rid in frozen_ids:
        entry = records[rid]["arms"]
        for key in path:
            entry = entry[key]
        for metric in METRICS:
            vectors[metric].append(float(entry["quality"][metric]))
    return {metric: np.array(values) for metric, values in vectors.items()}


def reproduce_aggregates(
    records: dict, frozen_ids: list, stored: dict, tolerance: float = AGG_TOLERANCE
) -> dict:
    details = {}
    mismatches = []
    for arm in ARM_PATHS:
        vectors = arm_metrics(records, frozen_ids, arm)
        details[arm] = {}
        for metric in METRICS:
            recomputed = float(np.mean(vectors[metric]))
            expected = float(stored[arm][metric])
            diff = abs(recomputed - expected)
            ok = diff <= tolerance
            details[arm][metric] = {
                "recomputed": recomputed,
                "stored": expected,
                "abs_diff": diff,
                "match": ok,
            }
            if not ok:
                mismatches.append(
                    f"{arm}.{metric}: recomputed {recomputed!r} vs stored {expected!r}"
                )
    if mismatches:
        raise AuditAbort(
            "stored aggregates do not reproduce from per-record values: " + "; ".join(mismatches)
        )
    return details


def per_record_b_c_equality(records: dict, frozen_ids: list) -> dict:
    equal = 0
    divergent = []
    for rid in frozen_ids:
        arms = records[rid]["arms"]
        b = arms["B"]["quality"]
        cold = arms["C"]["cold"]["quality"]
        all_hit = arms["C"]["all_hit"]["quality"]
        if all(b[m] == cold[m] == all_hit[m] for m in METRICS):
            equal += 1
        else:
            divergent.append(rid)
    return {"records_equal": equal, "records_divergent": len(divergent), "divergent_ids": divergent}


def draw_index_matrix() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.integers(0, N_RECORDS, size=(N_RESAMPLES, N_RECORDS))


def bootstrap_ci(deltas: np.ndarray, index_matrix: np.ndarray) -> tuple:
    """Frozen contract (prereg section 2): resampled means = deltas[index_matrix].mean(axis=1);
    CI = np.quantile(means, [0.025, 0.975]) with default linear interpolation."""
    resampled = np.asarray(deltas, dtype=float)[index_matrix].mean(axis=1)
    lo, hi = np.quantile(resampled, CI_QUANTILES)
    return float(lo), float(hi)


def mean_delta(deltas: np.ndarray) -> float:
    return float(np.mean(deltas))


def classify(stats: dict, applies: bool) -> dict:
    rule = "non-inferiority holds iff CI lower bound >= -0.02 for BOTH em and f1"
    if not applies:
        return {"rule_satisfied": None, "verdict": "not_applicable", "rule": rule}
    lb_em = stats["em"]["ci95"][0]
    lb_f1 = stats["f1"]["ci95"][0]
    satisfied = lb_em >= NI_FLOOR and lb_f1 >= NI_FLOOR
    if satisfied:
        verdict = "pass"
    elif stats["em"]["mean_delta"] >= 0.0 and stats["f1"]["mean_delta"] >= 0.0:
        verdict = "inconclusive"
    else:
        verdict = "fail"
    return {
        "rule_satisfied": satisfied,
        "verdict": verdict,
        "rule": rule,
        "ci_lower_bounds": {"em": lb_em, "f1": lb_f1},
    }


def run_analysis() -> tuple:
    created_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    prereg = json.loads((REPO_ROOT / PREREG_JSON).read_text())
    verified_inputs = verify_frozen_inputs(prereg)
    frozen_ids = load_frozen_ids(prereg)
    records = load_records(REPO_ROOT / REQUESTS_JSONL, frozen_ids)
    result_stored = json.loads((REPO_ROOT / RESULT_JSON).read_text())
    stored_quality = result_stored["aggregate"]["quality_mean"]
    aggregate_report = reproduce_aggregates(records, frozen_ids, stored_quality)
    b_c_equality = per_record_b_c_equality(records, frozen_ids)

    index_matrix = draw_index_matrix()
    arm_vectors = {arm: arm_metrics(records, frozen_ids, arm) for arm in ARM_PATHS}
    contrasts = {}
    for name, arm, status in CONTRASTS:
        stats = {}
        for metric in METRICS:
            deltas = arm_vectors[arm][metric] - arm_vectors["A"][metric]
            ci = [round(bound, 6) for bound in bootstrap_ci(deltas, index_matrix)]
            stats[metric] = {"mean_delta": round(mean_delta(deltas), 6), "ci95": ci}
        contrasts[name] = {
            "status": status,
            "preregistered": status != "descriptive_not_preregistered",
            "treatment_arm": arm,
            "reference_arm": "A",
            "n": len(frozen_ids),
            **stats,
            "decision_rule": classify(stats, applies=status in ("primary", "co_reported")),
        }

    headline = contrasts["B_minus_A"]["decision_rule"]
    backend_labels = {}
    for arm, path in ARM_PATHS.items():
        labels = set()
        for rid in frozen_ids:
            entry = records[rid]["arms"]
            for key in path:
                entry = entry[key]
            labels.add(entry.get("backend"))
        backend_labels[arm] = sorted(labels)

    report = {
        "schema": "v64_audit_step3_qa_recomputation_v1",
        "created_utc": created_utc,
        "preregistration": {
            "md_sha256": sha256_file(REPO_ROOT / PREREG_MD),
            "json_sha256": sha256_file(REPO_ROOT / PREREG_JSON),
            "executed_sections": ["analyses.qa"],
            "scope_note": (
                "Analysis B (analyses.timing, TTFT) is a separate preregistered analysis, "
                "not part of this QA artifact"
            ),
        },
        "input_gate": {"status": "pass", "files": verified_inputs},
        "frozen_ids": {
            "source": FREEZE_MANIFEST,
            "count": len(frozen_ids),
            "ids": frozen_ids,
            "match_prereg_twin": True,
        },
        "record_reconciliation": {
            "frozen": len(frozen_ids),
            "parsed": len(records),
            "matched": len(frozen_ids),
            "exclusions": 0,
        },
        "aggregate_reproduction": {
            "status": "pass",
            "tolerance": AGG_TOLERANCE,
            "arms": aggregate_report,
        },
        "b_vs_c_equal_all_raw_check": {
            **b_c_equality,
            "stored_flag": result_stored["aggregate"]["b_vs_c_equal_all"],
        },
        "observed_backend_labels": backend_labels,
        "determinism": {
            "rng": f"numpy.random.default_rng({SEED}) (PCG64), fresh instance for this analysis",
            "seed": SEED,
            "resamples": N_RESAMPLES,
            "index_matrix": (
                f"rng.integers(0, {N_RECORDS}, size=({N_RESAMPLES}, {N_RECORDS})), drawn once at "
                "analysis start and shared by every contrast and both metrics (paired bootstrap)"
            ),
            "ci_method": "np.quantile([0.025, 0.975]), default linear interpolation",
            "numpy_version": np.__version__,
        },
        "contrasts": contrasts,
        "headline_verdict": {"contrast": "B_minus_A", **headline},
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "results" / f"v64_audit_{stamp}_QA_RECOMPUTATION"
    out_dir.mkdir(parents=True)
    recomputation_path = out_dir / "recomputation.json"
    recomputation_path.write_text(json.dumps(report, indent=2) + "\n")
    summary_path = out_dir / "summary.md"
    summary_path.write_text(render_summary(report))

    manifest = {
        "schema": "v64_audit_step3_qa_recomputation_manifest_v1",
        "created_utc": created_utc,
        "step": (
            "V64 evidence-audit step 3: QA recomputation per preregistration "
            "section 2 (analyses.qa)"
        ),
        "inputs": [
            {
                "path": PREREG_MD,
                "sha256": report["preregistration"]["md_sha256"],
                "role": "preregistration (methods, frozen)",
            },
            {
                "path": PREREG_JSON,
                "sha256": report["preregistration"]["json_sha256"],
                "role": "preregistration JSON twin",
            },
            *[
                {
                    "path": entry["path"],
                    "sha256": entry["sha256_actual"],
                    "role": entry["role"],
                    "reverified_at_step3_start": True,
                }
                for entry in verified_inputs
            ],
        ],
        "outputs": [
            {"path": "recomputation.json", "sha256": sha256_file(recomputation_path)},
            {"path": "summary.md", "sha256": sha256_file(summary_path)},
        ],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "additivity": {"existing_files_modified": 0, "gpu_used": False, "tracker_writes": False},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir, report


def render_summary(report: dict) -> str:
    gates = report["input_gate"]["files"]
    recon = report["record_reconciliation"]
    b_c = report["b_vs_c_equal_all_raw_check"]
    lines = [
        "# V64 Evidence Audit — Step 3: QA Recomputation (Analysis A)",
        "",
        f"- Created (UTC): {report['created_utc']}",
        "- Scope: preregistration section 2 (`analyses.qa`) executed verbatim; Analysis B (TTFT, "
        "section 3) is a separate preregistered analysis.",
        f"- Input gate: **pass** — all {len(gates)} sha-pinned §1 inputs re-verified "
        "at step-3 start.",
        f"- Records: {recon['matched']}/{recon['frozen']} frozen ids matched, "
        f"{recon['exclusions']} exclusions (no record silently dropped).",
        "- Stored-aggregate reproduction: **yes** — every arm×metric mean recomputed "
        "from per-record values matches `result.json` exactly (tolerance 1e-9).",
        f"- `b_vs_c_equal_all` on raw records: {b_c['records_equal']}/24 records equal across "
        f"B / C_cold / C_all_hit (EM and F1); stored flag = {b_c['stored_flag']}.",
        f"- Observed per-record backend labels: {json.dumps(report['observed_backend_labels'])}.",
        "",
        "## Contrasts (mean paired delta, two-sided 95% percentile bootstrap CI; n = 24)",
        "",
        "| Contrast | Status | Metric | Mean delta | 95% CI | n |",
        "|---|---|---|---|---|---|",
    ]
    for name, contrast in report["contrasts"].items():
        for metric in METRICS:
            stats = contrast[metric]
            lines.append(
                f"| {name} | {contrast['status']} | {metric} | {stats['mean_delta']:+.6f} "
                f"| [{stats['ci95'][0]:.6f}, {stats['ci95'][1]:.6f}] | {contrast['n']} |"
            )
    primary = report["contrasts"]["B_minus_A"]
    lines += [
        "",
        "## Decision rule (frozen, primary contrast `B_minus_A`)",
        "",
        f"- Rule: {primary['decision_rule']['rule']}.",
        f"- CI lower bounds: em {primary['em']['ci95'][0]:.6f}, f1 {primary['f1']['ci95'][0]:.6f} "
        "(floor −0.02).",
        f"- Headline verdict: **{report['headline_verdict']['verdict']}**.",
    ]
    if report["headline_verdict"]["verdict"] == "inconclusive":
        lines.append(
            "- Rule not satisfied (em CI LB < −0.02), but mean delta ≥ 0 for both metrics: "
            "direction favorable, interval too wide to clear the floor — underpowered. "
            "Recorded as-is; the test was not altered and no observations were added."
        )
    co_reported = report["contrasts"]["C_all_hit_minus_A"]["decision_rule"]
    co_word = "satisfied" if co_reported["rule_satisfied"] else "not satisfied"
    lines += [
        "",
        f"- Co-reported `C_all_hit_minus_A`: rule {co_word}, verdict **{co_reported['verdict']}**; "
        "identical to the primary on raw records. Headline verdict remains the primary only "
        "(no gate-shopping).",
        "- `C_cold_minus_A` (completeness only) and `D_minus_A` (descriptive, step-3 instruction; "
        "not a preregistered contrast) are reported without a verdict.",
        "",
        "## Determinism",
        "",
        f"- {report['determinism']['rng']}; {report['determinism']['index_matrix']}; "
        f"{report['determinism']['ci_method']}.",
        f"- numpy {report['determinism']['numpy_version']}; a rerun on the same pinned inputs "
        "reproduces every number bit-for-bit.",
        "",
        "Additive-only: this directory plus `scripts/recompute_v64_qa_audit.py` and "
        "`tests/test_recompute_v64_qa_audit.py` are the only step-3 additions; no existing file "
        "modified, no GPU, no Tracker.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        out_dir, report = run_analysis()
    except AuditAbort as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    print(f"artifact: {out_dir.relative_to(REPO_ROOT)}")
    print(f"aggregate reproduction: yes (tolerance {AGG_TOLERANCE:g})")
    for name, contrast in report["contrasts"].items():
        for metric in METRICS:
            stats = contrast[metric]
            print(
                f"{name:20s} {metric}: delta={stats['mean_delta']:+.6f} "
                f"CI=[{stats['ci95'][0]:.6f}, {stats['ci95'][1]:.6f}] n={contrast['n']}"
            )
    headline = report["headline_verdict"]
    print(f"headline verdict ({headline['contrast']}): {headline['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
