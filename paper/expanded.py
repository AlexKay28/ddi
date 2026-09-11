"""Read-only, hash-checked postprocessing and full-width manuscript figures.

No model imports, experiments, or inferred QA/performance joins. Cache traces
are joined only within the exact-work V53 campaign.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T95 = 4.302652729911275


def source(evidence, name, root=ROOT):
    spec = evidence["sources"][name]
    if Path(spec["path"]).is_absolute() or ".." in Path(spec["path"]).parts:
        raise ValueError("Source path escapes repository")
    data = (root / spec["path"]).read_bytes()
    if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError(f"Source hash mismatch: {spec['path']}")
    if spec["path"].endswith(".jsonl"):
        return [json.loads(line) for line in data.splitlines()]
    if spec["path"].endswith(".log"):
        return data.decode("utf-8")
    return json.loads(data)


def flex_history(evidence, root=ROOT):
    """Independent historical experiments; never infer a timing decomposition."""
    artifact = evidence["artifacts"]["V11_flex_history"]
    if artifact["status"] != "historical_unvalidated" or artifact["sources"] != [
        "flex_kernel_history",
        "flex_model_history",
    ]:
        raise ValueError("Flex historical evidence boundary mismatch")
    result = {}
    shapes = [(3, 512, 1600), (9, 512, 4672), (25, 512, 12864), (25, 2048, 51264)]
    for scope, name in zip(("kernel", "model_prefill"), artifact["sources"], strict=True):
        text = source(evidence, name, root)
        sections = re.split(r"(?m)^\s*(?:=== )?N=(\d+) L=(\d+) T=(\d+)[^\n]*\n", text)
        rows = []
        for index in range(1, len(sections), 4):
            n, length, total = map(int, sections[index : index + 3])
            body = sections[index + 3]
            if scope == "kernel":
                pairs = re.findall(r"(?m)^  flex  full=\s*([\d.]+)ms  ddi_qf=\s*([\d.]+)ms", body)
            else:
                full = re.findall(r"(?m)^  flex full = ([\d.]+)ms$", body)
                ddi = re.findall(r"(?m)^  flex ddi_qf = ([\d.]+)ms$", body)
                pairs = list(zip(full, ddi, strict=True)) if len(full) == len(ddi) else []
            if len(pairs) != 1:
                raise ValueError(f"Flex {scope}: missing or duplicate detailed timing")
            full_ms, ddi_ms = map(float, pairs[0])
            if not all(math.isfinite(v) and v > 0 for v in (full_ms, ddi_ms)):
                raise ValueError(f"Flex {scope}: invalid latency")
            rows.append(
                {
                    "n": n,
                    "length": length,
                    "total": total,
                    "full_ms": full_ms,
                    "ddi_ms": ddi_ms,
                    "ratio": full_ms / ddi_ms,
                }
            )
        if [(r["n"], r["length"], r["total"]) for r in rows] != shapes:
            raise ValueError(f"Flex {scope}: unexpected or incomplete shapes")
        result[scope] = rows
    return result


def amdahl_speedup(fraction, component_speedup):
    """Illustrative baseline component fraction, not a measured attention fraction."""
    return 1 / ((1 - fraction) + fraction / component_speedup)


def pair_cost(n, length, query, hit=0):
    """Attention-pair accounting, not FLOPs or measured wall-clock speedup."""
    total = n * length + query
    full = total * (total + 1) / 2
    doc = n * length * (length + 1) / 2
    tail = query * n * length + query * (query + 1) / 2
    return full, doc + tail, (1 - hit) * doc + tail


def quality_analysis(evidence, root=ROOT):
    """Verify record pairing behind the descriptive document-count curves."""
    common = None
    for n, *means in evidence["artifacts"]["T5"]["rows"]:
        rows = source(evidence, f"doc_sweep_n{n}", root)
        arms = defaultdict(dict)
        for row in rows:
            parts = row["run_id"].split(":")
            if row["metric"] not in ("em", "f1"):
                continue
            key = (parts[3], row["metric"])
            if (
                parts[0] != "phase2_train_free"
                or parts[2] != "hotpot_qa"
                or parts[4] != "seed=0"
                or parts[5] != "3a59828b"
                or parts[-1] in arms[key]
            ):
                raise ValueError("T5 duplicate or unexpected row")
            arms[key][parts[-1]] = row["value"]
        keys = [(mask, metric) for metric in ("em", "f1") for mask in ("full", "ddi_qf")]
        if set(arms) != set(keys):
            raise ValueError("T5 missing arms")
        for key, expected_mean in zip(keys, means, strict=True):
            ids = set(arms[key])
            if len(ids) != 100 or (common is not None and ids != common):
                raise ValueError("T5 record pairing failed")
            common = ids
            if not math.isclose(statistics.mean(arms[key].values()), expected_mean, abs_tol=1e-12):
                raise ValueError("T5 aggregate mismatch")
    return {"common_record_ids": sorted(common), "n_per_arm": 100, "training_seeds": 1}


def cache_analysis(evidence, root=ROOT):
    timing = source(evidence, "cache_v53_timing", root)
    validation = source(evidence, "cache_v53_validation", root)
    summary = source(evidence, "cache_v53_summary", root)
    if summary["admissible_paper_evidence"] is not False:
        raise ValueError("V53 must remain diagnostic")
    modes = ("ddi_cache", "ddi_recompute", "full_causal")
    expected = {(s, r, m) for s in range(3) for r in range(60) for m in modes}
    indexed = {(r["seed"], r["request"], r["mode"]): r for r in timing}
    if len(timing) != 540 or set(indexed) != expected:
        raise ValueError("V53 timing keys are incomplete or duplicated")
    cases = {(r["seed"], r["case"]): r for r in validation}
    expected_cases = {
        (s, c)
        for s in range(3)
        for c in ["cold", "warm", "permuted", "partial", *[f"stream_{i}" for i in range(60)]]
    }
    if set(cases) != expected_cases or len(validation) != 192:
        raise ValueError("V53 validation keys are incomplete or duplicated")
    groups = defaultdict(list)
    qps = {m: [] for m in modes}
    for seed in range(3):
        for mode in modes:
            measured = [indexed[seed, r, mode] for r in range(10, 60)]
            value = 50000 / sum(r["elapsed_ms"] for r in measured)
            if not math.isclose(value, summary["qps"][mode][seed], rel_tol=1e-10):
                raise ValueError("V53 QPS summary does not match traces")
            qps[mode].append(value)
        for request in range(60):
            rows = [indexed[seed, request, mode] for mode in modes]
            for row in rows:
                if (
                    row["doc_lengths"] != [500] * 5
                    or row["query_len"] != 32
                    or row["generated_count"] != 16
                    or row["decode_forwards"] != 15
                    or len(row["generated_ids"]) != 16
                    or row["warmup"] != (request < 10)
                    or not math.isfinite(row["elapsed_ms"])
                    or row["elapsed_ms"] <= 0
                    or row["indices"] != rows[0]["indices"]
                ):
                    raise ValueError("V53 geometry, pairing or fixed-work contract failed")
            cached, fresh, _full = rows
            check = cases[seed, f"stream_{request}"]
            if (
                check["cache"]["generated_ids"] != cached["generated_ids"]
                or check["fresh"]["generated_ids"] != fresh["generated_ids"]
            ):
                raise ValueError("V53 validation and timing outputs differ")
            equal = cached["generated_ids"] == fresh["generated_ids"]
            if equal != check["greedy_equal"] or len(cached["hit_state"]) != 5:
                raise ValueError("V53 behavioral/hit contract failed")
            if request >= 10:
                groups[sum(cached["hit_state"])].append(
                    {
                        "seed": seed,
                        "request": request,
                        "equal": equal,
                        "cached_ms": cached["elapsed_ms"],
                        "fresh_ms": fresh["elapsed_ms"],
                        "ratio": fresh["elapsed_ms"] / cached["elapsed_ms"],
                    }
                )
    bins = []
    for hits, rows in sorted(groups.items()):
        latencies = sorted(r["cached_ms"] for r in rows)

        def quantile(q, latencies=latencies):
            pos = q * (len(latencies) - 1)
            lo, hi = math.floor(pos), math.ceil(pos)
            return latencies[lo] + (pos - lo) * (latencies[hi] - latencies[lo])

        bins.append(
            {
                "hits": hits,
                "n": len(rows),
                "equal": sum(r["equal"] for r in rows),
                "cached_median_ms": statistics.median(latencies),
                "q25_ms": quantile(0.25),
                "q75_ms": quantile(0.75),
                "fresh_median_ms": statistics.median(r["fresh_ms"] for r in rows),
                "median_paired_ratio": statistics.median(r["ratio"] for r in rows),
                "by_seed": [sum(r["seed"] == s for r in rows) for s in range(3)],
            }
        )
    failures = sum(not r["greedy_equal"] for r in validation)
    numerical = sum(not r["comparison"]["numerical_pass"] for r in validation)
    if failures != 25 or numerical != 192 or sum(b["n"] - b["equal"] for b in bins) != 19:
        raise ValueError("V53 failed gates must not be dropped")
    for mode in ("ddi_recompute", "full_causal"):
        ratios = [c / f for c, f in zip(qps["ddi_cache"], qps[mode], strict=True)]
        mean = statistics.mean(ratios)
        half = T95 * statistics.stdev(ratios) / math.sqrt(3)
        recorded = summary["paired_speedup"][mode]
        for observed, expected_value in zip(
            [mean, mean - half, mean + half, *ratios],
            [recorded["mean"], *recorded["ci95_t"], *recorded["per_seed"]],
            strict=True,
        ):
            if not math.isclose(observed, expected_value, rel_tol=1e-10):
                raise ValueError("V53 paired-ratio summary mismatch")
    artifact = evidence["artifacts"]["V53_cache_diagnostic"]
    hit_rate = sum(b["hits"] * b["n"] for b in bins) / 750
    if (
        artifact["status"] != "diagnostic_not_admissible"
        or not math.isclose(artifact["hit_rate"], hit_rate)
        or artifact["behavior_failures"]
        != {"greedy_ids": [25, 192], "measured_streams": [19, 150], "numerical_logits": [192, 192]}
    ):
        raise ValueError("V53 evidence boundary mismatch")
    return {
        "bins": bins,
        "qps": qps,
        "summary": summary,
        "hits": sum(b["hits"] * b["n"] for b in bins),
        "lookups": 750,
        "validation_failures": failures,
        "numerical_failures": numerical,
    }


def _pinned_matches(observed, expected, context):
    """Chain check: an evidence row must equal the hash-pinned artifact value."""
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{context}: pinned row does not match artifact shape")
        for index, (obs, exp) in enumerate(zip(observed, expected, strict=True)):
            _pinned_matches(obs, exp, f"{context}[{index}]")
        return
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise ValueError(f"{context}: unsupported artifact value type")
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise ValueError(f"{context}: pinned row must be numeric")
    if not math.isfinite(observed) or not math.isclose(
        observed, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{context}: pinned row {observed!r} does not match artifact {expected!r}")


def v65_serving_analysis(evidence, root=ROOT):
    """Verify T9 rows against the pinned V65 benchmark and combined-serving results."""
    benchmark = source(evidence, "v65_benchmark_result", root)
    combined = source(evidence, "v65_combined_result", root)
    artifact = evidence["artifacts"]["T9_v65_serving"]
    expected = {}
    for arm, label in (("A", "A monolithic"), ("B", "B canonical fresh"), ("C", "C all-hit reuse")):
        ci = benchmark["aggregate"]["arms"][arm]["wall_ms_ci"]
        expected[("benchmark", label)] = [ci["point"], [ci["ci_low"], ci["ci_high"]]]
    serving = combined["validation_1_combined_serving"]
    for arm, label in (
        ("A", "A monolithic"),
        ("all_hit", "all-hit reuse"),
        ("mixed_hit", "mixed-hit"),
    ):
        ci = serving["arms"][arm]["wall_ms_ci"]
        expected[("combined", label)] = [ci["point"], [ci["ci_low"], ci["ci_high"]]]
    for label, ratios, key in (
        ("benchmark C/A", benchmark["aggregate"]["ratios"], "C_over_A"),
        ("benchmark B/A", benchmark["aggregate"]["ratios"], "B_over_A"),
        ("combined A/all-hit", serving["ratios"], "A_over_all_hit"),
        ("combined A/mixed-hit", serving["ratios"], "A_over_mixed_hit"),
    ):
        ratio = ratios[key]
        expected[("ratio", label)] = [ratio["point"], [ratio["ci_low"], ratio["ci_high"]]]
    for stage, label, point, ci in artifact["rows"]:
        key = (stage, label)
        if key not in expected:
            raise ValueError(f"T9_v65_serving row is not backed by the pinned artifacts: {key}")
        _pinned_matches(point, expected[key][0], f"T9_v65_serving {label} median")
        _pinned_matches(ci, expected[key][1], f"T9_v65_serving {label} CI")
    if len(artifact["rows"]) != len(expected):
        raise ValueError("T9_v65_serving rows must cover every pinned arm and ratio")
    gate = benchmark["aggregate"]["gate"]
    if gate["pass"] is not True or gate["equivalence_all_equal"] is not True:
        raise ValueError("V65 benchmark gate must pass with all-equal outputs")
    equivalence = benchmark["aggregate"]["equivalence"]
    _pinned_matches(
        artifact["byte_identical_ac"],
        [equivalence["A"]["byte_identical_records"], equivalence["C"]["byte_identical_records"]],
        "T9_v65_serving byte-identical counts",
    )
    records = equivalence["A"]["records"]
    if records != equivalence["C"]["records"] or records != 24:
        raise ValueError("V65 benchmark record count mismatch")
    walls = serving["schedule_walls"]
    _pinned_matches(
        artifact["schedule_walls_s"]["A"],
        walls["A"]["wall_seconds_median"],
        "T9_v65_serving A schedule wall",
    )
    _pinned_matches(
        artifact["schedule_walls_s"]["all_hit"],
        walls["all_hit"]["wall_seconds_median"],
        "T9_v65_serving all-hit schedule wall",
    )
    return {
        "records": records,
        "byte_identical_ac": artifact["byte_identical_ac"],
        "schedule_walls_s": artifact["schedule_walls_s"],
    }


def qa_confirmation_analysis(evidence, root=ROOT):
    """Verify T10 rows against the pinned V64 QA audit and V65 combined confirmation."""
    v60 = source(evidence, "v60_confirmation_result", root)
    requests = source(evidence, "v60_confirmation_requests", root)
    audit = source(evidence, "v64_qa_recomputation", root)
    combined = source(evidence, "v65_combined_result", root)
    artifact = evidence["artifacts"]["T10_qa_confirmation"]
    if len(requests) != 24:
        raise ValueError("V60 confirmation requests.jsonl must hold 24 records")
    if audit["input_gate"]["status"] != "pass" or audit["record_reconciliation"]["matched"] != 24:
        raise ValueError("V64 QA recomputation gates failed")
    if audit["aggregate_reproduction"]["status"] != "pass":
        raise ValueError("V64 QA recomputation must reproduce the stored aggregates")
    for arm in ("A", "B"):
        for metric in ("em", "f1"):
            stored = audit["aggregate_reproduction"]["arms"][arm][metric]
            if not math.isclose(
                v60["aggregate"]["quality_mean"][arm][metric], stored["stored"], rel_tol=1e-12
            ):
                raise ValueError(f"V60 arm {arm} {metric} does not match the audited aggregate")
    contrast = audit["contrasts"]["B_minus_A"]
    if contrast["decision_rule"]["verdict"] != audit["headline_verdict"]["verdict"]:
        raise ValueError("V64 QA headline verdict mismatch")
    expected = {}
    for metric, key in (("EM", "em"), ("F1", "f1")):
        block = contrast[key]
        expected[("V64 (n=24)", metric)] = [
            block["mean_delta"],
            block["ci95"],
            contrast["decision_rule"]["verdict"],
        ]
    qa = combined["validation_2_qa_confirmation"]
    if qa["paired_records"] != 100 or qa["errors"] != 0 or qa["delta_direction"] != "C - A":
        raise ValueError("V65 combined QA confirmation is incomplete")
    if not math.isclose(qa["decision"]["margin"], artifact["floor"], abs_tol=1e-12):
        raise ValueError("T10_qa_confirmation floor does not match the pinned decision margin")
    verdict = "non-inferior" if qa["decision"]["non_inferior"] else "not non-inferior"
    for metric, key in (("EM", "em"), ("F1", "f1")):
        block = qa["analysis"][key]
        expected[("V65 (n=100)", metric)] = [block["mean"], block["ci95"], verdict]
    for cohort, metric, mean, lo, hi, row_verdict in artifact["rows"]:
        key = (cohort, metric)
        if key not in expected:
            raise ValueError(f"T10_qa_confirmation row is not backed by pinned artifacts: {key}")
        _pinned_matches(mean, expected[key][0], f"T10_qa_confirmation {cohort} {metric} mean")
        _pinned_matches([lo, hi], expected[key][1], f"T10_qa_confirmation {cohort} {metric} CI")
        if row_verdict != expected[key][2]:
            raise ValueError(f"T10_qa_confirmation {cohort} {metric} verdict mismatch")
    if len(artifact["rows"]) != len(expected):
        raise ValueError("T10_qa_confirmation rows must cover every pinned contrast")
    if artifact["prediction_equal"] != [qa["prediction_equal"], 100]:
        raise ValueError("T10_qa_confirmation prediction-equal count mismatch")
    if artifact["first_token_equal"] != [qa["first_token_equal"], 100]:
        raise ValueError("T10_qa_confirmation first-token-equal count mismatch")
    return {
        "v64_verdict": contrast["decision_rule"]["verdict"],
        "v65_non_inferior": qa["decision"]["non_inferior"],
        "floor": artifact["floor"],
        "prediction_equal": artifact["prediction_equal"],
        "first_token_equal": artifact["first_token_equal"],
    }


def ttft_grid_analysis(evidence, root=ROOT):
    """Verify T11 rows against the pinned V59 grid analysis and summary."""
    analysis = source(evidence, "v59_grid_analysis", root)
    summary = source(evidence, "v59_grid_summary", root)
    artifact = evidence["artifacts"]["T11_v59_crossover"]
    geometries = {(g["N"], g["L"]): g for g in analysis["geometries"]}
    if len(analysis["geometries"]) != 9 or set(geometries) != {
        (n, length) for n in (2, 5, 10) for length in (32, 128, 512)
    }:
        raise ValueError("V59 grid must hold the 9-cell N x L design")
    if summary["cell_count"] != artifact["cell_count"] or artifact["cell_count"] != 27:
        raise ValueError("V59 grid cell count mismatch")
    if summary["admissible_paper_evidence"] is not False:
        raise ValueError("V59 grid must remain diagnostic")
    crossovers = 0
    for n, length, fresh, warm, speedup, lo, hi, crossover in artifact["rows"]:
        geometry = geometries[(n, length)]
        context = f"T11_v59_crossover ({n},{length})"
        _pinned_matches(fresh, geometry["fresh_median_ms"], f"{context} fresh median")
        _pinned_matches(warm, geometry["warm_median_ms"], f"{context} warm median")
        _pinned_matches(speedup, geometry["warm_speedup"]["mean"], f"{context} warm speedup")
        ci = geometry["warm_speedup"]["ci95"]
        _pinned_matches([lo, hi], [min(ci), max(ci)], f"{context} warm speedup CI")
        mean_crossover = geometry["hit_rate_crossover"]
        if (crossover is None) != (mean_crossover is None):
            raise ValueError(f"{context} crossover presence mismatch")
        if mean_crossover is not None:
            _pinned_matches(crossover, mean_crossover["mean"], f"{context} crossover")
            crossovers += 1
    if len(artifact["rows"]) != 9:
        raise ValueError("T11_v59_crossover must render the full 9-cell grid")
    return {
        "cells": len(artifact["rows"]),
        "crossover_cells": crossovers,
        "summary_crossover_cells": summary["finite_reuse_crossovers"],
    }


def v57_matched_analysis(evidence, root=ROOT):
    """Verify T12 rows against the pinned V57 summary and post-audit."""
    summary = source(evidence, "v57_matched_summary", root)
    audit = source(evidence, "v57_matched_post_audit", root)
    artifact = evidence["artifacts"]["T12_v57_matched"]
    if audit["original_summary_sha256"] != evidence["sources"]["v57_matched_summary"]["sha256"]:
        raise ValueError("V57 post-audit does not pin the pinned summary file")
    if (
        audit["artifact_integrity_pass"] is not True
        or audit["request_count"] != artifact["n_requests"]
    ):
        raise ValueError("V57 post-audit integrity gate failed")
    expected = {}
    for metric, key in (("EM", "em"), ("F1", "f1"), ("presence", "presence")):
        contrast = audit["contrasts"][f"full_minus_ddi_qf_{key}"]
        expected[metric] = [
            summary["quality"]["full"][key],
            summary["quality"]["ddi_qf"][key],
            contrast["mean"],
            contrast["ci95_paired_bootstrap"],
        ]
    for metric, full, ddi, delta, lo, hi in artifact["rows"]:
        if metric not in expected:
            raise ValueError(f"T12_v57_matched metric is not backed by pinned artifacts: {metric}")
        block = expected[metric]
        _pinned_matches(full, block[0], f"T12_v57_matched {metric} full mean")
        _pinned_matches(ddi, block[1], f"T12_v57_matched {metric} ddi_qf mean")
        _pinned_matches(delta, block[2], f"T12_v57_matched {metric} contrast")
        _pinned_matches([lo, hi], block[3], f"T12_v57_matched {metric} contrast CI")
    if len(artifact["rows"]) != len(expected):
        raise ValueError("T12_v57_matched rows must cover every pinned metric")
    speedups = (
        ("prefill latency ratio", "prefill", "ddi_qf_prefill_speedup"),
        ("request latency ratio", "request", "ddi_qf_request_speedup"),
    )
    if [row[0] for row in artifact["timing_ratios"]] != [name for name, _, _ in speedups]:
        raise ValueError("T12_v57_matched timing ratio labels mismatch")
    for (label, scope, audit_key), row in zip(speedups, artifact["timing_ratios"], strict=True):
        block = summary["ddi_qf_speedup"][scope]
        audited = audit["contrasts"][audit_key]
        if not math.isclose(block["mean"], audited["mean"], rel_tol=1e-12):
            raise ValueError(f"V57 {scope} speedup differs between summary and post-audit")
        _pinned_matches(row[1], block["mean"], f"T12_v57_matched {label}")
        _pinned_matches(
            [row[2], row[3]], block["ci95_paired_bootstrap"], f"T12_v57_matched {label} CI"
        )
    return {"n_requests": artifact["n_requests"]}


def reference_artifact_analysis(evidence, root=ROOT):
    """Validate the pinned V58/V61/V64-timing artifacts that carry no rendered table."""
    v58 = source(evidence, "v58_cache_qa_summary", root)
    if v58["admissible_paper_evidence"] is not False or v58["record_count"] != 100:
        raise ValueError("V58 cache QA must remain a diagnostic with 100 records")
    v61 = source(evidence, "v61_timing_result", root)
    cells = v61["cells"]
    if len(cells) != 8 or any(cell["status"] != "ok" for cell in cells):
        raise ValueError("V61 timing grid must hold 8 successful cells")
    indexed = {}
    for cell in cells:
        key = (cell["n_docs"], cell["doc_length"])
        if key in indexed:
            raise ValueError("V61 timing grid cells are duplicated")
        ratio = cell["ratio"]
        if not math.isfinite(ratio["mean"]) or "ci95_low" not in ratio or "ci95_high" not in ratio:
            raise ValueError("V61 timing grid cell ratios are invalid")
        indexed[key] = ratio["mean"]
    audit = source(evidence, "v64_timing_recomputation", root)
    if (
        audit["input_gate"]["status"] != "pass"
        or audit["stored_aggregate_reproduction"]["status"] != "pass"
    ):
        raise ValueError("V64 timing recomputation gates failed")
    for record in audit["inventory_cross_check"]["ratios"].values():
        if record["match"] is not True:
            raise ValueError("V64 timing inventory cross-check failed")
    for cell in audit["timing"]["v61_cross_check"]["v61_cells"]:
        reference = indexed[(cell["n_docs"], cell["doc_length"])]
        if not math.isclose(reference, cell["ratio_mean"], abs_tol=1e-4):
            raise ValueError("V61 cells differ between the run result and the audit cross-check")
    return {
        "v58": "diagnostic_not_admissible",
        "v61_cells": len(cells),
        "v64_timing_recheck": "pass",
    }


def v66_boundary_analysis(evidence, root=ROOT):
    """Verify T13 rows against the pinned V66 fragment and boundary summary."""
    artifact = evidence["artifacts"]["T13_v66_boundary"]
    if artifact["data_role"] != "development":
        raise ValueError("T13_v66_boundary must remain development-role evidence")
    fragment = source(evidence, "v66_boundary_t9_followup", root)
    summary = source(evidence, "v66_boundary_summary", root)
    if fragment["schema_version"] != "v66_boundary_t9_followup_v1":
        raise ValueError("V66 fragment schema mismatch")
    if summary["schema_version"] != "v66_boundary_summary_v1":
        raise ValueError("V66 boundary summary schema mismatch")
    expected = [
        [row["L"], row["decode_budget"], row["crossover_h_wall"], row["crossover_h_ttft"]]
        for row in fragment["rows"]
    ]
    rows = artifact["rows"]
    if len(rows) != 15 or len(expected) != 15:
        raise ValueError("T13_v66_boundary must hold the 15-cell length x budget grid")
    wall_crossovers = ttft_crossovers = 0
    for row, want in zip(rows, expected, strict=True):
        context = f"T13_v66_boundary (L={row[0]}, {row[1]})"
        if row[0] != want[0] or row[1] != want[1]:
            raise ValueError(f"{context}: pinned label does not match artifact")
        for observed, wanted, label in (
            (row[2], want[2], "wall crossover"),
            (row[3], want[3], "TTFT crossover"),
        ):
            if (observed is None) != (wanted is None):
                raise ValueError(f"{context}: pinned {label} does not match artifact")
            if wanted is not None:
                _pinned_matches(observed, wanted, f"{context} {label}")
        block = summary["crossover"][f"L{row[0]}|{row[1]}"]
        for observed, wanted, label in (
            (row[2], block["wall_crossover_h"], "wall crossover"),
            (row[3], block["ttft_crossover_h"], "TTFT crossover"),
        ):
            if (observed is None) != (wanted is None):
                raise ValueError(f"{context}: summary {label} does not match artifact")
            if wanted is not None:
                _pinned_matches(observed, wanted, f"{context} summary {label}")
        wall_crossovers += row[2] is not None
        ttft_crossovers += row[3] is not None
    if artifact["repeats_per_cell"] != summary["grid_executed"]["repeats_used"]:
        raise ValueError("T13_v66_boundary repeats-per-cell mismatch")
    identity = summary["identity"]
    if identity["all_identical"] is not False or not identity["failing_cells"]:
        raise ValueError("V66 identity divergences must stay visible")
    identical = artifact["byte_identical_repeats"]
    _pinned_matches(
        identical,
        [identity["identity_checked_repeats"] - identity["divergent_repeat_records"],
         identity["identity_checked_repeats"]],
        "T13_v66_boundary byte-identical repeats",
    )
    return {
        "rows": len(rows),
        "wall_crossovers": wall_crossovers,
        "ttft_crossovers": ttft_crossovers,
        "byte_identical_repeats": identical,
        "repeats_per_cell": artifact["repeats_per_cell"],
        "divergent_repeat_records": identity["divergent_repeat_records"],
    }


def v69_attribution_analysis(evidence, root=ROOT):
    """Verify T14 rows against the pinned V69 attribution derivation."""
    artifact = evidence["artifacts"]["T14_v69_attribution"]
    if artifact["data_role"] != "development":
        raise ValueError("T14_v69_attribution must remain development-role evidence")
    attribution = source(evidence, "v69_attribution", root)
    attr = attribution["attribution"]
    if attribution["prereg_verdict"]["verdict"] != "derivation_only":
        raise ValueError("V69 attribution must remain a derivation-only verdict")
    expected_rows = [
        [
            "OPT-1 store-time validation markers",
            "validate_entry (per-request geometry+isfinite scan)",
            attr["opt1_removed_ms"],
        ],
        [
            "OPT-2 2 GiB LRU resident tier",
            "disk_load (synchronous per-doc torch.load)",
            attr["opt2_removed_ms"],
        ],
        [
            "OPT-3 single staged pinned H2D copy per layer (bounded envelope)",
            "cache_assemble (per-entry .to(device) + cat)",
            attr["opt3_envelope_ms"],
        ],
    ]
    rows = artifact["rows"]
    if len(rows) != len(expected_rows):
        raise ValueError("T14_v69_attribution must hold the three pinned optimizations")
    for row, want in zip(rows, expected_rows, strict=True):
        if row[0] != want[0] or row[1] != want[1]:
            raise ValueError(f"T14_v69_attribution label mismatch: {row[0]!r}")
        _pinned_matches(row[2], want[2], f"T14_v69_attribution {row[0]}")
    _pinned_matches(
        math.fsum(row[2] for row in rows), attr["reduction_ms"],
        "T14_v69_attribution removed-phase sum",
    )
    _pinned_matches(
        attr["opt1_removed_ms"] + attr["opt2_removed_ms"], attr["direct_removed_ms"],
        "V69 direct removed sum",
    )
    if not math.isclose(
        attr["direct_frac_of_reduction"] + attr["opt3_envelope_frac_of_reduction"],
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("V69 reduction shares must sum to one")
    if attr["opt3_envelope_within_budget"] is not True:
        raise ValueError("V69 OPT-3 envelope must fit its pinned budget")
    for field, key in (
        ("pre_total_ms", "pre_total_ms"),
        ("post_total_ms", "post_total_ms"),
        ("reduction_ms", "reduction_ms"),
        ("residual_ms", "residual_in_pre_ms"),
        ("residual_frac_of_pre", "residual_in_pre_frac"),
        ("opt3_budget_ms", "opt3_budget_ms"),
        ("opt3_implied_post_ms", "opt3_implied_post_cache_assemble_ms"),
    ):
        _pinned_matches(artifact[field], attr[key], f"T14_v69_attribution {field}")
    _pinned_matches(
        artifact["speedup_vs_pre"], attribution["speedup_vs_pre"],
        "T14_v69_attribution speedup_vs_pre",
    )
    return {
        "speedup_vs_pre": artifact["speedup_vs_pre"],
        "reduction_ms": artifact["reduction_ms"],
        "residual_frac_of_pre": artifact["residual_frac_of_pre"],
        "opt3_envelope_within_budget": attr["opt3_envelope_within_budget"],
        "verdict": attribution["prereg_verdict"]["verdict"],
    }


def coords(rows):
    return " ".join(f"({x},{y:.6f})" for x, y in rows)


def plot(style, rows):
    return rf"\addplot[{style}] coordinates {{{coords(rows)}}};"


def panels(options, bodies, height="5.2cm"):
    # Full-width panels, with legends outside axes and no on-bar prose.
    out = [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.55cm},"
        + rf"width=.455\linewidth,height={height},"
        + r"grid=major,grid style={gray!15},tick label style={font=\small},"
        r"label style={font=\small},title style={font=\small\bfseries},"
        r"legend style={font=\small,draw=none,at={(.5,-.28)},anchor=north,legend columns=2,"
        r"/tikz/every even column/.append style={column sep=5pt}}]",
    ]
    for option, body in zip(options, bodies, strict=True):
        out += [rf"\nextgroupplot[{option}]", *body]
    return "\n".join([*out, r"\end{groupplot}", r"\end{tikzpicture}", ""])


def mechanism():
    out = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,>=stealth,font=\small,"
        r"box/.style={draw,very thick,rounded corners=3pt,text width=3.4cm,"
        r"minimum height=1cm,align=center,inner sep=5pt}]",
        r"\node[anchor=west,text width=3.5cm,font=\bfseries,text=ddiBlue]"
        r" at (0,4.3) {DDI-QF attention};",
    ]
    # Three two-token document blocks, two query and two answer positions.
    for row in range(10):
        for col in range(10):
            allowed = col <= row and (row >= 6 or row // 2 == col // 2)
            color = "ddiBlue" if row < 6 else "ddiGold" if row < 8 else "ddiPurple"
            out.append(
                rf"\fill[{color if allowed else 'gray!9'}]"
                rf" ({col * 0.29:.2f},{3.6 - row * 0.29:.2f}) rectangle ++(.27,.27);"
            )
    out += [
        r"\node[anchor=north west,text width=3.8cm,align=left] at (0,.8)"
        r" {Keys $\rightarrow$; rows $\downarrow$\par Blue: documents\par"
        r" Orange: query\par Purple: answer};",
        r"\node[anchor=west,text width=8.6cm,font=\bfseries,text=ddiGold]"
        r" at (4.1,4.3) {COLD MISS: target vs implementation};",
        r"\node[box,draw=ddiGold,fill=ddiGold!9] (a) at (6,3.3)"
        r" {Target: sparse prefill\par Parity gate pending};",
        r"\node[box,draw=ddiBlue,fill=ddiBlue!8] (b) at (10.5,3.3)"
        r" {V53: dense SDPA\par per-document prefill};",
        r"\node[box,draw=ddiGreen,fill=ddiGreen!10] (c) at (6,1.6)"
        r" {Persist KVs\par exact identity key};",
        r"\node[box,draw=ddiPurple,fill=ddiPurple!8] (d) at (10.5,1.6)"
        r" {WARM HIT: load KVs\par Skip document prefill};",
        r"\draw[->,thick,dashed,ddiGold] (a) -- (c);",
        r"\draw[->,thick,ddiBlue] (b.south) -- ++(0,-.25) -| (c.north east);",
        r"\draw[->,thick,ddiGreen] (c) -- (d);",
        r"\node[anchor=west,text width=8.3cm,align=left] at (4.2,.05)"
        r" {Dashed: unvalidated target kernel. Solid: implemented cache path."
        r" V53 does not measure a sparse kernel.};",
        r"\end{tikzpicture}",
        "",
    ]
    return "\n".join(out)


def cache_cartoon():
    return r"""\begin{tikzpicture}[x=1cm,y=1cm,>=stealth,font=\small,
doc/.style={draw,very thick,rounded corners,text width=1.35cm,minimum height=.65cm,align=center},
note/.style={text width=4.8cm,align=left}]
\node[anchor=west,font=\bfseries] at (0,2.9)
{One KV per document identity; assembly follows each request};
\node[anchor=east] at (1.8,2) {Request 1};
\node[doc,draw=ddiGold,fill=ddiGold!15] at (3,2) {Doc 7};
\node[doc,draw=ddiGold,fill=ddiGold!15] at (5,2) {Doc 2};
\node[doc,draw=ddiGold,fill=ddiGold!15] at (7,2) {Doc 19};
\node[note] at (11,2) {Cold documents: compute, persist; query and decode follow.};
\node[anchor=east] at (1.8,.8) {Request 2};
\node[doc,draw=ddiGreen,fill=ddiGreen!15] at (3,.8) {Doc 19};
\node[doc,draw=ddiGreen,fill=ddiGreen!15] at (5,.8) {Doc 7};
\node[doc,draw=ddiGold,fill=ddiGold!15] at (7,.8) {Doc 4};
\node[note] at (11,.8) {Two hits in a new order; one miss. Same mask geometry.};
\draw[->,ddiPurple,thick] (7,1.65) -- (7,1.4) -- (3,1.4) -- (3,1.15);
\draw[->,ddiGreen,thick] (3,1.65) -- (3,1.55) -- (5,1.55) -- (5,1.15);
\node[anchor=west,text width=13cm,align=left] at (0,-.15)
{Illustration, not an observed request pair. Restart positions enable the intended reuse;
numerical and greedy equivalence must still be checked.};
\end{tikzpicture}
"""


def render_visuals(evidence):
    a = evidence["artifacts"]
    stats = cache_analysis(evidence)
    quality = quality_analysis(evidence)
    historical = flex_history(evidence)
    serving = v65_serving_analysis(evidence)
    qa = qa_confirmation_analysis(evidence)
    grid = ttft_grid_analysis(evidence)
    matched = v57_matched_analysis(evidence)
    reference = reference_artifact_analysis(evidence)
    boundary = v66_boundary_analysis(evidence)
    attribution = v69_attribution_analysis(evidence)
    out = {"V1_mask_geometry.tex": mechanism()}
    t3 = a["T3"]["rows"]
    out["V2_efficiency_intuition.tex"] = panels(
        [
            "title={Exact structural pair reduction},xlabel={documents $N$},ylabel={$P_F/P_D$},"
            "xmin=0,xmax=26,ymin=0,ymax=22,xtick={1,5,9,13,25}",
            "title={Measured SDPA prefill},xlabel={documents $N$},ylabel={median prefill (ms)},"
            "xmin=0,xmax=26,ymin=0,ymax=2100,xtick={1,5,9,13,25}",
        ],
        [
            [plot("ddiBlue,very thick,mark=*", ((r[0], 1 / (1 - r[7])) for r in t3))],
            [
                plot("ddiGold,very thick,mark=square*", ((r[0], r[1]) for r in t3)),
                plot("ddiBlue,very thick,mark=*,dashed", ((r[0], r[2]) for r in t3)),
                r"\legend{Full,DDI-QF}",
            ],
        ],
    )
    means = []
    dots = []
    for i, mode in enumerate(("full_causal", "ddi_recompute", "ddi_cache")):
        values = stats["qps"][mode]
        mean = statistics.mean(values)
        half = T95 * statistics.stdev(values) / math.sqrt(3)
        means.append(f"({i},{mean:.6f}) +- (0,{half:.6f})")
        dots.extend((i + (s - 1) * 0.09, v) for s, v in enumerate(values))
    out["V4_production_opportunity.tex"] = panels(
        [
            "title={V53: measured QPS, diagnostic only},ylabel={requests / second},"
            "xtick={0,1,2},xticklabels={Full,Fresh DDI,Cached},xmin=-.5,xmax=2.5,ymin=0,ymax=2.3",
            "title={Paired seed ratios, not preserved quality},ylabel={cache / reference QPS},"
            "xtick={0,1},xticklabels={Fresh DDI,Full causal},xmin=-.5,xmax=1.5,ymin=.95,ymax=1.36",
        ],
        [
            [
                r"\addplot[only marks,mark=square*,ddiGreen,very thick,"
                r"error bars/.cd,y dir=both,y explicit] coordinates {" + " ".join(means) + "};",
                rf"\addplot[only marks,mark=*,ddiPurple] coordinates {{{coords(dots)}}};",
            ],
            [
                r"\addplot[gray,dashed] coordinates {(-.5,1) (1.5,1)};",
                r"\addplot[only marks,mark=square*,ddiGreen,very thick,"
                r"error bars/.cd,y dir=both,y explicit] coordinates {"
                + " ".join(
                    f"({i},{v['mean']:.6f}) +- (0,{v['ci95_t'][1] - v['mean']:.6f})"
                    for i, v in enumerate(stats["summary"]["paired_speedup"].values())
                )
                + "};",
                plot(
                    "only marks,mark=*,ddiPurple",
                    (
                        (i + (s - 1) * 0.09, v)
                        for i, d in enumerate(stats["summary"]["paired_speedup"].values())
                        for s, v in enumerate(d["per_seed"])
                    ),
                ),
            ],
        ],
    )
    t5 = a["T5"]["rows"]
    out["V5_doc_quality.tex"] = panels(
        [
            f"title={{{metric}: same 100 Hotpot IDs}},xlabel={{documents $N$}},ylabel={{{metric}}},"
            "xtick={2,5,10},xmin=1,xmax=11,ymin=.3,ymax=.85"
            for metric in ("EM", "F1")
        ],
        [
            [
                plot("ddiGold,very thick,mark=square*", ((r[0], r[col]) for r in t5)),
                plot("ddiBlue,very thick,mark=*", ((r[0], r[col + 1]) for r in t5)),
                r"\legend{Full,DDI-QF}",
            ]
            for col in (1, 3)
        ],
    )
    t2 = [r for r in a["T2"]["rows"] if r[0] == "Qwen2.5-7B"]
    out["V6_mask_quality.tex"] = panels(
        [
            f"title={{{metric}: Qwen2.5-7B base}},ylabel={{{metric}}},"
            "xtick={0,1,2,3},xticklabels={Full,QF,Strict,Blind},ymin=0,ymax=.8,xmin=-.5,xmax=3.5"
            for metric in ("EM", "F1")
        ],
        [
            [
                plot(
                    "ddiGold,very thick,mark=square*,only marks",
                    ((i, r[col]) for i, r in enumerate(t2)),
                ),
                plot(
                    "ddiBlue,very thick,mark=*,only marks",
                    ((i, r[col + 1]) for i, r in enumerate(t2)),
                ),
                r"\legend{Hotpot,MuSiQue}",
            ]
            for col in (2, 5)
        ],
    )
    bins = stats["bins"]
    out["V7_hit_diagnostics.tex"] = panels(
        [
            "title={Matched requests: latency by cache hits},xlabel={hits out of five documents},"
            "ylabel={latency (ms)},xmin=-.4,xmax=5.4,xtick={0,1,2,3,4,5},ymin=350,ymax=1050",
            "title={Behavioral fidelity is not QA quality},xlabel={hits out of five documents},"
            "ylabel={identical greedy sequences (\\%)},xmin=-.4,xmax=5.4,"
            "xtick={0,1,2,3,4,5},ymin=0,ymax=108",
        ],
        [
            [
                r"\addplot[only marks,mark=*,ddiGreen,very thick,"
                r"error bars/.cd,y dir=both,y explicit] coordinates {"
                + " ".join(
                    f"({b['hits']},{b['cached_median_ms']:.3f}) "
                    f"+= (0,{b['q75_ms'] - b['cached_median_ms']:.3f}) "
                    f"-= (0,{b['cached_median_ms'] - b['q25_ms']:.3f})"
                    for b in bins
                )
                + "};",
                plot(
                    "only marks,mark=square*,ddiBlue",
                    ((b["hits"], b["fresh_median_ms"]) for b in bins),
                ),
                r"\legend{Cached median/IQR,Matched fresh median}",
            ],
            [
                plot(
                    "only marks,mark=*,ddiPurple,very thick",
                    ((b["hits"], 100 * b["equal"] / b["n"]) for b in bins),
                )
            ],
        ],
    )
    out["V8_cache_requests.tex"] = cache_cartoon()
    heat = [r"\begin{tikzpicture}[x=.46cm,y=3.3cm,font=\small]"]
    for n in range(2, 26):
        for j in range(20):
            h = (j + 0.5) / 20
            full, fresh, reuse = pair_cost(n, 500, 32, h)
            # Log color scale: fixed 1x--200x pair-work ratio, not measured speedup.
            shade = 100 * math.log(full / reuse) / math.log(200)
            heat.append(
                rf"\fill[ddiPurple!{shade:.2f}!white] ({n - 0.5},{j / 20}) rectangle ++(1,.05);"
            )
    heat += [
        r"\draw (1.5,0) rectangle (25.5,1);",
        r"\node[font=\bfseries,anchor=west] at (1.5,1.12)"
        r" {ANALYTICAL ONLY: full / reuse pair-count ratio};",
        r"\node[rotate=90] at (-1,.5) {document hit rate $h$};",
    ]
    for n in (2, 5, 10, 15, 20, 25):
        sparse = 1 - pair_cost(n, 500, 32)[1] / pair_cost(n, 500, 32)[0]
        heat += [
            rf"\node[anchor=north] at ({n},-.03) {{{n}}};",
            rf"\node[anchor=north,font=\footnotesize] at ({n},-.18) {{{100 * sparse:.1f}\%}};",
        ]
    for h in (0, 0.25, 0.5, 0.75, 1):
        heat.append(rf"\node[anchor=east] at (1.3,{h}) {{{h:.2f}}};")
    heat += [
        r"\node[anchor=north] at (13.5,-.34)"
        r" {$N$ (upper row); structural sparsity (lower row), $L=500,Q=32$};",
        r"\draw[ddiGreen,very thick,fill=white] (5,.945333333) circle[x radius=.22,y radius=.031];",
        r"\node[anchor=west,text width=3.7cm,align=left,font=\small] at (27,.78)"
        r" {V53 operating point:\par $N=5$, $h=94.53\%$\par"
        r" sparsity $77.96\%$\par No sweep measured.};",
        r"\fill[ddiPurple!13.08] (27,.35) rectangle ++(1,.07);",
        r"\fill[ddiPurple!56.54] (28,.35) rectangle ++(1,.07);",
        r"\fill[ddiPurple!100] (29,.35) rectangle ++(1,.07);",
        r"\node[anchor=north west,font=\footnotesize] at (27,.32)"
        r" {$2\times$\quad $20\times$\quad $200\times$};",
        r"\node[anchor=north west,font=\footnotesize] at (27,.2) {Log color scale};",
        r"\end{tikzpicture}",
    ]
    out["V9_analytical_surface.tex"] = "\n".join(heat) + "\n"
    out["V10_coverage.tex"] = (
        r"""\begin{tabular}{@{}p{4.3cm}p{5.4cm}p{5.4cm}@{}}
\toprule
\textbf{Requested relationship} & \textbf{Measured (validation program)} &
\textbf{Still missing} \\
\midrule
Quality / prefill speedup & \textcolor{ddiBlue}{V57: matched full-vs-DDI QA + timing,"""
        r""" same 100 requests} &
\textcolor{ddiRed}{Matched QA under a validated sparse kernel} \\[5pt]
Quality / KV reuse & \textcolor{ddiGreen}{V58 + V65: gold-answer QA under hits; V64 audited} &
\textcolor{ddiRed}{Non-inferiority established (fails narrowly)} \\[5pt]
Warm-path serving win & \textcolor{ddiGreen}{V65: 2.05$\times$ TTFT, byte-identical, gate passed} &
\textcolor{ddiRed}{Mixed/partial-hit beating baseline; prefix-cache baseline} \\[5pt]
Hit rate / sparsity & \textcolor{ddiPurple}{V59 measured crossover + analytical surface} &
\textcolor{ddiRed}{Controlled workload$\times$mask sweep at scale} \\
\bottomrule
\end{tabular}
"""
    )
    t8 = a["T8_transfer_estimands"]["rows"]
    out["V3_transfer_estimands.tex"] = panels(
        [
            f"title={{{arch}}},ylabel={{paired change}},xtick={{0,1,2,3}},"
            "xticklabels={H-EM,H-F1,M-EM,M-F1},ymin=-.18,ymax=.17,"
            "ybar,bar width=5pt,enlarge x limits=.18,legend columns=3"
            for arch in ("Qwen2.5-7B", "Llama-3.1-8B")
        ],
        [
            [
                plot(
                    f"fill={color},draw=none",
                    ((i, r[col]) for i, r in enumerate([r for r in t8 if r[0] == arch])),
                )
                for col, color in ((4, "ddiBlue"), (5, "ddiGold"), (6, "ddiPurple"))
            ]
            + [r"\legend{DDI,Full,Gap}"]
            for arch in ("Qwen2.5-7B", "Llama-3.1-8B")
        ],
    )
    table = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Hits & $n$ & Seed counts (0/1/2) & Equal IDs & Cache ms & Fresh ms & Paired ratio \\",
        r"\midrule",
    ]
    for b in bins:
        counts = "/".join(map(str, b["by_seed"]))
        table.append(
            f"{b['hits']} & {b['n']} & {counts} & {b['equal']}/{b['n']} & "
            f"{b['cached_median_ms']:.1f} & {b['fresh_median_ms']:.1f} & "
            f"{b['median_paired_ratio']:.3f} " + r"\\"
        )
    out["V11_flex_history.tex"] = (
        r"\begin{minipage}{\linewidth}\centering"
        + "\n"
        + panels(
            [
                rf"title={{{title}\\HISTORICAL / UNVALIDATED}},"
                r"title style={align=center,font=\small\bfseries},"
                r"xlabel={shape $(N,L)$},ylabel={raw latency ratio (full / DDI)},"
                r"xtick={0,1,2,3},xticklabels={{$(3,512)$},{$(9,512)$},"
                r"{$(25,512)$},{$(25,2048)$}},"
                r"x tick label style={font=\scriptsize},xmin=-.3,xmax=3.3,ymin=0,"
                + f"ymax={maximum}"
                for title, maximum in (("Raw Flex kernel", 9), ("Whole-model Flex prefill", 2))
            ],
            [
                [
                    plot("gray,dashed", [(-0.3, 1), (3.3, 1)]),
                    plot(style, ((i, r["ratio"]) for i, r in enumerate(historical[scope]))),
                ]
                for scope, style in (
                    ("kernel", "ddiBlue,very thick,mark=*"),
                    ("model_prefill", "ddiGold,very thick,dashed,mark=square*"),
                )
            ],
        )
        + r"\par\smallskip{\small Independent experiments; not a matched timing decomposition.}"
        + "\n"
        + r"\end{minipage}"
        + "\n"
    )
    out["V12_amdahl.tex"] = "\n".join(
        [
            "% Analytical illustration only; no measured timing inputs or fitted fractions.",
            r"\begin{tikzpicture}",
            r"\begin{axis}[width=.9\linewidth,height=6cm,xmin=1,xmax=10,ymin=1,ymax=4,"
            r"xtick={1,2,3,4,5,6,7,8,9,10},grid=major,grid style={gray!15},"
            r"title={ILLUSTRATIVE ONLY: $1/((1-f)+f/S)$},"
            r"title style={font=\small\bfseries},tick label style={font=\small},"
            r"label style={font=\small},xlabel={component speedup $S$},ylabel={total speedup},"
            r"legend style={font=\small,draw=none,at={(.5,-.28)},anchor=north,legend columns=4}]",
            *[
                plot(
                    style + ",very thick,no marks",
                    ((1 + i / 10, amdahl_speedup(fraction, 1 + i / 10)) for i in range(91)),
                )
                for fraction, style in (
                    (0.1, "ddiBlue,solid"),
                    (0.3, "ddiGold,dashed"),
                    (0.5, "ddiGreen,dotted"),
                    (0.8, "ddiPurple,dashdotted"),
                )
            ],
            r"\legend{$f=0.1$,$f=0.3$,$f=0.5$,$f=0.8$}",
            r"\end{axis}",
            r"\node[anchor=north,align=center,font=\small] at (current bounding box.south)"
            r" {Hypothetical baseline component fractions $f$; not measured attention fractions.};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    out["A1_cache_bins.tex"] = "\n".join([*table, r"\bottomrule", r"\end{tabular}", ""])
    out["analysis.json"] = (
        json.dumps(
            {
                "status": "posthoc_diagnostic_not_QA",
                "T5_pairing": quality,
                "flex_history": {"status": "historical_unvalidated", **historical},
                "T9_v65_serving": serving,
                "T10_qa_confirmation": qa,
                "T11_v59_crossover": grid,
                "T12_v57_matched": matched,
                "T13_v66_boundary": boundary,
                "T14_v69_attribution": attribution,
                "reference_artifacts": reference,
                **stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    lineage = {
        "V1_mask_geometry.tex": ["T1_mask_taxonomy", "V53_cache_diagnostic"],
        "V2_efficiency_intuition.tex": ["T3"],
        "V3_transfer_estimands.tex": ["T8_transfer_estimands"],
        "V4_production_opportunity.tex": ["V53_cache_diagnostic"],
        "V5_doc_quality.tex": ["T5"],
        "V6_mask_quality.tex": ["T2"],
        "V7_hit_diagnostics.tex": ["V53_cache_diagnostic"],
        "V8_cache_requests.tex": ["T1_mask_taxonomy", "V53_cache_diagnostic"],
        "V9_analytical_surface.tex": ["T1_mask_taxonomy", "V53_cache_diagnostic"],
        "V10_coverage.tex": ["T2", "T3", "T5", "V53_cache_diagnostic"],
        "V11_flex_history.tex": ["V11_flex_history"],
        "V12_amdahl.tex": [],
        "A1_cache_bins.tex": ["V53_cache_diagnostic"],
    }
    for name in out:
        if name.endswith(".tex"):
            names = sorted({s for artifact in lineage[name] for s in a[artifact]["sources"]})
            hashes = "; ".join(
                f"{s['path']} sha256={s['sha256']}"
                for s in (evidence["sources"][key] for key in names)
            )
            out[name] = (
                "% Generated by paper/render.py; aggregates: paper/evidence.json.\n% Sources: "
                + (hashes or "None (analytical illustration)")
                + "\n"
                + out[name]
            )
    return out
