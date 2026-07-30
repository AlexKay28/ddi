"""Tests for ddi/aggregate.py — YAML-driven JSON table builder.

The aggregator now emits a single JSON envelope per YAML: each ``report.tables[]``
entry becomes one object with a normalized ``{model: axis1: axis2: ...: value}``
data body. These tests walk the emitted dict; the colorization / delta annotations
that used to be asserted on Markdown output are gone (consumers now compute deltas
from the raw values).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ddi.aggregate import (
    _row_model,
    aggregate,
    collect_table,
    iter_rows,
    load_rows,
)


def _write_row(
    store: Path,
    run_id: str,
    metric: str,
    value: float,
    seed: int = 0,
    model_revision: str = "test-model@abc",
) -> None:
    """Append one JSONL row in the same shape store.append writes."""
    with store.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "value": value,
                    "provenance": {"seed": seed, "model_revision": model_revision},
                }
            )
            + "\n"
        )


# ---- kind: quality --------------------------------------------------------------------------


def _seed_quality_store(store: Path) -> None:
    """One f1 row per (mask, dataset) so we hit every column."""
    for ds in (
        "synthetic_qa",
        "hotpot_qa",
        "musique",
        "kilt_nq",
    ):
        _write_row(store, f"phase2_train_free:Qwen:{ds}:full:seed=0:sha:r0", "f1", 0.8)
        _write_row(
            store, f"phase2_train_free:Qwen:{ds}:ddi_qf:seed=0:sha:r0", "f1", 0.79
        )
        _write_row(store, f"phase2_train_free:Qwen:{ds}:blind:seed=0:sha:r0", "f1", 0.1)


def test_collect_quality_kind_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_quality_store(store)
    spec = {"kind": "quality", "rows": "mask", "cols": "dataset", "metric": "f1"}
    body = collect_table(spec, store)

    assert body["kind"] == "quality"
    assert body["metric"] == "f1"
    assert body["models"] == ["Qwen"]
    # data body is {model: {mask: {dataset: value}}}.
    qwen = body["data"]["Qwen"]
    # Mask order: `full`, `ddi_qf`, `blind` per _QUALITY_PREFERRED_MASKS.
    assert list(qwen) == ["full", "ddi_qf", "blind"]
    # Every dataset column present, no dashes / nulls.
    for mask in ("full", "ddi_qf", "blind"):
        assert set(qwen[mask]) == {"synthetic_qa", "hotpot_qa", "musique", "kilt_nq"}
    # Raw numeric values, no delta annotations.
    assert qwen["full"]["hotpot_qa"] == pytest.approx(0.8)
    assert qwen["ddi_qf"]["hotpot_qa"] == pytest.approx(0.79)
    assert qwen["blind"]["synthetic_qa"] == pytest.approx(0.1)


def test_collect_quality_missing_cell_omits_key(tmp_path: Path) -> None:
    """A dataset with no rows for a given mask must be OMITTED (no null, no dash)."""
    store = tmp_path / "runs.jsonl"
    # Only `full` on hotpot_qa; `ddi_qf` on musique. Cross-cells absent.
    _write_row(store, "phase2_train_free:Qwen:hotpot_qa:full:seed=0:sha:r0", "f1", 0.7)
    _write_row(store, "phase2_train_free:Qwen:musique:ddi_qf:seed=0:sha:r0", "f1", 0.4)

    body = collect_table({"kind": "quality", "metric": "f1"}, store)
    qwen = body["data"]["Qwen"]

    assert "hotpot_qa" in qwen["full"]
    assert "musique" not in qwen["full"], "missing musique/full cell must be omitted"
    assert "musique" in qwen["ddi_qf"]
    assert "hotpot_qa" not in qwen["ddi_qf"]


def test_collect_quality_excludes_nq_and_stress_and_composition(tmp_path: Path) -> None:
    """Historical rows for pruned datasets are silently dropped."""
    store = tmp_path / "runs.jsonl"
    _write_row(store, "phase2_train_free:Qwen:nq:full:seed=0:sha:r0", "f1", 0.9)
    _write_row(
        store, "phase2_train_free:Qwen:composition:full:seed=0:sha:r0", "f1", 0.5
    )
    _write_row(
        store,
        "phase2_train_free:Qwen:ddi_stress__fact_chain:full:seed=0:sha:r0",
        "f1",
        0.55,
    )
    _write_row(store, "phase2_train_free:Qwen:hotpot_qa:full:seed=0:sha:r0", "f1", 0.7)

    body = collect_table({"kind": "quality", "metric": "f1"}, store)
    qwen_full = body["data"]["Qwen"]["full"]
    assert "nq" not in qwen_full
    assert "composition" not in qwen_full
    assert "ddi_stress__fact_chain" not in qwen_full
    assert qwen_full["hotpot_qa"] == pytest.approx(0.7)


def test_collect_quality_backing_rows_and_seeds(tmp_path: Path) -> None:
    """backing_rows counts every row that contributed; seeds is the sorted distinct set."""
    store = tmp_path / "runs.jsonl"
    _write_row(
        store, "phase2_train_free:Qwen:hotpot_qa:full:seed=0:sha:r0", "f1", 0.7, seed=0
    )
    _write_row(
        store, "phase2_train_free:Qwen:hotpot_qa:full:seed=1:sha:r0", "f1", 0.72, seed=1
    )
    _write_row(
        store, "phase2_train_free:Qwen:hotpot_qa:full:seed=2:sha:r0", "f1", 0.74, seed=2
    )

    body = collect_table({"kind": "quality", "metric": "f1"}, store)
    assert body["backing_rows"] == 3
    assert body["distinct_run_ids"] == 3
    assert body["seeds"] == [0, 1, 2]
    # Mean across seeds.
    assert body["data"]["Qwen"]["full"]["hotpot_qa"] == pytest.approx(0.72)


def test_collect_quality_rejects_unknown_metric(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_quality_store(store)
    with pytest.raises(ValueError, match="unknown metric"):
        collect_table({"kind": "quality", "metric": "bogus"}, store)


def test_collect_quality_empty_store_raises(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    with pytest.raises(ValueError, match="no phase2_train_free rows"):
        collect_table({"kind": "quality", "metric": "f1"}, store)


# ---- kind: timing ---------------------------------------------------------------------------


def _seed_timing_store(store: Path) -> None:
    """Seed one (L=512, sdpa) sub-block with full + ddi_qf at N=1..25."""
    for n in (1, 3, 5, 9, 13, 25):
        # `full` arm — TTFT scales roughly linearly with N.
        _write_row(
            store,
            f"phase3_timing:Qwen:N={n}:L=512:Q=1:full:sdpa:seed=0:sha",
            "ftt_median_ms",
            100.0 * n,
        )
        # `ddi_qf` — ~1.5× faster + analytical says n×.
        _write_row(
            store,
            f"phase3_timing:Qwen:N={n}:L=512:Q=1:ddi_qf:sdpa:seed=0:sha",
            "ftt_median_ms",
            100.0 * n / 1.5,
        )
        _write_row(
            store,
            f"phase3_timing:Qwen:N={n}:L=512:Q=1:ddi_qf:sdpa:seed=0:sha",
            "analytical_speedup",
            float(n),
        )
        _write_row(
            store,
            f"phase3_timing:Qwen:N={n}:L=512:Q=1:ddi_qf:sdpa:seed=0:sha",
            "mask_sparsity",
            (n - 1) / n if n > 0 else 0.0,
        )


def test_collect_timing_kind_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_timing_store(store)
    body = collect_table(
        {"kind": "timing", "metric": "realized"},
        store,
    )
    assert body["kind"] == "timing"
    assert body["metric"] == "realized"
    # Data shape: {model: {L: {backend: {mask: {N: value}}}}}.
    qwen = body["data"]["Qwen"]
    assert list(qwen) == ["512"]  # L keys stringified
    sdpa = qwen["512"]["sdpa"]
    assert list(sdpa) == ["full", "ddi_qf"]  # mask order
    # N keys are stringified ints, ordered per _TIMING_N_COLUMNS.
    assert list(sdpa["full"]) == ["1", "3", "5", "9", "13", "25"]
    # full is defined as realized=1.0 everywhere.
    assert sdpa["full"]["1"] == pytest.approx(1.0)
    assert sdpa["full"]["25"] == pytest.approx(1.0)
    # ddi_qf: realized = full_ttft / variant_ttft = 100n / (100n/1.5) = 1.5.
    assert sdpa["ddi_qf"]["3"] == pytest.approx(1.5)


def test_collect_timing_derives_realization_ratio(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_timing_store(store)
    body = collect_table({"kind": "timing", "metric": "realization"}, store)
    # At any N, analytical=N, realized=1.5 → realization = 1.5 / N.
    ddi_qf = body["data"]["Qwen"]["512"]["sdpa"]["ddi_qf"]
    assert ddi_qf["3"] == pytest.approx(0.5)
    assert ddi_qf["9"] == pytest.approx(1.5 / 9)


def test_collect_timing_ttft_raw_values(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_timing_store(store)
    body = collect_table({"kind": "timing", "metric": "ttft"}, store)
    # Raw milliseconds, no formatting.
    full = body["data"]["Qwen"]["512"]["sdpa"]["full"]
    assert full["1"] == pytest.approx(100.0)
    assert full["25"] == pytest.approx(2500.0)


def test_collect_timing_missing_metric_cell_omits_key(tmp_path: Path) -> None:
    """A mask with no ttft row at a given N must OMIT that N key entirely."""
    store = tmp_path / "runs.jsonl"
    # Only N=1 and N=5 for full — N=3 is a hole.
    for n in (1, 5):
        _write_row(
            store,
            f"phase3_timing:Qwen:N={n}:L=512:Q=1:full:sdpa:seed=0:sha",
            "ftt_median_ms",
            100.0 * n,
        )
    body = collect_table({"kind": "timing", "metric": "ttft"}, store)
    full = body["data"]["Qwen"]["512"]["sdpa"]["full"]
    assert "1" in full and "5" in full
    assert "3" not in full  # missing cell — key omitted, no null
    assert "25" not in full


def test_collect_timing_rejects_unknown_metric(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_timing_store(store)
    with pytest.raises(ValueError, match="unknown metric"):
        collect_table({"kind": "timing", "metric": "bogus"}, store)


# ---- kind: throughput -----------------------------------------------------------------------


def _seed_throughput_store(store: Path) -> None:
    for n, qps_cache, qps_recompute, qps_full in [
        (3, 12.0, 4.0, 2.5),
        (9, 20.0, 5.5, 3.0),
    ]:
        _write_row(
            store,
            f"phase4_throughput:Qwen:ddi_cache:restart:N={n}:reuse=zipf:seed=0:sha",
            "qps",
            qps_cache,
        )
        _write_row(
            store,
            f"phase4_throughput:Qwen:ddi_recompute:restart:N={n}:reuse=uniform:seed=0:sha",
            "qps",
            qps_recompute,
        )
        _write_row(
            store,
            f"phase4_throughput:Qwen:full_causal:restart:N={n}:reuse=uniform:seed=0:sha",
            "qps",
            qps_full,
        )


def test_collect_throughput_kind_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_throughput_store(store)
    body = collect_table({"kind": "throughput", "metric": "qps"}, store)
    assert body["kind"] == "throughput"
    assert body["metric"] == "qps"
    qwen = body["data"]["Qwen"]
    # Mode order follows _THROUGHPUT_PREFERRED_MODES.
    assert list(qwen) == ["ddi_cache", "ddi_recompute", "full_causal"]
    # N keys sorted numerically.
    assert list(qwen["ddi_cache"]) == ["3", "9"]
    assert qwen["ddi_cache"]["3"] == pytest.approx(12.0)
    assert qwen["ddi_cache"]["9"] == pytest.approx(20.0)
    assert qwen["full_causal"]["3"] == pytest.approx(2.5)


def test_collect_throughput_metric_defaults_to_qps(tmp_path: Path) -> None:
    """Spec without ``metric`` defaults to ``qps``."""
    store = tmp_path / "runs.jsonl"
    _seed_throughput_store(store)
    body = collect_table({"kind": "throughput"}, store)
    assert body["metric"] == "qps"


def test_collect_throughput_empty_store_raises(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    with pytest.raises(ValueError, match="no phase4_throughput rows"):
        collect_table({"kind": "throughput", "metric": "qps"}, store)


# ---- kind: correctness ----------------------------------------------------------------------


def _seed_correctness_store(store: Path) -> None:
    """Seed one (model, L=512) sub-block with full_causal / ddi_recompute / ddi_cache
    across N in {3, 9, 25}, two queries per doc so max/mean differ.
    """
    for n in (3, 9, 25):
        for q in (0, 1):
            _write_row(
                store,
                f"phase4_cache:Qwen:full_causal:N={n}:L=512:seed=0:sha:q{q}",
                "max_abs_logit_diff",
                0.0,
            )
            _write_row(
                store,
                f"phase4_cache:Qwen:ddi_recompute:N={n}:L=512:seed=0:sha:q{q}",
                "max_abs_logit_diff",
                1e-5 if q == 0 else 5e-5,
            )
            _write_row(
                store,
                f"phase4_cache:Qwen:ddi_cache:N={n}:L=512:seed=0:sha:q{q}",
                "max_abs_logit_diff",
                2e-5 if q == 0 else 8e-5,
            )


def test_collect_correctness_kind_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    body = collect_table({"kind": "correctness"}, store)
    assert body["kind"] == "correctness"
    assert body["metric"] == "max_abs_logit_diff"
    qwen = body["data"]["Qwen"]
    # Mode order: full_causal first (baseline), then remaining modes alphabetically.
    assert list(qwen) == ["full_causal", "ddi_cache", "ddi_recompute"]
    # Under each mode: {L: {N: value}} — L stringified, N stringified, both sorted.
    assert list(qwen["ddi_recompute"]) == ["512"]
    assert list(qwen["ddi_recompute"]["512"]) == ["3", "9", "25"]


def test_collect_correctness_uses_max_across_queries(tmp_path: Path) -> None:
    """The cell aggregator must take max across queries (worst-case gate, not mean)."""
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    body = collect_table({"kind": "correctness"}, store)
    ddi_recompute_512 = body["data"]["Qwen"]["ddi_recompute"]["512"]
    # Q0=1e-5, Q1=5e-5 → max = 5e-5.
    assert ddi_recompute_512["3"] == pytest.approx(5e-5)
    # Mean would be 3e-5 — must NOT match.
    assert ddi_recompute_512["3"] != pytest.approx(3e-5)


def test_collect_correctness_full_causal_zero(tmp_path: Path) -> None:
    """`full_causal` rows are recorded verbatim (0.0 from the store)."""
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    body = collect_table({"kind": "correctness"}, store)
    for n in ("3", "9", "25"):
        assert body["data"]["Qwen"]["full_causal"]["512"][n] == pytest.approx(0.0)


def test_collect_correctness_multiple_ls(tmp_path: Path) -> None:
    """Different L values each get their own key under the mode."""
    store = tmp_path / "runs.jsonl"
    _write_row(
        store,
        "phase4_cache:Qwen:full_causal:N=3:L=256:seed=0:sha:q0",
        "max_abs_logit_diff",
        0.0,
    )
    _write_row(
        store,
        "phase4_cache:Qwen:full_causal:N=3:L=512:seed=0:sha:q0",
        "max_abs_logit_diff",
        0.0,
    )
    body = collect_table({"kind": "correctness"}, store)
    fc = body["data"]["Qwen"]["full_causal"]
    assert set(fc) == {"256", "512"}


def test_collect_correctness_empty_store_raises(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    with pytest.raises(ValueError, match="no phase4_cache rows"):
        collect_table({"kind": "correctness"}, store)


# ---- kind: restart_quality ------------------------------------------------------------------


def _seed_restart_quality_store(store: Path) -> None:
    for ds in ("synthetic_qa", "hotpot_qa"):
        _write_row(
            store,
            f"phase4_restart:Qwen:{ds}:tag:sequential:seed=0:sha:r0",
            "f1",
            0.75,
        )
        _write_row(
            store,
            f"phase4_restart:Qwen:{ds}:tag:restart:seed=0:sha:r0",
            "f1",
            0.74,
        )


def test_collect_restart_quality_kind_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_restart_quality_store(store)
    body = collect_table({"kind": "restart_quality", "metric": "f1"}, store)
    assert body["kind"] == "restart_quality"
    qwen = body["data"]["Qwen"]
    # Layout order: sequential first (baseline), then restart.
    assert list(qwen) == ["sequential", "restart"]
    assert qwen["sequential"]["synthetic_qa"] == pytest.approx(0.75)
    assert qwen["restart"]["hotpot_qa"] == pytest.approx(0.74)


def test_collect_restart_quality_empty_store_raises(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    with pytest.raises(ValueError, match="no phase4_restart rows"):
        collect_table({"kind": "restart_quality", "metric": "f1"}, store)


# ---- Dispatcher errors ----------------------------------------------------------------------


def test_collect_table_rejects_unknown_kind(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    with pytest.raises(ValueError, match="unknown table kind"):
        collect_table({"kind": "not_a_kind"}, store)


# ---- Model filter ---------------------------------------------------------------------------


def test_model_filter_narrows_quality(tmp_path: Path) -> None:
    """model_filter drops rows whose run_id doesn't name the requested model."""
    store = tmp_path / "runs.jsonl"
    _write_row(
        store, "phase2_train_free:ModelA:hotpot_qa:full:seed=0:sha:r0", "f1", 0.7
    )
    _write_row(
        store, "phase2_train_free:ModelB:hotpot_qa:full:seed=0:sha:r0", "f1", 0.5
    )
    body = collect_table(
        {"kind": "quality", "metric": "f1"}, store, model_filter="ModelA"
    )
    assert body["models"] == ["ModelA"]
    assert "ModelB" not in body["data"]
    assert body["data"]["ModelA"]["full"]["hotpot_qa"] == pytest.approx(0.7)


# ---- End-to-end: aggregate() writes an envelope with tables[] --------------------------------


def test_aggregate_composes_envelope(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_quality_store(store)
    _seed_throughput_store(store)

    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        yaml.safe_dump(
            {
                "kind": "quality",
                "model": "Qwen",
                "report": {
                    "tables": [
                        {"kind": "quality", "metric": "f1"},
                        {"kind": "throughput", "metric": "qps"},
                    ]
                },
            }
        )
    )
    out = tmp_path / "summary.json"
    aggregate(exp_yaml, out, store=store)

    envelope = json.loads(out.read_text())
    # Envelope fields present.
    assert envelope["yaml"] == str(exp_yaml)
    assert envelope["store"] == str(store)
    assert envelope["model_filter"] is None
    assert isinstance(envelope["generated_utc"], str)
    assert envelope["generated_utc"].endswith("Z")
    assert isinstance(envelope["git_sha"], str)
    # store_sha256_8 is 8 hex chars (or "missing" if the store file were absent — it isn't).
    assert len(envelope["store_sha256_8"]) == 8
    # Two tables in the order declared in the YAML.
    kinds = [t["kind"] for t in envelope["tables"]]
    assert kinds == ["quality", "throughput"]


def test_aggregate_missing_store_sets_store_sha256_missing(tmp_path: Path) -> None:
    """When the store file is absent, envelope carries store_sha256_8 = "missing"."""
    store = tmp_path / "runs.jsonl"
    _seed_quality_store(store)  # write, so tables render
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        yaml.safe_dump(
            {"report": {"tables": [{"kind": "quality", "metric": "f1"}]}}
        )
    )
    # Point the envelope at a non-existent path for the hash step — collect_table itself
    # takes the same argument so we need a workable store for collection. Emulate the
    # missing-file branch on the envelope helper directly.
    out = tmp_path / "summary.json"
    aggregate(exp_yaml, out, store=store)
    envelope = json.loads(out.read_text())
    # Not the "missing" branch here — sanity that the happy path still writes 8 hex.
    assert len(envelope["store_sha256_8"]) == 8

    from ddi.aggregate import _store_sha256_8
    assert _store_sha256_8(tmp_path / "no-such-file.jsonl") == "missing"


def test_aggregate_model_filter_echoed_in_envelope(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _write_row(
        store, "phase2_train_free:ModelA:hotpot_qa:full:seed=0:sha:r0", "f1", 0.7
    )
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        yaml.safe_dump(
            {"report": {"tables": [{"kind": "quality", "metric": "f1"}]}}
        )
    )
    out = tmp_path / "summary.json"
    aggregate(exp_yaml, out, store=store, model_filter="ModelA")
    envelope = json.loads(out.read_text())
    assert envelope["model_filter"] == "ModelA"


def test_aggregate_overwrites_out(tmp_path: Path) -> None:
    """Each aggregate() run overwrites --out fully (no --append mode)."""
    store = tmp_path / "runs.jsonl"
    _seed_throughput_store(store)
    out = tmp_path / "summary.json"
    out.write_text("garbage that should be replaced")
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        yaml.safe_dump(
            {"report": {"tables": [{"kind": "throughput", "metric": "qps"}]}}
        )
    )
    aggregate(exp_yaml, out, store=store)
    envelope = json.loads(out.read_text())
    assert envelope["tables"][0]["kind"] == "throughput"


def test_aggregate_rejects_unknown_kind(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        yaml.safe_dump({"report": {"tables": [{"kind": "not_a_kind"}]}})
    )
    with pytest.raises(ValueError, match="unknown table kind"):
        aggregate(exp_yaml, tmp_path / "summary.json", store=store)


def test_aggregate_rejects_empty_tables(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(yaml.safe_dump({"report": {"tables": []}}))
    with pytest.raises(ValueError, match="report.tables is empty"):
        aggregate(exp_yaml, tmp_path / "summary.json", store=store)


def test_aggregate_cli_has_no_append_flag() -> None:
    """The --append flag was dropped in the JSON refactor."""
    from ddi.aggregate import _build_parser
    parser = _build_parser()
    # Flatten every option string across every action.
    all_options = [opt for a in parser._actions for opt in a.option_strings]
    assert "--append" not in all_options


# ---- load_rows / _row_model preserved from #115 ---------------------------------------------


def test_load_rows_no_filter_returns_all(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _write_row(store, "phase2_train_free:A:hotpot_qa:full:seed=0:sha:r0", "f1", 0.1)
    _write_row(store, "phase2_train_free:B:hotpot_qa:full:seed=0:sha:r0", "f1", 0.2)
    rows = load_rows(store)
    assert len(rows) == 2


def test_load_rows_model_filter_narrows(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _write_row(store, "phase2_train_free:A:hotpot_qa:full:seed=0:sha:r0", "f1", 0.1)
    _write_row(store, "phase2_train_free:B:hotpot_qa:full:seed=0:sha:r0", "f1", 0.2)
    rows = load_rows(store, model_filter="A")
    assert len(rows) == 1
    assert _row_model(rows[0]) == "A"


def test_load_rows_model_filter_drops_shapeless_rows(tmp_path: Path) -> None:
    """Rows whose run_id doesn't match ``<prefix>:<model>:...`` are dropped by any filter."""
    store = tmp_path / "runs.jsonl"
    _write_row(store, "phase2_train_free:A:hotpot_qa:full:seed=0:sha:r0", "f1", 0.1)
    # Malformed run_id with no colon at all — _row_model returns None.
    _write_row(store, "no-colon-run-id", "f1", 0.9)
    rows = load_rows(store, model_filter="A")
    # Only the one shapely row for A.
    assert len(rows) == 1


def test_row_model_extracts_second_segment() -> None:
    assert _row_model({"run_id": "phase2_train_free:Qwen:foo:bar"}) == "Qwen"
    assert _row_model({"run_id": "phase3_timing:Llama-3.1:N=1:L=512"}) == "Llama-3.1"


def test_row_model_returns_none_for_bad_shape() -> None:
    assert _row_model({"run_id": ""}) is None
    assert _row_model({"run_id": "single-segment"}) is None
    assert _row_model({"run_id": ":empty-prefix:xx"}) is None
    assert _row_model({}) is None


def test_iter_rows_skips_blank_lines(tmp_path: Path) -> None:
    """iter_rows is tolerant of blank lines in the JSONL."""
    store = tmp_path / "runs.jsonl"
    store.write_text(
        '{"run_id": "a:x", "metric": "f1", "value": 0.1}\n'
        '\n'
        '{"run_id": "b:y", "metric": "f1", "value": 0.2}\n'
    )
    rows = list(iter_rows(store))
    assert len(rows) == 2
