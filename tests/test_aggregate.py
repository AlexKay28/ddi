"""Tests for ddi/aggregate.py — YAML-driven table builder (issue #86)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ddi.aggregate import _render_cell, aggregate, render_table
from ddi.reports._provenance import FOOTER_MARKER


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


def test_render_quality_kind_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_quality_store(store)
    spec = {"kind": "quality", "rows": "mask", "cols": "dataset", "metric": "f1"}
    out = render_table(spec, store)

    # Header carries the fixed dataset column order from #72 (pruned in #105 to the
    # four live QA datasets — composition / ddi_stress__* are gone).
    header = next(line for line in out.splitlines() if line.startswith("| mask |"))
    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    assert cols == [
        "mask",
        "synthetic_qa",
        "hotpot_qa",
        "musique",
        "kilt_nq",
    ]
    # `full` row is the baseline (no Δ annotation).
    full_row = next(line for line in out.splitlines() if line.startswith("| full |"))
    assert "Δ" not in full_row
    # `ddi_qf` row shows Δ vs full and (0.79 vs 0.80 = -1.25%) qualifies for 🟢.
    qf_row = next(line for line in out.splitlines() if line.startswith("| ddi_qf |"))
    assert "Δ−1.3%" in qf_row  # .1f formatting rounds 1.25 → 1.3 (banker's rounding)
    assert "🟢" in qf_row
    # `blind` row never gets 🟢 (ground-rule-7 collision).
    blind_row = next(line for line in out.splitlines() if line.startswith("| blind |"))
    assert "🟢" not in blind_row


def test_quality_excludes_nq_column(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _write_row(store, "phase2_train_free:Qwen:nq:full:seed=0:sha:r0", "f1", 0.9)
    _write_row(store, "phase2_train_free:Qwen:synthetic_qa:full:seed=0:sha:r0", "f1", 0.3)
    out = render_table(
        {"kind": "quality", "rows": "mask", "cols": "dataset", "metric": "f1"}, store
    )
    header = next(line for line in out.splitlines() if line.startswith("| mask |"))
    assert " nq " not in header
    assert "synthetic_qa" in header


def test_quality_drops_stale_stress_and_composition_rows(tmp_path: Path) -> None:
    """Post-#89 the composition / ddi_stress loaders are gone; historical rows must be
    silently dropped rather than blowing up the renderer or leaking into a column."""
    store = tmp_path / "runs.jsonl"
    # Stale rows that used to feed removed columns.
    _write_row(store, "phase2_train_free:Qwen:composition:full:seed=0:sha:r0", "f1", 0.5)
    _write_row(
        store,
        "phase2_train_free:Qwen:ddi_stress__fact_chain:full:seed=0:sha:r0",
        "f1",
        0.55,
    )
    # A live row so the renderer has something to emit (else it raises).
    _write_row(store, "phase2_train_free:Qwen:hotpot_qa:full:seed=0:sha:r0", "f1", 0.7)

    out = render_table(
        {"kind": "quality", "rows": "mask", "cols": "dataset", "metric": "f1"}, store
    )
    header = next(line for line in out.splitlines() if line.startswith("| mask |"))
    assert "composition" not in header
    assert "ds/fact_chain" not in header
    assert "hotpot_qa" in header


# ---- kind: timing ---------------------------------------------------------------------------


def _seed_timing_store(store: Path) -> None:
    """Seed one (L=512, sdpa) sub-block with full + ddi_qf + blind at N=1..25."""
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


def test_render_timing_kind_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_timing_store(store)
    spec = {
        "kind": "timing",
        "rows": "mask",
        "cols": "N",
        "metric": "realized",
        "split_by": ["L", "backend"],
    }
    out = render_table(spec, store)

    # Header advertises the metric and one sub-block per (model, L, backend).
    assert "timing — realized speedup" in out
    assert "Qwen" in out
    assert "L=512" in out
    assert "backend=sdpa" in out
    # Header columns fixed to N=1..25.
    header = next(line for line in out.splitlines() if line.startswith("| mask |"))
    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    assert cols == ["mask", "N=1", "N=3", "N=5", "N=9", "N=13", "N=25"]
    # `full` row: realized ≡ 1.0×.
    full_row = next(line for line in out.splitlines() if line.startswith("| full |"))
    assert "1.00x" in full_row
    # `ddi_qf` row: realized = full/variant TTFT = 1.5×.
    qf_row = next(line for line in out.splitlines() if line.startswith("| ddi_qf |"))
    assert "1.50x" in qf_row


def test_timing_derives_realization_ratio(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_timing_store(store)
    spec = {
        "kind": "timing",
        "rows": "mask",
        "cols": "N",
        "metric": "realization",
        "split_by": ["L", "backend"],
    }
    out = render_table(spec, store)
    # At N=3, analytical=3.0, realized=1.5 → realization = 0.500.
    qf_row = next(line for line in out.splitlines() if line.startswith("| ddi_qf |"))
    assert "0.500" in qf_row


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


def test_render_throughput_kind_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_throughput_store(store)
    spec = {"kind": "throughput", "rows": "mode", "cols": "N", "metric": "qps"}
    out = render_table(spec, store)
    header = next(line for line in out.splitlines() if line.startswith("| mode |"))
    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    assert cols == ["mode", "N=3", "N=9"]
    cache_row = next(
        line for line in out.splitlines() if line.startswith("| ddi_cache |")
    )
    cells = [c.strip() for c in cache_row.strip().strip("|").split("|")]
    assert cells == ["ddi_cache", "12.0000", "20.0000"]
    # Row order follows _THROUGHPUT_PREFERRED_MODES.
    row_labels = [
        line.split("|")[1].strip()
        for line in out.splitlines()
        if line.startswith("|") and "N=" not in line and "---" not in line
    ]
    # Skip the header row label ("mode") — keep only data rows.
    data_labels = [lbl for lbl in row_labels if lbl != "mode"]
    assert data_labels == ["ddi_cache", "ddi_recompute", "full_causal"]


# ---- kind: correctness ----------------------------------------------------------------------


def _seed_correctness_store(store: Path) -> None:
    """Seed one (model, L=512) sub-block with full_causal / ddi_recompute / ddi_cache
    across N in {3, 9, 25}, with two queries per doc so max/mean differ.
    """
    for n in (3, 9, 25):
        for q in (0, 1):
            # `full_causal` is the baseline — always 0.0 per the runner contract.
            _write_row(
                store,
                f"phase4_cache:Qwen:full_causal:N={n}:L=512:seed=0:sha:q{q}",
                "max_abs_logit_diff",
                0.0,
            )
            # `ddi_recompute` — small non-zero diff; second query is worse so max() must win.
            _write_row(
                store,
                f"phase4_cache:Qwen:ddi_recompute:N={n}:L=512:seed=0:sha:q{q}",
                "max_abs_logit_diff",
                1e-5 if q == 0 else 5e-5,
            )
            # `ddi_cache` — a hair larger than recompute (cache round-trip noise).
            _write_row(
                store,
                f"phase4_cache:Qwen:ddi_cache:N={n}:L=512:seed=0:sha:q{q}",
                "max_abs_logit_diff",
                2e-5 if q == 0 else 8e-5,
            )


def test_render_correctness_kind_shape(tmp_path: Path) -> None:
    """kind: correctness renders one sub-table per (model, L) with mode × N cells."""
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    spec = {
        "kind": "correctness",
        "rows": "mode",
        "cols": "N",
        "metric": "max_abs_logit_diff",
    }
    out = render_table(spec, store)

    assert "correctness — max_abs_logit_diff" in out
    assert "Qwen — L=512" in out
    header = next(line for line in out.splitlines() if line.startswith("| mode |"))
    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    assert cols == ["mode", "N=3", "N=9", "N=25"]


def test_render_correctness_full_causal_is_baseline_zero(tmp_path: Path) -> None:
    """`full_causal` row always renders 0.00e+00 (baseline by definition)."""
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    out = render_table({"kind": "correctness", "metric": "max_abs_logit_diff"}, store)
    full_row = next(line for line in out.splitlines() if line.startswith("| full_causal |"))
    cells = [c.strip() for c in full_row.strip().strip("|").split("|")]
    # cells[0] == "full_causal"; every value cell must be exact zero in scientific form.
    for cell in cells[1:]:
        assert cell == "0.00e+00", f"baseline cell not zero: {cell!r}"


def test_render_correctness_cell_uses_max_across_queries(tmp_path: Path) -> None:
    """The cell aggregator must take max across queries (worst-case gate, not mean)."""
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    out = render_table({"kind": "correctness", "metric": "max_abs_logit_diff"}, store)
    recompute_row = next(
        line for line in out.splitlines() if line.startswith("| ddi_recompute |")
    )
    # Q0=1e-5, Q1=5e-5 → max = 5e-5 → formatted "5.00e-05".
    assert "5.00e-05" in recompute_row
    # If the renderer took the mean, the value would be 3.00e-05 — must NOT appear.
    assert "3.00e-05" not in recompute_row


def test_render_correctness_row_order_full_causal_first(tmp_path: Path) -> None:
    """Row order: full_causal first (baseline), then remaining modes alphabetically."""
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    out = render_table({"kind": "correctness", "metric": "max_abs_logit_diff"}, store)
    mode_labels = [
        line.split("|")[1].strip()
        for line in out.splitlines()
        if line.startswith("|") and "N=" not in line and "---" not in line
    ]
    data_labels = [lbl for lbl in mode_labels if lbl != "mode"]
    # `ddi_cache` < `ddi_recompute` alphabetically → cache comes before recompute.
    assert data_labels == ["full_causal", "ddi_cache", "ddi_recompute"]


def test_render_correctness_scientific_notation(tmp_path: Path) -> None:
    """Cells render as scientific notation (`{value:.2e}`) — bf16 diffs are O(1e-4)."""
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    out = render_table({"kind": "correctness", "metric": "max_abs_logit_diff"}, store)
    # Some cell in the body has scientific-notation form with lowercase `e`.
    body_rows = [
        line for line in out.splitlines() if line.startswith("| ") and "e-" in line
    ]
    assert body_rows, "no scientific-notation cell found in correctness table"


def test_render_correctness_appends_provenance_footer(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_correctness_store(store)
    out = render_table({"kind": "correctness", "metric": "max_abs_logit_diff"}, store)
    assert FOOTER_MARKER in out
    assert out.count(FOOTER_MARKER) == 1


def test_render_correctness_multiple_ls_emits_sub_blocks(tmp_path: Path) -> None:
    """One sub-table per (model, L) — different L values must each get their own block."""
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
    out = render_table({"kind": "correctness", "metric": "max_abs_logit_diff"}, store)
    assert "Qwen — L=256" in out
    assert "Qwen — L=512" in out


def test_render_correctness_empty_store_raises(tmp_path: Path) -> None:
    """No phase4_cache rows in the store → loud error, not empty output."""
    store = tmp_path / "runs.jsonl"
    store.touch()
    with pytest.raises(ValueError, match="no phase4_cache rows"):
        render_table({"kind": "correctness", "metric": "max_abs_logit_diff"}, store)


# ---- Colorization contract ------------------------------------------------------------------


def test_color_floor_vs_floor_no_green() -> None:
    """Both values below 0.01 → no 🟢 (baseline is noise)."""
    cell = _render_cell(0.003, 0.005)
    assert "🟢" not in cell
    # Percentage would explode, so the small-baseline branch shows absolute Δ.
    assert "Δ" in cell


def test_color_loss_ceiling_no_green() -> None:
    """Loss above the ceiling on both arms → no 🟢 (uniform-over-vocab, not parity)."""
    cell = _render_cell(15.0, 15.1, is_loss=True)
    assert "🟢" not in cell


def test_color_real_parity_gets_green() -> None:
    """Real signal on both arms with |Δ| < 5% of full → 🟢."""
    cell = _render_cell(0.79, 0.80)
    assert "🟢" in cell
    assert "Δ" in cell


def test_color_blind_row_disallows_green() -> None:
    """allow_green=False (used on the blind row) suppresses 🟢 even at parity."""
    cell = _render_cell(0.79, 0.80, allow_green=False)
    assert "🟢" not in cell


def test_color_over_threshold_no_green() -> None:
    """|Δ| ≥ 5% of full → no 🟢."""
    cell = _render_cell(0.7, 0.8)
    assert "🟢" not in cell


# ---- Provenance footer ----------------------------------------------------------------------


def test_render_table_appends_provenance_footer(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_quality_store(store)
    out = render_table(
        {"kind": "quality", "rows": "mask", "cols": "dataset", "metric": "f1"}, store
    )
    assert FOOTER_MARKER in out
    # Footer mentions the store path and a store-hash prefix ("sha256 `...`").
    assert "sha256" in out


def test_all_three_kinds_include_footer(tmp_path: Path) -> None:
    """Every kind must emit exactly one footer marker per table."""
    store = tmp_path / "runs.jsonl"
    _seed_quality_store(store)
    _seed_timing_store(store)
    _seed_throughput_store(store)
    for spec in [
        {"kind": "quality", "rows": "mask", "cols": "dataset", "metric": "f1"},
        {"kind": "timing", "rows": "mask", "cols": "N", "metric": "ttft"},
        {"kind": "throughput", "rows": "mode", "cols": "N", "metric": "qps"},
    ]:
        out = render_table(spec, store)
        assert out.count(FOOTER_MARKER) == 1, f"kind {spec['kind']} missing exactly one footer"


# ---- End-to-end: aggregate() composes two tables --------------------------------------------


def test_aggregate_composes_two_tables(tmp_path: Path) -> None:
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
                        {
                            "kind": "quality",
                            "rows": "mask",
                            "cols": "dataset",
                            "metric": "f1",
                        },
                        {
                            "kind": "throughput",
                            "rows": "mode",
                            "cols": "N",
                            "metric": "qps",
                        },
                    ]
                },
            }
        )
    )
    out = tmp_path / "summary.md"
    aggregate(exp_yaml, out, store=store)

    text = out.read_text()
    # Both tables present.
    assert "### Qwen — F1" in text
    assert "throughput — qps" in text
    # Exactly one footer per table.
    assert text.count(FOOTER_MARKER) == 2


def test_aggregate_append_mode_preserves_prior_content(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_throughput_store(store)
    out = tmp_path / "summary.md"
    out.write_text("# preface\n")
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        yaml.safe_dump(
            {
                "report": {
                    "tables": [
                        {
                            "kind": "throughput",
                            "rows": "mode",
                            "cols": "N",
                            "metric": "qps",
                        }
                    ]
                }
            }
        )
    )
    aggregate(exp_yaml, out, store=store, append=True)
    text = out.read_text()
    assert text.startswith("# preface\n")
    assert "throughput — qps" in text


def test_aggregate_rejects_unknown_kind(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        yaml.safe_dump({"report": {"tables": [{"kind": "not_a_kind"}]}})
    )
    with pytest.raises(ValueError, match="unknown table kind"):
        aggregate(exp_yaml, tmp_path / "summary.md", store=store)


def test_aggregate_rejects_empty_tables(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(yaml.safe_dump({"report": {"tables": []}}))
    with pytest.raises(ValueError, match="report.tables is empty"):
        aggregate(exp_yaml, tmp_path / "summary.md", store=store)
