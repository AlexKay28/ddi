"""Tests for ddi/reports/phase3_realization.py — T3 table and F3 figure renderers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.report import list_reports, render


def _write_row(store: Path, run_id: str, metric: str, value: float | str, seed: int = 0) -> None:
    with store.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "value": value,
                    "provenance": {"seed": seed},
                }
            )
            + "\n"
        )


def _seed_cell(
    store: Path,
    *,
    n: int,
    l: int,
    q: int,
    variant_tag: str,
    ftt_ms: float,
    analytical: float | None = None,
    sparsity: float | None = None,
    backend: str = "sdpa",
    seed: int = 0,
    model: str = "Qwen/Qwen2.5-7B",
) -> None:
    rid = (
        f"phase3_timing:{model}:N={n}:L={l}:Q={q}:{variant_tag}:{backend}:"
        f"seed={seed}:abcd1234"
    )
    _write_row(store, rid, "ftt_median_ms", ftt_ms, seed)
    if analytical is not None:
        _write_row(store, rid, "analytical_speedup", analytical, seed)
    if sparsity is not None:
        _write_row(store, rid, "mask_sparsity", sparsity, seed)


def test_t3_registered_and_f3_registered():
    reports = list_reports()
    assert "T3" in reports
    assert "F3" in reports


def test_t3_empty_store_raises_when_no_phase3_rows(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    store.touch()
    _write_row(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    with pytest.raises(ValueError, match="phase3_timing"):
        render("T3", store=store)


def test_t3_renders_markdown_with_expected_columns(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # Two Ns; each has full + ddi_qf ftt across two seeds. Realized = full / ddi_qf.
    for seed in (0, 1):
        _seed_cell(
            store, n=1, l=512, q=64, variant_tag="full", ftt_ms=100.0, seed=seed
        )
        _seed_cell(
            store,
            n=1,
            l=512,
            q=64,
            variant_tag="ddi_qf",
            ftt_ms=100.0,
            analytical=1.0,
            sparsity=0.0,
            seed=seed,
        )
        _seed_cell(
            store, n=13, l=512, q=64, variant_tag="full", ftt_ms=720.0, seed=seed
        )
        _seed_cell(
            store,
            n=13,
            l=512,
            q=64,
            variant_tag="ddi_qf",
            ftt_ms=600.0,
            analytical=10.59,
            sparsity=0.906,
            seed=seed,
        )

    out = render("T3", store=store)
    assert isinstance(out, str)
    # Header shape from RESEARCH_PLAN §7.
    assert (
        "| N | analytical | realized (median) | realization ratio | sparsity |" in out
    )
    # N=1 row: analytical=1.00x, realized=1.00x, ratio=1.000
    assert "| 1 | 1.00x | 1.00x | 1.000 | 0.000 |" in out
    # N=13 row: analytical=10.59x, realized=720/600=1.20x, ratio=1.20/10.59=0.113
    assert "| 13 | 10.59x | 1.20x | 0.113 | 0.906 |" in out


def test_t3_renders_na_for_missing_arm(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # Only a `full` arm at N=5 — no ddi_qf to pair with.
    _seed_cell(store, n=5, l=512, q=64, variant_tag="full", ftt_ms=200.0)
    # analytical alone (no ftt) still creates a row for N=9.
    _seed_cell(
        store,
        n=9,
        l=512,
        q=64,
        variant_tag="ddi_qf",
        ftt_ms=100.0,
        analytical=7.39,
        sparsity=0.865,
    )
    out = render("T3", store=store)
    # N=5: realized missing => n/a for realized and ratio, sparsity also n/a
    assert "| 5 |" in out
    assert "n/a" in out


def test_t3_is_idempotent(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    for seed in (0, 1, 2):
        _seed_cell(
            store, n=3, l=512, q=64, variant_tag="full", ftt_ms=100.0 + seed, seed=seed
        )
        _seed_cell(
            store,
            n=3,
            l=512,
            q=64,
            variant_tag="ddi_qf",
            ftt_ms=50.0 + seed,
            analytical=2.59,
            sparsity=0.614,
            seed=seed,
        )
    a = render("T3", store=store)
    b = render("T3", store=store)
    assert a == b


def test_f3_renders_pdf(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    for l in (128, 512):
        for n in (1, 5, 13):
            for seed in (0, 1):
                _seed_cell(
                    store,
                    n=n,
                    l=l,
                    q=64,
                    variant_tag="full",
                    ftt_ms=100.0 * n,
                    seed=seed,
                )
                _seed_cell(
                    store,
                    n=n,
                    l=l,
                    q=64,
                    variant_tag="ddi_qf",
                    ftt_ms=100.0 * n / 2.0,
                    analytical=float(n),
                    sparsity=0.5,
                    seed=seed,
                )

    out = tmp_path / "F3.pdf"
    render("F3", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f3_empty_store_raises(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    store = tmp_path / "runs.jsonl"
    store.touch()
    out = tmp_path / "F3.pdf"
    with pytest.raises(ValueError, match="phase3_timing"):
        render("F3", store=store, output=out)


# ---- T_perf_* renderers (EXP-3, #73) -------------------------------------------------------


_T_PERF_NAMES = (
    "T_perf_ttft",
    "T_perf_tpot",
    "T_perf_sparsity",
    "T_perf_analytical",
    "T_perf_realized",
    "T_perf_realization",
)


def test_t_perf_renderers_registered():
    reports = list_reports()
    for name in _T_PERF_NAMES:
        assert name in reports, f"{name} not registered"


def _perf_cell(
    store: Path,
    *,
    n: int,
    l: int,
    variant_tag: str,
    ttft_ms: float,
    analytical: float | None = None,
    sparsity: float | None = None,
    tpot_ms: float | None = None,
    backend: str = "sdpa",
    seed: int = 0,
    model: str = "Qwen/Qwen2.5-7B",
) -> None:
    rid = (
        f"phase3_timing:{model}:N={n}:L={l}:Q=64:{variant_tag}:{backend}:"
        f"seed={seed}:abcd1234"
    )
    _write_row(store, rid, "ftt_median_ms", ttft_ms, seed)
    if tpot_ms is not None:
        _write_row(store, rid, "tpot_median_ms", tpot_ms, seed)
    if analytical is not None:
        _write_row(store, rid, "analytical_speedup", analytical, seed)
    if sparsity is not None:
        _write_row(store, rid, "mask_sparsity", sparsity, seed)


def test_t_perf_ttft_n_column_order(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    for n in (1, 3, 5, 9, 13, 25):
        _perf_cell(store, n=n, l=512, variant_tag="full", ttft_ms=100.0 * n)
    out = render("T_perf_ttft", store=store)
    header = next(line for line in out.splitlines() if line.startswith("| mask |"))
    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    assert cols == ["mask", "N=1", "N=3", "N=5", "N=9", "N=13", "N=25"]


def test_t_perf_realized_speedup_derived_from_full_pairing(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # full arm at N=13, L=512, sdpa, seed=0 -> 720 ms
    _perf_cell(store, n=13, l=512, variant_tag="full", ttft_ms=720.0)
    # ddi_qf arm at same shape -> 600 ms; analytical 10.59x
    _perf_cell(
        store,
        n=13,
        l=512,
        variant_tag="ddi_qf",
        ttft_ms=600.0,
        analytical=10.59,
        sparsity=0.906,
    )
    out = render("T_perf_realized", store=store)
    # realized for ddi_qf = 720/600 = 1.20x, full row = 1.00x
    ddi_row = next(line for line in out.splitlines() if line.startswith("| ddi_qf |"))
    cells = [c.strip() for c in ddi_row.strip().strip("|").split("|")]
    # N columns are 1, 3, 5, 9, 13, 25 -> N=13 is index 5
    assert cells[5] == "1.20x"

    full_row = next(line for line in out.splitlines() if line.startswith("| full |"))
    full_cells = [c.strip() for c in full_row.strip().strip("|").split("|")]
    assert full_cells[5] == "1.00x"


def test_t_perf_realization_ratio_derived(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    _perf_cell(store, n=13, l=512, variant_tag="full", ttft_ms=720.0)
    _perf_cell(
        store,
        n=13,
        l=512,
        variant_tag="ddi_qf",
        ttft_ms=600.0,
        analytical=10.59,
    )
    out = render("T_perf_realization", store=store)
    ddi_row = next(line for line in out.splitlines() if line.startswith("| ddi_qf |"))
    cells = [c.strip() for c in ddi_row.strip().strip("|").split("|")]
    # ratio = (720/600) / 10.59 = 1.20 / 10.59 = 0.113
    assert cells[5] == "0.113"


def test_t_perf_missing_full_pair_renders_em_dash_for_derived(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # No `full` arm at N=13 — realized_speedup for ddi_qf must be em-dash.
    _perf_cell(
        store,
        n=13,
        l=512,
        variant_tag="ddi_qf",
        ttft_ms=600.0,
        analytical=10.59,
        sparsity=0.906,
    )
    out_realized = render("T_perf_realized", store=store)
    ddi_row = next(
        line for line in out_realized.splitlines() if line.startswith("| ddi_qf |")
    )
    cells = [c.strip() for c in ddi_row.strip().strip("|").split("|")]
    assert cells[5] == "—"

    # But ttft (a row-based, non-derived metric) has a real value.
    out_ttft = render("T_perf_ttft", store=store)
    ddi_row = next(
        line for line in out_ttft.splitlines() if line.startswith("| ddi_qf |")
    )
    cells = [c.strip() for c in ddi_row.strip().strip("|").split("|")]
    assert cells[5] == "600.00"


def test_t_perf_multiple_l_backend_sub_blocks(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    for l in (128, 512, 2048):
        for backend in ("sdpa", "fa2"):
            _perf_cell(
                store, n=13, l=l, variant_tag="full", ttft_ms=100.0, backend=backend
            )
    out = render("T_perf_ttft", store=store)
    # 3 Ls x 2 backends = 6 sub-blocks
    subheadings = [line for line in out.splitlines() if line.startswith("## ")]
    assert len(subheadings) == 6
    for l in (128, 512, 2048):
        for backend in ("sdpa", "fa2"):
            assert f"L={l}, backend={backend}" in out


def test_t_perf_ttft_raises_when_empty(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    store.touch()
    with pytest.raises(ValueError, match="phase3_timing"):
        render("T_perf_ttft", store=store)


def test_t_perf_full_sparsity_is_zero(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    _perf_cell(store, n=13, l=512, variant_tag="full", ttft_ms=100.0)
    out = render("T_perf_sparsity", store=store)
    full_row = next(line for line in out.splitlines() if line.startswith("| full |"))
    cells = [c.strip() for c in full_row.strip().strip("|").split("|")]
    assert cells[5] == "0.000"


def test_t_perf_ttft_averages_across_seeds(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    for seed, value in ((0, 100.0), (1, 200.0)):
        _perf_cell(
            store, n=13, l=512, variant_tag="full", ttft_ms=value, seed=seed
        )
    out = render("T_perf_ttft", store=store)
    full_row = next(line for line in out.splitlines() if line.startswith("| full |"))
    cells = [c.strip() for c in full_row.strip().strip("|").split("|")]
    # Mean of 100 and 200 is 150.
    assert cells[5] == "150.00"
