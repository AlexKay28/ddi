"""Tests for ddi/reports/phase2_base_vs_instruct.py — T4_probe_instruct renderer."""

from __future__ import annotations

import json
from pathlib import Path

from ddi.report import list_reports, render
from ddi.reports.phase2_base_vs_instruct import _family_and_kind, _parse


def _write(store: Path, run_id: str, metric: str, value: float, seed: int = 0) -> None:
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


def test_t4_is_registered():
    assert "T4_probe_instruct" in list_reports()


def test_parse_valid_run_id():
    rid = "phase2_train_free:Qwen/Qwen2.5-7B:nq:ddi_qf:seed=0:sha:ex_0"
    assert _parse(rid) == ("Qwen/Qwen2.5-7B", "nq", "ddi_qf")


def test_parse_rejects_non_phase2_prefix():
    assert _parse("phase1_baseline:x:y:seed=0:sha:ex_0") is None


def test_family_and_kind():
    assert _family_and_kind("Qwen/Qwen2.5-7B") == ("Qwen/Qwen2.5-7B", "base")
    assert _family_and_kind("Qwen/Qwen2.5-7B-Instruct") == ("Qwen/Qwen2.5-7B", "instruct")
    assert _family_and_kind("meta-llama/Llama-3.1-8B-instruct") == (
        "meta-llama/Llama-3.1-8B",
        "instruct",
    )


def test_t4_renders_with_paired_data(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # base Qwen: full=0.70, ddi_qf=0.65 → gap_base = 0.05
    # instruct Qwen: full=0.72, ddi_qf=0.71 → gap_instruct = 0.01 (instruct is more DDI-friendly)
    for seed in (0, 1, 2):
        _write(store, f"phase2_train_free:Qwen:nq:full:seed={seed}:sha:ex_{seed}", "em", 0.70, seed)
        _write(store, f"phase2_train_free:Qwen:nq:ddi_qf:seed={seed}:sha:ex_{seed}", "em", 0.65, seed)
        _write(
            store,
            f"phase2_train_free:Qwen-Instruct:nq:full:seed={seed}:sha:ex_{seed}",
            "em",
            0.72,
            seed,
        )
        _write(
            store,
            f"phase2_train_free:Qwen-Instruct:nq:ddi_qf:seed={seed}:sha:ex_{seed}",
            "em",
            0.71,
            seed,
        )

    out = render("T4_probe_instruct", store=store)
    assert isinstance(out, str)
    assert "Qwen" in out
    assert "nq" in out
    # effect = 0.05 - 0.01 = 0.04
    assert "+0.0400" in out
    assert "<!-- provenance -->" in out
