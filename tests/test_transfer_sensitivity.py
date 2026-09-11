from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.transfer_sensitivity import analyze_sensitivity


def _write(path: Path, seed: int, offset: float) -> None:
    with path.open("w", encoding="utf-8") as target:
        for mask, value in (
            ("full", 0.8 + offset),
            ("ddi_qf", 0.5 + offset),
            ("blind", 0.2 + offset),
        ):
            for record in range(4):
                target.write(
                    json.dumps(
                        {
                            "run_id": f"x:M:data:{mask}:seed={seed}:h:r{record}",
                            "record_id": f"r{record}",
                            "metrics": {
                                "em": value,
                                "f1": value,
                                "presence": value,
                                "loss": 1 - value,
                            },
                            "provenance": {"seed": seed, "checkpoint_revision": str(offset)},
                        }
                    )
                    + "\n"
                )


def test_sensitivity_reports_training_estimands_and_joint_overlap(tmp_path: Path) -> None:
    architectures = {}
    for architecture, shift in (("A", 0.0), ("B", 0.1)):
        base, tuned = {}, {}
        for seed in range(3):
            base[seed] = tmp_path / f"{architecture}-base-{seed}.jsonl"
            tuned[seed] = tmp_path / f"{architecture}-tuned-{seed}.jsonl"
            _write(base[seed], seed, shift)
            _write(tuned[seed], seed, shift + 0.1)
        architectures[architecture] = (base, tuned)

    result = analyze_sensitivity(architectures, repeats=100, bootstrap_seed=1)
    cell = next(
        cell for cell in result["cells"] if cell["architecture"] == "A" and cell["metric"] == "f1"
    )
    assert cell["aggregate"]["ddi_training_change"]["mean"] == pytest.approx(0.1)
    assert cell["aggregate"]["full_path_change"]["mean"] == pytest.approx(0.1)
    assert cell["aggregate"]["full_ddi_gap_change"]["mean"] == pytest.approx(0.0)
    assert cell["manifest_overlap"]["0"]["common"] == 4
    assert cell["aggregate"]["ddi_training_change"]["posthoc_paired_ci95"] == pytest.approx(
        [0.1, 0.1]
    )
