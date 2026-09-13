from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.transfer_analysis import analyze


def _write_predictions(path: Path, seed: int, checkpoint: str | None, offset: float) -> None:
    with path.open("w", encoding="utf-8") as target:
        for mask, value in (("full", 0.8), ("ddi_qf", 0.6 + offset), ("blind", 0.2)):
            for record in range(10):
                row = {
                    "run_id": f"phase2_train_free:M:dataset:{mask}:seed={seed}:hash:r{record}",
                    "record_id": f"r{record}",
                    "metrics": {"em": value, "f1": value},
                    "provenance": {"seed": seed, "checkpoint_revision": checkpoint},
                }
                target.write(json.dumps(row) + "\n")


def test_analysis_accepts_seed_specific_checkpoints(tmp_path: Path) -> None:
    base = {}
    tuned = {}
    for seed in range(3):
        base[seed] = tmp_path / f"base-{seed}.jsonl"
        tuned[seed] = tmp_path / f"tuned-{seed}.jsonl"
        _write_predictions(base[seed], seed, None, 0.0)
        _write_predictions(tuned[seed], seed, f"checkpoint-{seed}", 0.2)

    result = analyze(base, tuned, repeats=100, bootstrap_seed=1)
    cell = next(row for row in result["cells"] if row["metric"] == "f1")
    assert cell["admissible"]
    assert cell["aggregate"]["parity_gap"]["mean"] == pytest.approx(0.0)
    assert cell["aggregate"]["normalized_closure"]["mean"] == pytest.approx(1.0)
    assert cell["verdict"] == "WITHIN_TOLERANCE"


def test_analysis_rejects_mismatched_record_ids(tmp_path: Path) -> None:
    base = {}
    tuned = {}
    for seed in range(3):
        base[seed] = tmp_path / f"base-{seed}.jsonl"
        tuned[seed] = tmp_path / f"tuned-{seed}.jsonl"
        _write_predictions(base[seed], seed, None, 0.0)
        _write_predictions(tuned[seed], seed, f"checkpoint-{seed}", 0.19)
    rows = tuned[0].read_text().splitlines()
    row = json.loads(rows[0])
    row["record_id"] = "different"
    rows[0] = json.dumps(row)
    tuned[0].write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="record IDs differ"):
        analyze(base, tuned, repeats=10)
