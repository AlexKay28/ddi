"""End-to-end integration: YAML -> ddi.run -> store -> ddi.aggregate -> summary.json.

This exercises the full pipeline through a single YAML in one process: a
``kind: quality`` config runs against a tiny random Llama on a stub loader,
writes rows into a jsonl store, and is then aggregated into a JSON summary.

Assertions cover the two contracts the aggregate module owns end-to-end:

  1. ``ddi.run`` writes at least one row of the expected shape and metric
     surface into the target store.
  2. ``ddi.aggregate`` reads that store and emits a JSON envelope whose
     ``tables[0].data`` maps ``model → mask → dataset → value`` for the run.

Gated on ``DDI_INTEGRATION=1`` — matches the convention already in
``test_run.py`` and ``test_inference.py`` because the tiny model download
and CPU inference push CI runtime past the pure-unit budget.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ddi import aggregate as ddi_aggregate
from ddi import run as ddi_run

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_INTEG = pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="integration test; set DDI_INTEGRATION=1 to enable",
)


@_INTEG
def test_yaml_to_run_to_aggregate_e2e(tmp_path: Path) -> None:
    """Full round-trip: YAML -> ddi.run -> jsonl store -> ddi.aggregate -> summary.json."""
    # Must be one of the fixed dataset keys the aggregator's quality collector
    # recognizes (`ddi/aggregate.py::_QUALITY_DATASET_COLUMNS`) — anything else is
    # silently dropped from the JSON and the collect ends up with zero cells.
    ds_name = "synthetic_qa"

    store = tmp_path / "runs.jsonl"
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(
        f"kind: quality\n"
        f"seed: 0\n"
        f"model: {_TINY}\n"
        f"datasets: [{ds_name}]\n"
        f"masks: [full]\n"
        f"metrics: [f1]\n"
        f"max_records: 1\n"
        f"max_new_tokens: 4\n"
        f"output_store: {store}\n"
        f"report:\n"
        f"  tables:\n"
        f"    - kind: quality\n"
        f"      rows: mask\n"
        f"      cols: dataset\n"
        f"      metric: f1\n"
    )

    # Step 1: YAML -> ddi.run -> store rows.
    ddi_run.run(yaml_path)

    rows = [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    assert rows, "ddi.run wrote no rows to the store"
    assert all(r["run_id"].startswith("phase2_train_free:") for r in rows)
    assert any(r["metric"] == "f1" for r in rows)
    for r in rows:
        assert "provenance" in r and "seed" in r["provenance"]

    # Step 2: same YAML -> ddi.aggregate -> summary.json.
    summary = tmp_path / "summary.json"
    ddi_aggregate.aggregate(yaml_path, summary, store=store)

    envelope = json.loads(summary.read_text())
    # Envelope fields present and well-formed.
    assert envelope["yaml"] == str(yaml_path)
    assert envelope["store"] == str(store)
    assert len(envelope["store_sha256_8"]) == 8
    assert isinstance(envelope["git_sha"], str)
    # Exactly one table (the single `report.tables[0]` entry).
    assert len(envelope["tables"]) == 1
    tbl = envelope["tables"][0]
    assert tbl["kind"] == "quality"
    assert tbl["metric"] == "f1"
    # data shape: {model: {mask: {dataset: value}}}. The only mask we ran was `full`.
    models = list(tbl["data"])
    assert len(models) == 1
    assert "full" in tbl["data"][models[0]]
    assert ds_name in tbl["data"][models[0]]["full"]
