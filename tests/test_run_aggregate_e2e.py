"""End-to-end integration: YAML -> ddi.run -> store -> ddi.aggregate -> summary.md (issue #90).

This closes the #81 refactor by exercising the *full* new pipeline through a
single YAML in one process: a `kind: quality` config runs against a tiny
random Llama on a stub loader, writes rows into a jsonl store, and is then
aggregated into a Markdown summary. The assertions cover the two contracts
the aggregate module owns end-to-end:

  1. ``ddi.run`` writes at least one row of the expected shape and metric
     surface into the target store.
  2. ``ddi.aggregate`` reads that store, emits a Markdown table with the
     fixed dataset-column header from #72, and appends the
     ``<!-- provenance -->`` footer from ``ddi.reports._provenance``.

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
from ddi.reports._provenance import FOOTER_MARKER

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_INTEG = pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="integration test; set DDI_INTEGRATION=1 to enable",
)


@_INTEG
def test_yaml_to_run_to_aggregate_e2e(tmp_path: Path) -> None:
    """Full round-trip: YAML -> ddi.run -> jsonl store -> ddi.aggregate -> summary.md."""
    # Must be one of the fixed dataset columns the aggregator's quality renderer
    # recognizes (`ddi/aggregate.py::_QUALITY_DATASET_COLUMNS`) — anything else is
    # silently dropped from the table and the render ends up with zero rows.
    # #105 pruned the column list to four live QA datasets; `synthetic_qa` is stable
    # (hermetic — no network, deterministic per (split, seed)).
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
    # Every row must carry the runner prefix + metric field the aggregator keys on.
    assert all(r["run_id"].startswith("phase2_train_free:") for r in rows)
    assert any(r["metric"] == "f1" for r in rows)
    # Provenance envelope (seed + model_revision) is what the aggregator's footer hashes.
    for r in rows:
        assert "provenance" in r and "seed" in r["provenance"]

    # Step 2: same YAML -> ddi.aggregate -> summary.md.
    summary = tmp_path / "summary.md"
    ddi_aggregate.aggregate(yaml_path, summary, store=store)

    text = summary.read_text()
    # Fixed dataset-column header from #72 (aggregate's quality renderer).
    header = next(line for line in text.splitlines() if line.startswith("| mask |"))
    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    assert cols[0] == "mask"
    assert "synthetic_qa" in cols  # first fixed column in the pruned #105 dataset order
    # The `full` row must be present (it's the only mask we ran).
    assert any(line.startswith("| full |") for line in text.splitlines())
    # Provenance footer marker emitted exactly once for the single table.
    assert text.count(FOOTER_MARKER) == 1
    assert "sha256" in text  # footer includes the store content hash
