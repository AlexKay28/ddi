"""The dataset list a graph note advertises must be the list the job actually runs.

VIBEQUEUE-2 row 8 finished nine eval cells while its own report header read
``datasets: hotpot_qa, musique, kilt_nq, synthetic_qa (all four, always)``. Nothing
failed and nothing warned — kilt_nq had been dropped from the YAML in ``d1e2f6662c``
(its loader raises rather than fabricating title-only documents, R1) but four graph
notes carried their own hardcoded copy of the list and none were updated.

That is the worst shape of wrong: a report describing an experiment that did not happen,
posted to a ticket where the numbers are read by people who did not launch the run. The
fix was to write the list once, in ``constant.EVAL_DATASETS``. These tests are what stop
it drifting again — edit either the constant or a YAML alone and one of them fails.

The graph module itself (``nirvana/graphs/ddi_rows.py``) cannot be imported here because
it needs ``vh3``, which only exists inside the ya build. ``constant`` has no such
dependency, which is precisely why the shared list lives there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]

if str(_ROOT / "nirvana") not in sys.path:
    sys.path.insert(0, str(_ROOT / "nirvana"))

from common import constant  # noqa: E402

# Both configs feed the same claim: a train/eval dataset mismatch is the confound
# `ddi/prompt.py` and this constant exist to prevent, so they are checked together.
_CONFIGS = ("quality.yaml", "finetune.yaml")


def _yaml_datasets(name: str) -> list[str]:
    raw = (_ROOT / "experiments" / name).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw) or {}
    return list(parsed["datasets"])


@pytest.mark.parametrize("name", _CONFIGS)
def test_constant_matches_yaml(name: str) -> None:
    """`EVAL_DATASETS` is the list the job will actually load, in order."""
    assert list(constant.EVAL_DATASETS) == _yaml_datasets(name), (
        f"constant.EVAL_DATASETS disagrees with experiments/{name}. "
        "Update the constant and the YAML together — the graph notes render the "
        "constant, so a mismatch means every posted report describes a run that did "
        "not happen."
    )


def test_quality_and_finetune_agree() -> None:
    """Train and eval must see the same datasets.

    Checked independently of the constant: if someone edits both YAMLs and the constant
    consistently but wrongly, the test above passes and this one still catches a
    train/eval split.
    """
    quality, finetune = (_yaml_datasets(n) for n in _CONFIGS)
    assert quality == finetune, (
        f"quality.yaml has {quality} but finetune.yaml has {finetune}; a model trained "
        "on one dataset list and scored on another reports a deficit that has nothing "
        "to do with the mask"
    )


def test_kilt_nq_is_not_advertised() -> None:
    """The specific regression: kilt_nq must not reappear without its loader working.

    `ddi/data/kilt_nq.py` raises rather than fabricating title-only documents (R1), so a
    config naming it aborts every run that reaches the third dataset. Six runs were
    stopped for exactly this.
    """
    assert "kilt_nq" not in constant.EVAL_DATASETS
    for name in _CONFIGS:
        assert "kilt_nq" not in _yaml_datasets(name), (
            f"experiments/{name} lists kilt_nq; its loader raises, so every run using "
            "the default dataset list will abort partway through"
        )


def test_datasets_note_renders_the_constant() -> None:
    """The note string graphs render must contain exactly the configured datasets.

    This asserts on `constant.DATASETS_NOTE` — **the same object `ddi_rows` renders** —
    rather than rebuilding the string here. The earlier version did rebuild it, and so
    only ever checked that its own copy equalled the constant, which cannot fail. It
    missed a fifth hardcoded dataset list in `_rebaseline_row` for exactly that reason.
    """
    note = constant.DATASETS_NOTE
    assert note.startswith("datasets:  ")
    assert note == "datasets:  " + ", ".join(constant.EVAL_DATASETS)
    for ds in constant.EVAL_DATASETS:
        assert ds in note
    assert "kilt_nq" not in note
    assert "all four" not in note


def test_graph_notes_do_not_hardcode_a_dataset_list() -> None:
    """No note in `ddi_rows.py` may spell the dataset list out by hand.

    The constant existed and four notes still carried their own copies; the fix was
    applied to those four and a fifth was written afterwards. Neither the constant nor
    the note-string test can catch that — only a check on the source can, because the
    copy is *correct* on the day it is written and goes stale silently later.

    Source-level rather than import-level because `ddi_rows` needs `vh3`, which exists
    only inside the ya build.
    """
    source = (_ROOT / "nirvana" / "graphs" / "ddi_rows.py").read_text(encoding="utf-8")
    joined = ", ".join(constant.EVAL_DATASETS)
    offenders = [
        line.strip()
        for line in source.splitlines()
        if joined in line and "DATASETS_NOTE" not in line
    ]
    assert not offenders, (
        "a graph note spells the dataset list out instead of rendering "
        f"constant.DATASETS_NOTE: {offenders}"
    )
