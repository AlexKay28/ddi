"""Tests for the dataset framework (issue #9, Phase 1.1)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from ddi.data import list_datasets, load, manifest, register_loader
from ddi.data.loaders import _REGISTRY
from ddi.data.splits import hash_split, split_ids


@pytest.fixture(autouse=True)
def manifest_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the manifest root away from the real results/splits/ for every test."""
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


# ---------------------------------------------------------------------------
# registry + dispatch
# ---------------------------------------------------------------------------


def test_register_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def dummy(split: str, seed: int) -> Iterator[dict]:
        yield {"id": f"{split}-{seed}"}

    # Snapshot + restore the registry so we can add/remove test loaders freely.
    monkeypatch.setattr("ddi.data.loaders._REGISTRY", dict(_REGISTRY))

    register_loader("dummy_ds")(dummy)
    assert "dummy_ds" in list_datasets()

    got = list(load("dummy_ds", "train", 0))
    assert got == [{"id": "train-0"}]

    with pytest.raises(ValueError):
        register_loader("dummy_ds")(dummy)

    with pytest.raises(KeyError):
        list(load("nonexistent_ds", "train", 0))


def test_list_datasets_includes_registered_loaders() -> None:
    names = list_datasets()
    assert "synthetic_qa" in names
    assert "nq" in names
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# synthetic loader
# ---------------------------------------------------------------------------


def test_synthetic_deterministic() -> None:
    a = list(load("synthetic_qa", "train", 42))
    b = list(load("synthetic_qa", "train", 42))
    assert a == b


def test_synthetic_record_shape() -> None:
    records = list(load("synthetic_qa", "val", 7))
    assert len(records) > 0
    for rec in records:
        assert isinstance(rec["id"], str)
        assert isinstance(rec["info_contexts"], list)
        assert all(isinstance(d, str) for d in rec["info_contexts"])
        assert isinstance(rec["dialog"], list)
        assert all(
            isinstance(m, dict) and set(m) == {"role", "content"} for m in rec["dialog"]
        )
        assert rec["dialog"][-1]["role"] == "assistant"
        assert isinstance(rec["possible_answers"], list)
        assert all(isinstance(a, str) for a in rec["possible_answers"])
        assert rec["hop_type"] in {"single", "multi", "unknown"}
        assert rec["gold_position"] is None or isinstance(rec["gold_position"], int)
        assert isinstance(rec["meta"], dict)

        # The synthetic answer appears verbatim in exactly one document.
        ans = rec["possible_answers"][0]
        hits = sum(ans in doc for doc in rec["info_contexts"])
        assert hits == 1


def test_synthetic_split_partition() -> None:
    train = list(load("synthetic_qa", "train", 0))
    val = list(load("synthetic_qa", "val", 0))
    test = list(load("synthetic_qa", "test", 0))
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20

    ids = {r["id"] for r in train} | {r["id"] for r in val} | {r["id"] for r in test}
    assert len(ids) == 100  # disjoint splits


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def test_split_ids_partition() -> None:
    ids = [f"id-{i:03d}" for i in range(100)]
    a = split_ids(ids, val_frac=0.2, test_frac=0.2, seed=0)
    b = split_ids(ids, val_frac=0.2, test_frac=0.2, seed=0)
    c = split_ids(ids, val_frac=0.2, test_frac=0.2, seed=1)

    assert a == b
    assert a != c

    total = sum(len(v) for v in a.values())
    assert total == 100
    # No overlap between splits.
    assert not (set(a["train"]) & set(a["val"]))
    assert not (set(a["train"]) & set(a["test"]))
    assert not (set(a["val"]) & set(a["test"]))


def test_hash_split_order_invariant() -> None:
    a = hash_split(["a", "b", "c"])
    b = hash_split(["c", "a", "b"])
    c = hash_split(["b", "c", "a"])
    assert a == b == c

    assert hash_split(["a", "b"]) != hash_split(["a", "b", "c"])


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_manifest_writes_then_validates(tmp_path: Path) -> None:
    root = tmp_path / "splits"
    splits: dict = {
        "train": (10, "abcd" * 4),
        "val": (2, "1111" * 4),
        "test": (2, "2222" * 4),
    }

    assert manifest.read("toy", root=root) is None
    manifest.validate_or_write("toy", splits, root=root)

    payload = json.loads((root / "toy.json").read_text())
    assert payload["dataset"] == "toy"
    assert payload["splits"]["train"] == {"n": 10, "sha256": "abcd" * 4}

    # Second call with identical inputs succeeds.
    manifest.validate_or_write("toy", splits, root=root)

    # Mutating a hash raises with a message that mentions the split.
    tampered: dict = dict(splits)
    tampered["val"] = (2, "dead" * 4)
    with pytest.raises(RuntimeError, match=r"toy/val"):
        manifest.validate_or_write("toy", tampered, root=root)

    # Mutating a count raises too.
    tampered2: dict = dict(splits)
    tampered2["train"] = (11, "abcd" * 4)
    with pytest.raises(RuntimeError, match=r"toy/train"):
        manifest.validate_or_write("toy", tampered2, root=root)


def test_manifest_default_root_monkeypatched(manifest_root: Path) -> None:
    # Sanity check: writes go to the tmp root, not the real results/splits/.
    manifest.write("probe", {"train": (1, "0" * 16)})
    assert (manifest_root / "probe.json").exists()
    assert not Path("results/splits/probe.json").exists()


# ---------------------------------------------------------------------------
# synthetic loader interacts with the manifest correctly
# ---------------------------------------------------------------------------


def test_synthetic_manifest_written_on_first_load(manifest_root: Path) -> None:
    list(load("synthetic_qa", "train", 0))
    path = manifest_root / "synthetic_qa.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["dataset"] == "synthetic_qa"
    assert set(payload["splits"]) == {"train", "val", "test"}
    assert payload["splits"]["train"]["n"] == 60


def test_synthetic_manifest_mismatch_raises(manifest_root: Path) -> None:
    # First load writes.
    list(load("synthetic_qa", "train", 0))
    # Tamper with the committed hash so the next load should refuse.
    path = manifest_root / "synthetic_qa.json"
    payload = json.loads(path.read_text())
    payload["splits"]["train"]["sha256"] = "0" * 16
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=r"synthetic_qa/train"):
        list(load("synthetic_qa", "train", 0))


# ---------------------------------------------------------------------------
# NQ integration — gated so unit runs stay offline
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="NQ loader hits HuggingFace; opt in with DDI_INTEGRATION=1",
)
def test_nq_load_first_record() -> None:
    stream = load("nq", "val", seed=0)
    rec = next(iter(stream))

    assert isinstance(rec["id"], str) and rec["id"]
    assert rec["info_contexts"] == []
    assert len(rec["dialog"]) == 2
    assert rec["dialog"][0]["role"] == "user"
    assert rec["dialog"][1]["role"] == "assistant"
    assert isinstance(rec["possible_answers"], list) and rec["possible_answers"]
    assert rec["hop_type"] == "single"
    assert rec["gold_position"] is None
    assert isinstance(rec["meta"], dict)
