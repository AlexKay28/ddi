"""Tests for the persistent record cache under ``ddi/data/records_cache.py``.

The cache exists so a networked build node fetches datasets from the HF hub
once, and every subsequent invocation — including fully offline serverless
containers — reads records from disk. These tests cover the round-trip plus
the corner cases (missing file, empty file, partial write) that the fast-path
gate in every loader depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ddi.data import manifest, records_cache


@pytest.fixture(autouse=True)
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect both the cache root (via manifest.DEFAULT_ROOT) into tmp_path.

    ``records_cache`` late-binds against ``manifest.DEFAULT_ROOT`` so the two
    artefacts always ship together; the same monkeypatch covers both here.
    """
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


def test_records_path_uses_manifest_root(cache_root: Path) -> None:
    p = records_cache.records_path("hotpot_qa", 0, "val")
    assert p == cache_root / "hotpot_qa__seed0__val_records.jsonl"


def test_write_then_read_roundtrips(cache_root: Path) -> None:
    records = [
        {"id": "a", "info_contexts": ["x"], "meta": {"k": 1}},
        {"id": "b", "info_contexts": ["y", "z"], "meta": {"k": 2}},
    ]
    records_cache.write_split("hotpot_qa", 0, "val", records)

    got = list(records_cache.read_split("hotpot_qa", 0, "val"))
    assert got == records


def test_write_creates_parent_directory(cache_root: Path) -> None:
    # cache_root does not exist yet; writing should mkdir it.
    assert not cache_root.exists()
    records_cache.write_split("some_ds", 7, "train", [{"id": "x"}])
    assert cache_root.is_dir()
    assert records_cache.records_path("some_ds", 7, "train").is_file()


def test_has_cached_false_when_missing(cache_root: Path) -> None:
    assert records_cache.has_cached("hotpot_qa", 0, ("train", "val", "test")) is False


def test_has_cached_false_when_only_some_splits_present(cache_root: Path) -> None:
    records_cache.write_split("some_ds", 0, "train", [{"id": "a"}])
    records_cache.write_split("some_ds", 0, "val", [{"id": "b"}])
    # `test` still missing.
    assert records_cache.has_cached("some_ds", 0, ("train", "val", "test")) is False


def test_has_cached_true_when_all_present(cache_root: Path) -> None:
    for s in ("train", "val", "test"):
        records_cache.write_split("some_ds", 0, s, [{"id": s}])
    assert records_cache.has_cached("some_ds", 0, ("train", "val", "test")) is True


def test_has_cached_false_on_empty_file(cache_root: Path) -> None:
    # Simulates a crashed writer that opened the file but died before writing:
    # if we trust an empty file, downstream yields zero records forever.
    for s in ("train", "val"):
        records_cache.write_split("some_ds", 0, s, [{"id": s}])
    empty = records_cache.records_path("some_ds", 0, "test")
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    assert empty.is_file() and empty.stat().st_size == 0
    assert records_cache.has_cached("some_ds", 0, ("train", "val", "test")) is False


def test_read_split_tolerates_blank_lines(cache_root: Path) -> None:
    path = records_cache.records_path("some_ds", 0, "val")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Blank lines are common at the end of hand-edited files; they must not
    # surface as an empty-dict record downstream.
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")

    got = list(records_cache.read_split("some_ds", 0, "val"))
    assert got == [{"id": "a"}, {"id": "b"}]


def test_write_overwrites_previous_contents(cache_root: Path) -> None:
    records_cache.write_split("some_ds", 0, "val", [{"id": "old"}])
    records_cache.write_split("some_ds", 0, "val", [{"id": "new1"}, {"id": "new2"}])

    got = list(records_cache.read_split("some_ds", 0, "val"))
    assert got == [{"id": "new1"}, {"id": "new2"}]


def test_seed_and_split_partition_disk_layout(cache_root: Path) -> None:
    # Different seeds and different splits must not collide.
    records_cache.write_split("some_ds", 0, "val", [{"id": "s0-val"}])
    records_cache.write_split("some_ds", 1, "val", [{"id": "s1-val"}])
    records_cache.write_split("some_ds", 0, "test", [{"id": "s0-test"}])

    assert list(records_cache.read_split("some_ds", 0, "val")) == [{"id": "s0-val"}]
    assert list(records_cache.read_split("some_ds", 1, "val")) == [{"id": "s1-val"}]
    assert list(records_cache.read_split("some_ds", 0, "test")) == [{"id": "s0-test"}]
