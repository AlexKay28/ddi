"""Tests for run identity capture (issue #7)."""

from __future__ import annotations

import subprocess
from datetime import datetime

import pytest

from ddi import provenance
from ddi.provenance import Provenance, capture


def _head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_capture_matches_current_head() -> None:
    prov = capture("cfg", 0, "model@rev", "data@rev")
    assert prov.git_sha == _head_sha()
    assert len(prov.git_sha) == 40


def test_capture_field_types_and_values() -> None:
    prov = capture("cfg-hash", 42, "model@abc", "data@def")
    assert isinstance(prov, Provenance)
    assert isinstance(prov.git_sha, str)
    assert isinstance(prov.dirty, bool)
    assert isinstance(prov.config_hash, str) and prov.config_hash == "cfg-hash"
    assert isinstance(prov.seed, int) and prov.seed == 42
    assert isinstance(prov.model_revision, str) and prov.model_revision == "model@abc"
    assert isinstance(prov.dataset_revision, str) and prov.dataset_revision == "data@def"
    assert isinstance(prov.gpu_model, str)
    assert isinstance(prov.driver_version, str)
    assert isinstance(prov.cuda_version, str)
    assert isinstance(prov.torch_version, str)
    assert isinstance(prov.timestamp_utc, str)


def test_timestamp_parses_and_is_utc() -> None:
    prov = capture("cfg", 0, "m", "d")
    parsed = datetime.fromisoformat(prov.timestamp_utc)
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_dirty_flag_reflects_working_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return real_run(cmd, *args, **kwargs)
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M some/file.py\n", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    prov = capture("cfg", 0, "m", "d")
    assert prov.dirty is True

    def fake_run_clean(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return real_run(cmd, *args, **kwargs)
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(provenance.subprocess, "run", fake_run_clean)
    prov = capture("cfg", 0, "m", "d")
    assert prov.dirty is False


def test_driver_version_unknown_when_nvidia_smi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(provenance.torch.cuda, "get_device_name", lambda _idx: "FakeGPU")
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)

    prov = capture("cfg", 0, "m", "d")
    assert prov.driver_version == "unknown"
    assert prov.gpu_model == "FakeGPU"
