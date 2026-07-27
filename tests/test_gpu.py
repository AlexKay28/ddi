"""Tests for ddi/gpu.py."""

from __future__ import annotations

import pytest
import torch

from ddi import gpu


def test_is_available_matches_torch():
    assert gpu.is_available() == torch.cuda.is_available()


def test_gpu_info_shape():
    info = gpu.gpu_info()
    assert isinstance(info, dict)
    assert "cuda" in info
    assert "torch" in info
    if info["cuda"]:
        for k in ("cuda_version", "device_name", "compute_capability", "vram_gb", "bf16_supported"):
            assert k in info


def test_require_cuda_raises_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        gpu.require_cuda()


def test_configure_cpu_is_noop(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = gpu.configure(verbose=True)
    assert info["cuda"] is False
    captured = capsys.readouterr()
    assert "cpu-only" in captured.err


def test_configure_returns_info_dict():
    info = gpu.configure(verbose=False)
    assert isinstance(info, dict)
    assert "cuda" in info


def test_configure_silent_when_verbose_false(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gpu.configure(verbose=False)
    captured = capsys.readouterr()
    assert captured.err == ""
