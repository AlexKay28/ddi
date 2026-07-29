"""Tests for the YAML-driven experiment runner (issue #85).

Two layers:
  1. Pure-unit tests — YAML parsing, missing-key errors, per-cell isolation.
     No model load, so they run everywhere.
  2. Integration tests (gated on `DDI_INTEGRATION=1`) — end-to-end quality /
     cache dispatch on the tiny random Llama used elsewhere in the suite.

The store-row-shape parity is the load-bearing property here: the aggregator
in #86 has to be able to consume rows from either the old runners or
`ddi.run` without knowing which produced them, so every integration test
asserts on the run_id prefix and the metric names.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ddi import run as ddi_run

# ---------------------------------------------------------------------------
# Pure-unit: YAML loading + top-level validation.
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(body)
    return p


def test_load_yaml_returns_config_and_hash(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "kind: quality\nseed: 0\n")
    cfg, cfg_hash = ddi_run.load_yaml(p)
    assert cfg == {"kind": "quality", "seed": 0}
    assert isinstance(cfg_hash, str) and len(cfg_hash) == 16


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "- 1\n- 2\n")
    with pytest.raises(ValueError, match="not a mapping"):
        ddi_run.load_yaml(p)


def test_run_missing_kind_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "seed: 0\nmodel: irrelevant\n")
    with pytest.raises(ValueError, match="missing required key: 'kind'"):
        ddi_run.run(p)


def test_run_unknown_kind_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "kind: mystery\nseed: 0\n")
    with pytest.raises(ValueError, match="unknown kind"):
        ddi_run.run(p)


def test_run_missing_seed_raises(tmp_path: Path) -> None:
    # `kind` is valid but no seed is set anywhere.
    p = _write_yaml(tmp_path, "kind: quality\n")
    with pytest.raises(ValueError, match="'seed' must be set"):
        ddi_run.run(p)


def test_run_seed_override_wins_over_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--seed overrides YAML seed; runner dispatches with the CLI seed."""
    seen: dict[str, int] = {}

    def _fake_quality(cfg, cfg_hash, seed, yaml_path):
        seen["seed"] = seed

    monkeypatch.setitem(ddi_run._DISPATCH, "quality", _fake_quality)
    p = _write_yaml(tmp_path, "kind: quality\nseed: 5\n")
    ddi_run.run(p, seed_override=99)
    assert seen["seed"] == 99


# ---------------------------------------------------------------------------
# Pure-unit: mask spec parsing + variant tag helpers.
# ---------------------------------------------------------------------------


def test_parse_mask_spec_string() -> None:
    assert ddi_run._parse_mask_spec("ddi_qf") == ("ddi_qf", None)


def test_parse_mask_spec_dict() -> None:
    assert ddi_run._parse_mask_spec({"variant": "ddi_bridge_k", "bridge_k": 4}) == (
        "ddi_bridge_k",
        4,
    )


def test_parse_mask_spec_missing_variant_raises() -> None:
    with pytest.raises(ValueError, match="missing 'variant'"):
        ddi_run._parse_mask_spec({"bridge_k": 4})


def test_parse_mask_spec_wrong_type_raises() -> None:
    with pytest.raises(TypeError):
        ddi_run._parse_mask_spec(42)


def test_variant_tag() -> None:
    assert ddi_run._variant_tag("ddi_qf", None) == "ddi_qf"
    assert ddi_run._variant_tag("ddi_bridge_k", 8) == "ddi_bridge_k_k8"


def test_mask_kwargs_bridge_k_only_for_bridge() -> None:
    with pytest.raises(ValueError, match="only valid for variant='ddi_bridge_k'"):
        ddi_run._mask_kwargs("ddi_qf", 4)


def test_mask_kwargs_bridge_variant_requires_k() -> None:
    with pytest.raises(ValueError, match="requires bridge_k"):
        ddi_run._mask_kwargs("ddi_bridge_k", None)


def test_mask_kwargs_ok_paths() -> None:
    assert ddi_run._mask_kwargs("ddi_qf", None) == {}
    assert ddi_run._mask_kwargs("ddi_bridge_k", 3) == {"bridge_k": 3}


# ---------------------------------------------------------------------------
# Pure-unit: analytical speedup + sparsity parity with phase3_timing.
# ---------------------------------------------------------------------------


def test_analytical_speedup_full_is_one() -> None:
    assert ddi_run._analytical_speedup("full", N=13, L=512, Q=64) == 1.0


def test_analytical_speedup_ddi_qf_matches_paper_target() -> None:
    """RESEARCH_PLAN §7 table: N=13, L=512, Q=64 -> 10.59x for ddi_qf."""
    got = ddi_run._analytical_speedup("ddi_qf", N=13, L=512, Q=64)
    assert got == pytest.approx(10.59, rel=0.01)


def test_sparsity_matches_target() -> None:
    from ddi.masks import build_mask

    allow = build_mask([512] * 13, query_len=64, answer_len=0, variant="ddi_qf")
    assert ddi_run._sparsity(allow) == pytest.approx(0.906, abs=0.01)


# ---------------------------------------------------------------------------
# Pure-unit: per-dispatcher required-key validation.
# ---------------------------------------------------------------------------


def test_quality_requires_model(tmp_path: Path) -> None:
    body = "kind: quality\nseed: 0\ndatasets: [nq]\nmasks: [full]\nmetrics: [f1]\n"
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required key: 'model'"):
        ddi_run.run(p)


def test_quality_requires_datasets(tmp_path: Path) -> None:
    body = "kind: quality\nseed: 0\nmodel: irrelevant\nmasks: [full]\nmetrics: [f1]\n"
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required key: 'datasets'"):
        ddi_run.run(p)


def test_timing_requires_backends(tmp_path: Path) -> None:
    body = (
        "kind: timing\nseed: 0\nmodel: irrelevant\n"
        "Ns: [1]\nLs: [1]\nmasks: [full]\n"
    )
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required key: 'backends'"):
        ddi_run.run(p)


def test_cache_requires_task(tmp_path: Path) -> None:
    body = "kind: cache\nseed: 0\nmodel: irrelevant\n"
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required key: 'task'"):
        ddi_run.run(p)


def test_cache_unknown_task_raises(tmp_path: Path) -> None:
    body = "kind: cache\nseed: 0\nmodel: irrelevant\ntask: mystery\n"
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown task"):
        ddi_run.run(p)


# ---------------------------------------------------------------------------
# Pure-unit: per-cell isolation.  We monkeypatch a dispatcher's inner loop
# so the outer wrapper's try/except is what we're exercising.
# ---------------------------------------------------------------------------


def test_per_cell_isolation_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """First cell throws; second cell must still run and be reachable."""
    called: list[tuple[str, str]] = []

    def _raise_or_record(**kwargs):
        called.append((kwargs["dataset"], kwargs["mask_variant"]))
        if kwargs["dataset"] == "dsA":
            raise RuntimeError("boom")

    monkeypatch.setattr(ddi_run, "_quality_cell", _raise_or_record)
    # Bypass real model load — the fake _quality_cell doesn't touch it.
    monkeypatch.setattr(
        ddi_run.inference,
        "load",
        lambda *a, **k: (_FakeModel(), object()),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    body = (
        "kind: quality\nseed: 0\nmodel: irrelevant\n"
        "datasets: [dsA, dsB]\nmasks: [full]\nmetrics: [f1]\n"
    )
    p = _write_yaml(tmp_path, body)
    ddi_run.run(p)

    assert called == [("dsA", "full"), ("dsB", "full")]
    err = capsys.readouterr().err
    assert "cell FAILED" in err and "dsA" in err


class _FakeModel:
    """Minimal stub with an `eval()` and a `.to()` for the runner's setup path."""

    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self


# ---------------------------------------------------------------------------
# Integration: real dispatch through the store on a tiny random Llama.
# Gated on DDI_INTEGRATION=1 because it triggers a small HF download.
# ---------------------------------------------------------------------------

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_INTEG = pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="integration test; set DDI_INTEGRATION=1 to enable",
)


def _read_rows(store: Path) -> list[dict]:
    return [json.loads(line) for line in store.read_text().splitlines() if line.strip()]


@_INTEG
def test_quality_dispatch_writes_expected_rows(tmp_path: Path) -> None:
    """End-to-end: kind: quality YAML → store rows carry phase2_train_free prefix + metrics."""
    from ddi.data import register_loader

    # Tiny in-memory loader so we don't hit the HF datasets hub.
    ds_name = "tiny_qa_run85"

    def _loader(split, seed):
        yield {
            "id": "r1",
            "info_contexts": ["alpha beta"],
            "dialog": [
                {"role": "user", "content": "What?"},
                {"role": "assistant", "content": "beta"},
            ],
            "possible_answers": ["beta"],
        }

    try:
        register_loader(ds_name)(_loader)
    except ValueError:
        pass  # already registered in a previous test invocation

    store = tmp_path / "runs.jsonl"
    body = (
        f"kind: quality\nseed: 0\nmodel: {_TINY}\n"
        f"datasets: [{ds_name}]\nmasks: [full]\nmetrics: [em, f1, presence, loss]\n"
        f"max_records: 1\nmax_new_tokens: 4\noutput_store: {store}\n"
    )
    p = _write_yaml(tmp_path, body)
    ddi_run.run(p)

    rows = _read_rows(store)
    assert rows, "quality dispatcher wrote no rows"
    assert all(r["run_id"].startswith("phase2_train_free:") for r in rows)
    metrics_seen = {r["metric"] for r in rows}
    assert {"em", "f1", "presence", "loss"} <= metrics_seen


@_INTEG
def test_timing_dispatch_writes_expected_rows(tmp_path: Path) -> None:
    """End-to-end: kind: timing YAML → store has phase3_timing prefix + ftt/tpot rows."""
    store = tmp_path / "runs.jsonl"
    body = (
        f"kind: timing\nseed: 0\nmodel: {_TINY}\n"
        "Ns: [1]\nLs: [8]\nbackends: [sdpa]\nmasks: [full]\n"
        "Q: 4\nn_repeats: 1\nwarmup: 0\nn_new_tokens: 2\n"
        f"output_store: {store}\n"
    )
    p = _write_yaml(tmp_path, body)
    ddi_run.run(p)

    rows = _read_rows(store)
    assert rows
    assert all(r["run_id"].startswith("phase3_timing:") for r in rows)
    metrics_seen = {r["metric"] for r in rows}
    assert "mask_sparsity" in metrics_seen
    assert "analytical_speedup" in metrics_seen
    # Timing rows are written as median / iqr_low / iqr_high / repeats.
    assert any(m.startswith("ftt_") for m in metrics_seen)
    assert any(m.startswith("tpot_") for m in metrics_seen)


@_INTEG
def test_cache_correctness_dispatch_writes_expected_rows(tmp_path: Path) -> None:
    """End-to-end: kind: cache task: correctness → store has phase4_cache prefix + kv rows."""
    from ddi.data import register_loader

    ds_name = "tiny_docs_run85"

    def _loader(split, seed):
        yield {
            "id": "r1",
            "info_contexts": ["alpha beta"],
            "dialog": [{"role": "user", "content": "?"}, {"role": "assistant", "content": "beta"}],
            "possible_answers": ["beta"],
        }

    try:
        register_loader(ds_name)(_loader)
    except ValueError:
        pass

    store = tmp_path / "runs.jsonl"
    cache_dir = tmp_path / "cache"
    body = (
        f"kind: cache\ntask: correctness\nseed: 0\nmodel: {_TINY}\n"
        f"dataset: {ds_name}\nmax_records: 1\nlayout: restart\n"
        f"cache_dir: {cache_dir}\noutput_store: {store}\n"
    )
    p = _write_yaml(tmp_path, body)
    ddi_run.run(p)

    rows = _read_rows(store)
    assert rows
    assert all(r["run_id"].startswith("phase4_cache:") for r in rows)
    metrics_seen = {r["metric"] for r in rows}
    assert {"kv_deterministic", "kv_cache_roundtrip"} <= metrics_seen
