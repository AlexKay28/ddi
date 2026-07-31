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
    # `kind` and `model` are valid but no seed is set anywhere.
    p = _write_yaml(tmp_path, "kind: quality\nmodel: irrelevant\n")
    with pytest.raises(ValueError, match="'seed' must be set"):
        ddi_run.run(p)


def test_run_seed_override_wins_over_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--seed overrides YAML seed; runner dispatches with the CLI seed."""
    seen: dict[str, int] = {}

    def _fake_quality(cfg, cfg_hash, seed, yaml_path):
        seen["seed"] = seed

    monkeypatch.setitem(ddi_run._DISPATCH, "quality", _fake_quality)
    p = _write_yaml(tmp_path, "kind: quality\nmodel: irrelevant\nseed: 5\n")
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
    with pytest.raises(ValueError, match="'model' must be set"):
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


def test_cache_correctness_requires_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cache/correctness must reject configs missing the `modes` key (post-#105 schema)."""
    monkeypatch.setattr(
        ddi_run.inference, "load", lambda *a, **k: (_FakeModel(), object())
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    body = (
        "kind: cache\ntask: correctness\nseed: 0\nmodel: irrelevant\n"
        "Ns: [1]\nLs: [8]\n"
    )
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required key: 'modes'"):
        ddi_run.run(p)


def test_cache_correctness_unknown_mode_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cache/correctness must reject an unknown mode (typo guard on the YAML)."""
    monkeypatch.setattr(
        ddi_run.inference, "load", lambda *a, **k: (_FakeModel(), object())
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    body = (
        "kind: cache\ntask: correctness\nseed: 0\nmodel: irrelevant\n"
        "modes: [ddi_cache, ddi_recompute, full_causal, garbage]\n"
        "Ns: [1]\nLs: [8]\n"
    )
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown mode"):
        ddi_run.run(p)


def test_cache_correctness_missing_full_causal_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cache/correctness requires `full_causal` in `modes` (it is the baseline reference)."""
    monkeypatch.setattr(
        ddi_run.inference, "load", lambda *a, **k: (_FakeModel(), object())
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    body = (
        "kind: cache\ntask: correctness\nseed: 0\nmodel: irrelevant\n"
        "modes: [ddi_cache, ddi_recompute]\nNs: [1]\nLs: [8]\n"
    )
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="'full_causal' must be in modes"):
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
    """End-to-end: kind: cache task: correctness → phase4_cache rows with the new
    (mode × N × L × query_i) shape from #105.

    Asserts:
      - Every run_id carries the `phase4_cache:` prefix and the mode/N=/L=/q fragments.
      - Every emitted metric is `max_abs_logit_diff`.
      - `full_causal` rows report exactly 0.0 (baseline is itself).
      - Row count matches `len(modes) × len(Ns) × len(Ls) × n_queries_per_doc`.
    """
    store = tmp_path / "runs.jsonl"
    cache_dir = tmp_path / "cache"
    modes = ["ddi_cache", "ddi_recompute", "full_causal"]
    Ns = [1, 2]
    Ls = [8]
    n_queries_per_doc = 2
    body = (
        f"kind: cache\ntask: correctness\nseed: 0\nmodel: {_TINY}\n"
        f"modes: {modes}\nNs: {Ns}\nLs: {Ls}\n"
        f"n_queries_per_doc: {n_queries_per_doc}\n"
        f"Q: 4\nanswer_len: 2\n"
        f"cache_dir: {cache_dir}\noutput_store: {store}\n"
    )
    p = _write_yaml(tmp_path, body)
    ddi_run.run(p)

    rows = _read_rows(store)
    assert rows, "cache/correctness dispatcher wrote no rows"
    # Every row carries the phase4_cache prefix + N=/L=/q fragments.
    for r in rows:
        rid = r["run_id"]
        assert rid.startswith("phase4_cache:")
        assert ":N=" in rid and ":L=" in rid and ":q" in rid
        # Mode segment must be one of the requested modes.
        mode_seg = rid.split(":")[2]
        assert mode_seg in modes, f"unexpected mode segment {mode_seg!r} in {rid!r}"

    # Every row is `max_abs_logit_diff` — no other metric name is emitted.
    metrics_seen = {r["metric"] for r in rows}
    assert metrics_seen == {"max_abs_logit_diff"}

    # full_causal rows are exactly 0.0 (baseline is itself by construction).
    full_rows = [r for r in rows if ":full_causal:" in r["run_id"]]
    assert full_rows, "no full_causal rows emitted"
    assert all(r["value"] == 0.0 for r in full_rows)

    # Total row count = |modes| × |Ns| × |Ls| × n_queries_per_doc.
    expected = len(modes) * len(Ns) * len(Ls) * n_queries_per_doc
    assert len(rows) == expected, f"expected {expected} rows, got {len(rows)}"


def test_finetune_dispatch_writes_expected_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kind: finetune dispatcher forwards YAML args to ddi.finetune.finetune and
    the wrapping cell tracker emits one 'cell 1/1' line + a summary line — matches
    the dispatch shape of every other kind.

    We monkey-patch `ddi.finetune.finetune` to a recorder so this test stays
    pure-unit (no model load) while still exercising the full ``run.py``
    dispatch path (YAML parse → provenance capture → cell tracker).
    """
    from ddi import finetune as ddi_finetune

    seen: dict[str, object] = {}

    def _fake_finetune(model_name, datasets, output_dir, **kwargs):
        seen["model_name"] = model_name
        seen["datasets"] = datasets
        seen["output_dir"] = output_dir
        seen["kwargs"] = kwargs
        # Emit one row so the test can also check the store-path plumbing.
        from ddi.store import append

        append(
            "finetune:test:ddi_qf:seed=0:xx:step=1",
            "kl_loss",
            1.23,
            kwargs["provenance"],
            store=kwargs["store_path"],
        )

    monkeypatch.setattr(ddi_finetune, "finetune", _fake_finetune)

    store = tmp_path / "runs.jsonl"
    output_dir = tmp_path / "ckpt"
    body = (
        "kind: finetune\nseed: 0\nmodel: irrelevant\n"
        "datasets: [dsA, dsB]\n"
        f"output_dir: {output_dir}\n"
        "max_records: 4\nheld_out_frac: 0.25\nepochs: 2\n"
        "lr: 5.0e-5\nkl_temperature: 2.0\ngrad_accum_steps: 2\n"
        "warmup_steps: 7\nlog_every: 10\n"
        f"output_store: {store}\n"
    )
    p = _write_yaml(tmp_path, body)
    ddi_run.run(p)

    assert seen["model_name"] == "irrelevant"
    assert seen["datasets"] == ["dsA", "dsB"]
    assert seen["output_dir"] == output_dir
    kw = seen["kwargs"]
    assert kw["max_records"] == 4
    assert kw["held_out_frac"] == 0.25
    assert kw["epochs"] == 2
    assert kw["lr"] == 5e-5
    assert kw["kl_temperature"] == 2.0
    assert kw["grad_accum_steps"] == 2
    assert kw["warmup_steps"] == 7
    assert kw["log_every"] == 10
    assert kw["store_path"] == store
    # Provenance is stamped by the dispatcher, not the caller.
    assert kw["provenance"].seed == 0
    assert kw["provenance"].model_revision == "irrelevant"

    # Store row plumbing: the fake finetuner's append went through.
    rows = _read_rows(store)
    assert rows and rows[0]["metric"] == "kl_loss" and rows[0]["value"] == 1.23


def test_finetune_dispatch_missing_datasets_raises(tmp_path: Path) -> None:
    body = (
        "kind: finetune\nseed: 0\nmodel: irrelevant\n"
        f"output_dir: {tmp_path / 'ckpt'}\n"
    )
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required key: 'datasets'"):
        ddi_run.run(p)


def test_finetune_dispatch_missing_output_dir_raises(tmp_path: Path) -> None:
    body = "kind: finetune\nseed: 0\nmodel: irrelevant\ndatasets: [ds]\n"
    p = _write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required key: 'output_dir'"):
        ddi_run.run(p)


@_INTEG
def test_cache_restart_quality_dispatch_writes_expected_rows(tmp_path: Path) -> None:
    """End-to-end: kind: cache task: restart_quality → phase4_restart rows for each dataset x layout."""
    from ddi.data import register_loader

    ds_name = "tiny_restart_run85"

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
    body = (
        f"kind: cache\ntask: restart_quality\nseed: 0\nmodel: {_TINY}\n"
        f"datasets: [{ds_name}]\nmasks: [full]\nlayouts: [sequential, restart]\n"
        f"max_records: 1\nmax_new_tokens: 4\noutput_store: {store}\n"
    )
    p = _write_yaml(tmp_path, body)
    ddi_run.run(p)

    rows = _read_rows(store)
    assert rows, "restart_quality dispatcher wrote no rows"
    assert all(r["run_id"].startswith("phase4_restart:") for r in rows)
    metrics_seen = {r["metric"] for r in rows}
    assert {"loss", "em", "f1", "presence"} <= metrics_seen
    layouts_seen = {r["run_id"].split(":")[4] for r in rows}
    assert layouts_seen == {"sequential", "restart"}
