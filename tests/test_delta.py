"""Tests for `ddi.delta` — the non-inferiority margin Δ (issue #12)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from ddi.delta import compute_delta, record_delta
from ddi.provenance import Provenance


@pytest.fixture
def prov() -> Provenance:
    return Provenance(
        git_sha="0" * 40,
        dirty=False,
        config_hash="cfg-hash",
        seed=0,
        model_revision="model@rev",
        dataset_revision="data@rev",
        gpu_model="cpu",
        driver_version="unknown",
        cuda_version="cpu",
        torch_version="2.4.0",
        timestamp_utc="2026-07-27T14:33:12+00:00",
    )


def _write_row(
    store: Path,
    run_id: str,
    metric: str,
    value: float,
    prov: Provenance,
    seed: int | None = None,
) -> None:
    """Direct JSONL write bypassing ddi.store so we can control the provenance seed."""
    prov_dict = dataclasses.asdict(prov)
    if seed is not None:
        prov_dict["seed"] = seed
    store.parent.mkdir(parents=True, exist_ok=True)
    with open(store, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "value": value,
                    "provenance": prov_dict,
                }
            )
            + "\n"
        )


def _seed_baseline(
    store: Path,
    prov: Provenance,
    *,
    model: str = "Qwen/Qwen2.5-7B",
    dataset: str = "nq",
    metric: str = "em",
    variant: str = "full",
    per_seed_values: dict[int, list[float]] | None = None,
) -> None:
    """Seed the store with baseline rows whose per-seed means are exactly those requested.

    The run_id is the real one `ddi.run._quality_cell` writes. Building it any other way
    is what let `compute_delta` filter on a prefix nothing produced while these tests
    stayed green.
    """
    if per_seed_values is None:
        per_seed_values = {
            0: [0.6] * 10,
            1: [0.65] * 10,
            2: [0.7] * 10,
        }
    for seed, values in per_seed_values.items():
        for i, v in enumerate(values):
            run_id = (
                f"phase2_train_free:{model}:{dataset}:{variant}:seed={seed}:abcd1234:rec{i}"
            )
            _write_row(store, run_id, metric, v, prov, seed=seed)


def test_compute_delta_basic(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov)

    delta = compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")

    assert delta.n_seeds == 3
    assert set(delta.per_seed_means) == {0, 1, 2}
    assert delta.per_seed_means[0] == pytest.approx(0.6, abs=1e-9)
    assert delta.per_seed_means[1] == pytest.approx(0.65, abs=1e-9)
    assert delta.per_seed_means[2] == pytest.approx(0.7, abs=1e-9)

    # Sample std (ddof=1) of [0.6, 0.65, 0.7]. Mean = 0.65; squared devs = 0.0025 + 0 + 0.0025;
    # variance = 0.005 / 2 = 0.0025; std = 0.05.
    expected_std = math.sqrt(0.005 / 2)
    assert delta.std == pytest.approx(expected_std, abs=1e-9)
    assert delta.delta == pytest.approx(2 * expected_std, abs=1e-9)
    assert delta.k == 2.0


def test_compute_delta_allows_one_checkpoint_per_seed(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    for seed, value in enumerate((0.6, 0.65, 0.7)):
        seed_prov = dataclasses.replace(
            prov,
            seed=seed,
            checkpoint_revision=f"results/checkpoints/tuned-seed-{seed}",
        )
        _write_row(
            store,
            f"phase2_train_free:TUNED:nq:full:seed={seed}:abcd1234:rec0",
            "em",
            value,
            seed_prov,
            seed=seed,
        )

    delta = compute_delta(store, model="TUNED", dataset="nq", metric="em")
    assert delta.n_seeds == 3


def test_compute_delta_rejects_conflicting_checkpoint_within_seed(
    tmp_path: Path, prov: Provenance
) -> None:
    store = tmp_path / "runs.jsonl"
    for checkpoint in ("checkpoint-a", "checkpoint-b"):
        seed_prov = dataclasses.replace(prov, checkpoint_revision=checkpoint)
        _write_row(
            store,
            "phase2_train_free:TUNED:nq:full:seed=0:abcd1234:rec0",
            "em",
            0.6,
            seed_prov,
            seed=0,
        )

    with pytest.raises(ValueError, match="seed=0"):
        compute_delta(store, model="TUNED", dataset="nq", metric="em")


def test_compute_delta_requires_min_seeds(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_baseline(
        store, prov, per_seed_values={0: [0.6] * 3, 1: [0.7] * 3}
    )

    with pytest.raises(ValueError, match=r">= 3 seeds"):
        compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")


def test_compute_delta_ignores_other_datasets(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov, dataset="nq")
    # Poison with rows from a different (model, dataset) that would skew std if included.
    _seed_baseline(
        store,
        prov,
        dataset="triviaqa",
        per_seed_values={0: [0.0] * 5, 1: [1.0] * 5, 2: [0.0] * 5},
    )
    _seed_baseline(
        store,
        prov,
        model="meta-llama/Llama-3.1-8B",
        per_seed_values={0: [0.0] * 5, 1: [1.0] * 5, 2: [0.0] * 5},
    )

    delta = compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")
    expected_std = math.sqrt(0.005 / 2)
    assert delta.std == pytest.approx(expected_std, abs=1e-9)


def test_compute_delta_ignores_other_metrics(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov, metric="em")
    # Same model/dataset but a different metric — must not contaminate the em result.
    _seed_baseline(
        store,
        prov,
        metric="f1",
        per_seed_values={0: [0.0] * 5, 1: [1.0] * 5, 2: [0.0] * 5},
    )

    delta = compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")
    expected_std = math.sqrt(0.005 / 2)
    assert delta.std == pytest.approx(expected_std, abs=1e-9)


def test_compute_delta_ignores_non_baseline_runs(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov)
    # A non-baseline run_id (say, from a later Phase 2 sweep) must be excluded even if
    # model/dataset/metric all match.
    for seed in (0, 1, 2):
        for i in range(5):
            _write_row(
                store,
                f"phase2_ddi_qf:Qwen/Qwen2.5-7B:nq:seed={seed}:abcd1234:rec{i}",
                "em",
                0.0 if seed != 1 else 1.0,
                prov,
                seed=seed,
            )

    delta = compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")
    expected_std = math.sqrt(0.005 / 2)
    assert delta.std == pytest.approx(expected_std, abs=1e-9)


def test_record_delta_writes_expected_rows(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov)
    delta = compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")

    record_delta(store, delta, prov)

    rows = [json.loads(line) for line in store.read_text().splitlines() if line.strip()]

    per_seed_rows = [r for r in rows if r["metric"] == "em_per_seed_mean"]
    assert len(per_seed_rows) == 3
    # The seed the mean applies to lives in the run_id (the aggregate row shares one
    # provenance so all rows carry the same provenance.seed); parse it back out.
    per_seed_by_run = {r["run_id"].rsplit(":seed=", 1)[-1]: r["value"] for r in per_seed_rows}
    assert per_seed_by_run["0"] == pytest.approx(0.6, abs=1e-9)
    assert per_seed_by_run["1"] == pytest.approx(0.65, abs=1e-9)
    assert per_seed_by_run["2"] == pytest.approx(0.7, abs=1e-9)
    for r in per_seed_rows:
        assert r["run_id"].startswith("phase1_delta:Qwen/Qwen2.5-7B:nq:metric=em:")

    delta_rows = [r for r in rows if r["metric"] == "delta_em"]
    assert len(delta_rows) == 1
    assert delta_rows[0]["value"] == pytest.approx(delta.delta, abs=1e-9)
    assert delta_rows[0]["run_id"] == "phase1_delta:Qwen/Qwen2.5-7B:nq:metric=em:k=2"


def test_custom_k(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov)

    delta = compute_delta(
        store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em", k=1.5
    )
    expected_std = math.sqrt(0.005 / 2)
    assert delta.k == 1.5
    assert delta.delta == pytest.approx(1.5 * expected_std, abs=1e-9)


def test_only_full_variant_feeds_delta(tmp_path: Path, prov: Provenance) -> None:
    """`ddi_qf` / `blind` rows must not widen Δ (ground rule 9).

    Δ is the noise floor the parity claim is judged against. Folding the masked
    variants into it would inflate the margin exactly when the mask has an effect,
    which is the one case the margin has to stay honest for.
    """
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov, variant="full")
    # Wildly dispersed rows under the other two masks: if they leaked into the
    # computation the std would be dominated by them, not by the `full` spread.
    for variant in ("ddi_qf", "blind"):
        _seed_baseline(
            store,
            prov,
            variant=variant,
            per_seed_values={0: [0.0] * 10, 1: [5.0] * 10, 2: [10.0] * 10},
        )

    delta = compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")

    assert delta.n_seeds == 3
    assert delta.per_seed_means == {
        0: pytest.approx(0.6),
        1: pytest.approx(0.65),
        2: pytest.approx(0.7),
    }
    assert delta.std == pytest.approx(math.sqrt(0.005 / 2), abs=1e-9)


def test_legacy_phase1_baseline_rows_are_not_counted(tmp_path: Path, prov: Provenance) -> None:
    """The prefix this module used to filter on is not a source of Δ.

    Nothing in the harness has ever written `phase1_baseline:` rows. Accepting them
    would resurrect the original bug in reverse: a store containing only hand-written
    rows of that shape would report a Δ that no eval produced.
    """
    store = tmp_path / "runs.jsonl"
    for seed, value in ((0, 0.6), (1, 0.65), (2, 0.7)):
        for i in range(10):
            _write_row(
                store,
                f"phase1_baseline:Qwen/Qwen2.5-7B:nq:seed={seed}:abcd1234:rec{i}",
                "em",
                value,
                prov,
                seed=seed,
            )

    with pytest.raises(ValueError, match="requires >= 3 seeds"):
        compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")


def test_run_id_seed_disagreeing_with_provenance_is_skipped(
    tmp_path: Path, prov: Provenance
) -> None:
    """The docstring's promise, as a test: a row whose two seeds disagree is dropped.

    `compute_delta` documents that the seed comes from `provenance.seed` "so a row whose
    provenance disagrees with its id is skipped rather than silently counted under the
    wrong seed". It parsed `parts[4]` only to satisfy the length check and compared it to
    nothing, so the row was counted — under the provenance's seed, which is precisely the
    "silently counted" outcome the sentence rules out.

    The poison row here is `seed=7` in the run_id and seed 1 in the provenance. Counted, it
    drags seed 1's mean from 0.65 to ~9.2 and Δ with it; Δ gates every parity claim in the
    grid, so this is the one sentence in the module an auditor would actually lean on.
    """
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov)  # seeds 0/1/2, means 0.6 / 0.65 / 0.7

    _write_row(
        store,
        "phase2_train_free:Qwen/Qwen2.5-7B:nq:full:seed=7:abcd1234:rec99",
        "em",
        95.0,
        prov,
        seed=1,
    )

    delta = compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")

    assert set(delta.per_seed_means) == {0, 1, 2}
    assert delta.per_seed_means[1] == pytest.approx(0.65, abs=1e-9)
    assert delta.std == pytest.approx(math.sqrt(0.005 / 2), abs=1e-9)


def test_seed_mismatched_rows_cannot_make_up_the_third_seed(
    tmp_path: Path, prov: Provenance
) -> None:
    """Skipped means skipped: a mismatched row does not count towards the >= 3 gate either.

    Two honest seeds plus rows whose run_id claims a third would otherwise satisfy
    `_MIN_SEEDS` with a seed that no run ever produced under that id.
    """
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov, per_seed_values={0: [0.6] * 5, 1: [0.7] * 5})
    for i in range(5):
        _write_row(
            store,
            f"phase2_train_free:Qwen/Qwen2.5-7B:nq:full:seed=2:abcd1234:rec{i}",
            "em",
            0.8,
            prov,
            seed=5,  # provenance disagrees with the run_id
        )

    with pytest.raises(ValueError, match=r">= 3 seeds"):
        compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")


def test_other_model_or_dataset_is_excluded(tmp_path: Path, prov: Provenance) -> None:
    """Positional parsing must not let a different model or dataset bleed in."""
    store = tmp_path / "runs.jsonl"
    _seed_baseline(store, prov, model="Qwen/Qwen2.5-7B", dataset="nq")
    _seed_baseline(
        store,
        prov,
        model="other/model",
        dataset="nq",
        per_seed_values={3: [9.0] * 10, 4: [9.5] * 10, 5: [10.0] * 10},
    )
    _seed_baseline(
        store,
        prov,
        model="Qwen/Qwen2.5-7B",
        dataset="hotpot_qa",
        per_seed_values={6: [1.0] * 10, 7: [2.0] * 10, 8: [3.0] * 10},
    )

    delta = compute_delta(store, model="Qwen/Qwen2.5-7B", dataset="nq", metric="em")

    assert set(delta.per_seed_means) == {0, 1, 2}
