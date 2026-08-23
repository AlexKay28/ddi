"""Tests for `ddi.parity` — the two-condition parity decision rule."""

from __future__ import annotations

import json

import pytest

from ddi.parity import (
    compute_cell,
    compute_deltas,
    compute_parity,
    format_cells,
    orient,
    read_means,
)
from ddi.parity import main as parity_main


def _tables(full_b, qf_b, blind_b, full_t, qf_t, blind_t):
    return (
        {"full": full_b, "ddi_qf": qf_b, "blind": blind_b},
        {"full": full_t, "ddi_qf": qf_t, "blind": blind_t},
    )


def test_orient_flips_only_lower_is_better_metrics() -> None:
    assert orient("f1", 0.5) == 0.5
    assert orient("em", 0.5) == 0.5
    assert orient("presence", 0.5) == 0.5
    assert orient("loss", 2.0) == -2.0


def test_perfect_result_is_success() -> None:
    base, tuned = _tables(0.60, 0.40, 0.20, 0.60, 0.60, 0.20)
    cell = compute_cell("hotpot_qa", "f1", base, tuned)
    assert cell.admissible
    assert cell.parity_gap == pytest.approx(0.0)
    assert cell.preservation_gap == pytest.approx(0.0)
    assert cell.closure == pytest.approx(1.0)
    assert cell.preservation == pytest.approx(1.0)
    assert cell.verdict.startswith("SUCCESS")


def test_parity_by_regression_is_caught() -> None:
    """`ddi_qf == full` on the tuned model, but `full` collapsed — must not pass."""
    base, tuned = _tables(0.60, 0.40, 0.20, 0.40, 0.40, 0.20)
    cell = compute_cell("hotpot_qa", "f1", base, tuned)
    assert cell.parity_gap == pytest.approx(0.0)  # parity looks perfect
    assert cell.preservation_gap == pytest.approx(0.5)  # but half the headroom is gone
    assert cell.verdict.startswith("PARITY BY REGRESSION")


def test_full_improving_is_not_reported_as_regression() -> None:
    """`full` getting BETTER must not print "full degraded".

    The preservation gap `r = (full_base - full_tuned)/S` is negative when the finetune
    improves `full`. Testing `abs(r)` conflates that with a collapse: VIBEQUEUE-2 row 2 on
    Qwen3.5-4B raised musique f1 from 0.5092 to 0.5699 (r = -0.134) and was reported as
    PARITY BY REGRESSION — the opposite of what happened.
    """
    # base: full .50 / ddi_qf .30 / blind .10   → S = 0.40
    # tuned: full .58 / ddi_qf .58 / blind .10  → parity held, `full` up by .08
    base, tuned = _tables(0.50, 0.30, 0.10, 0.58, 0.58, 0.10)
    cell = compute_cell("musique", "f1", base, tuned)
    assert cell.parity_gap == pytest.approx(0.0)
    assert cell.preservation_gap < 0  # negative == improvement
    assert cell.verdict.startswith("SUCCESS")
    assert "degraded" not in cell.verdict


def test_end_to_end_gap_is_the_sum_of_the_two_halves() -> None:
    base, tuned = _tables(0.60, 0.40, 0.20, 0.55, 0.45, 0.20)
    cell = compute_cell("hotpot_qa", "f1", base, tuned)
    assert cell.end_to_end_gap == pytest.approx(cell.parity_gap + cell.preservation_gap)


def test_no_effect_when_model_intact_but_mask_still_costs() -> None:
    base, tuned = _tables(0.60, 0.40, 0.20, 0.60, 0.40, 0.20)
    cell = compute_cell("hotpot_qa", "f1", base, tuned)
    assert cell.verdict.startswith("PARITY NOT REACHED")


def test_leakage_invalidates_the_cell() -> None:
    base, tuned = _tables(0.60, 0.60, 0.20, 0.60, 0.60, 0.40)
    cell = compute_cell("hotpot_qa", "f1", base, tuned)
    assert "INVALID" in cell.verdict
    assert cell.leakage == pytest.approx(0.5)
    assert "improved WITHOUT documents" in cell.verdict


def test_blind_moving_DOWN_invalidates_without_claiming_an_improvement() -> None:
    """The gate is `abs(leak) > tol`, so it fires both ways; the string must follow.

    VIBEQUEUE-2, musique/loss on `__all4_s0`: `blind` loss went 3.8887 -> 4.1755, so the
    tuned model got *worse* with no documents and `leak` was negative. The cell was
    correctly invalidated and then described as "the tuned model improved WITHOUT
    documents" — the opposite of what happened. A verdict string that states the sign
    backwards is worse than no string, because it is read instead of the number.
    """
    # base: full .60 / ddi_qf .60 / blind .40   tuned: same, but blind DOWN to .20
    base, tuned = _tables(0.60, 0.60, 0.40, 0.60, 0.60, 0.20)
    cell = compute_cell("hotpot_qa", "f1", base, tuned)
    assert cell.leakage < 0
    assert "INVALID" in cell.verdict
    assert "got WORSE without documents" in cell.verdict
    assert "improved" not in cell.verdict


def test_zero_headroom_is_inadmissible_not_perfect() -> None:
    """The regression this whole module exists for: no span means no claim.

    These are the real numbers from VIBEQUEUE-2 row 3 (hotpot_qa, f1). Naively they
    read as near-perfect parity; in fact the documents were worth 0.0017 f1, so the
    ratio is noise over noise.
    """
    base, tuned = _tables(0.0432, 0.0416, 0.0415, 0.0432, 0.0416, 0.0415)
    cell = compute_cell("hotpot_qa", "f1", base, tuned)
    assert not cell.admissible
    assert cell.parity_gap is None
    assert cell.closure is None
    assert "INADMISSIBLE" in cell.verdict


def test_em_all_zero_is_inadmissible() -> None:
    base, tuned = _tables(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cell = compute_cell("hotpot_qa", "em", base, tuned)
    assert not cell.admissible
    assert cell.span == pytest.approx(0.0)


def test_delta_raises_the_admissibility_floor() -> None:
    base, tuned = _tables(0.30, 0.20, 0.20, 0.30, 0.30, 0.20)
    assert compute_cell("d", "f1", base, tuned, delta=0.01).admissible
    # Δ=0.05 with k=3 needs a span above 0.15; this one is exactly 0.10.
    assert not compute_cell("d", "f1", base, tuned, delta=0.05).admissible


def test_loss_orientation_treats_lower_as_better() -> None:
    """Loss falls from base to tuned under both masks → success, not regression."""
    base, tuned = _tables(5.0, 6.0, 7.0, 5.0, 5.0, 7.0)
    cell = compute_cell("hotpot_qa", "loss", base, tuned)
    assert cell.span == pytest.approx(2.0)  # -5 - (-7)
    assert cell.closure == pytest.approx(1.0)
    assert cell.verdict.startswith("SUCCESS")


def test_missing_mask_raises() -> None:
    with pytest.raises(ValueError, match="missing mask"):
        compute_cell("d", "f1", {"full": 1.0, "blind": 0.0}, {"full": 1.0, "ddi_qf": 1.0, "blind": 0.0})


def _write_store(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for run_id, metric, value in rows:
            f.write(json.dumps({"run_id": run_id, "metric": metric, "value": value}) + "\n")


def test_read_means_averages_per_record_rows(tmp_path) -> None:
    store = tmp_path / "runs.jsonl"
    _write_store(store, [
        ("phase2_train_free:M:hotpot_qa:full:seed=0:abc:r1", "f1", 0.4),
        ("phase2_train_free:M:hotpot_qa:full:seed=0:abc:r2", "f1", 0.6),
        ("some_other_prefix:M:hotpot_qa:full:seed=0:abc:r3", "f1", 99.0),
    ])
    means = read_means(store)
    assert means[("M", "hotpot_qa", "full", "f1")] == pytest.approx(0.5)
    assert ("M", "hotpot_qa", "full", "loss") not in means


def test_compute_parity_end_to_end(tmp_path) -> None:
    store = tmp_path / "runs.jsonl"
    rows = []
    for model, vals in (("BASE", (0.6, 0.4, 0.2)), ("TUNED", (0.6, 0.6, 0.2))):
        for mask, v in zip(("full", "ddi_qf", "blind"), vals):
            rows.append((f"phase2_train_free:{model}:hotpot_qa:{mask}:seed=0:abc:r1", "f1", v))
    _write_store(store, rows)

    cells = compute_parity(store, "BASE", "TUNED", metrics=["f1"])
    assert len(cells) == 1
    assert cells[0].verdict.startswith("SUCCESS")
    assert "hotpot_qa" in format_cells(cells)


def test_compute_parity_rejects_identical_model_keys(tmp_path) -> None:
    """Guards the aggregator trap: same `model:` string means both checkpoints averaged."""
    store = tmp_path / "runs.jsonl"
    _write_store(store, [("phase2_train_free:M:hotpot_qa:full:seed=0:abc:r1", "f1", 0.4)])
    with pytest.raises(ValueError, match="same string"):
        compute_parity(store, "M", "M")


# ---- Δ actually reaching the gate ------------------------------------------------------
#
# `compute_parity(..., deltas=...)` was the only route a measured Δ had into `compute_cell`
# and nothing anywhere passed one: `main` had no flag for it, and no other call site
# existed. So `floor` was always `min_span` and `tol` was always 0.05, and `--k` was read,
# threaded through and then multiplied into a `delta` that was always None. Every SUCCESS
# and every PARITY NOT REACHED the grid has produced was decided by two constants. These
# tests exercise the wiring, so they assert on `main` rather than on `compute_cell`.


def _write_quality_rows(store, rows) -> None:
    """Append quality rows carrying a provenance, which `ddi.delta` needs and `_write_store` omits.

    `rows` are `(model, dataset, mask, seed, metric, value)`. The seed goes into the run_id
    *and* the provenance, because `compute_delta` skips rows where the two disagree.
    """
    with open(store, "a", encoding="utf-8") as f:
        for model, dataset, mask, seed, metric, value in rows:
            run_id = f"phase2_train_free:{model}:{dataset}:{mask}:seed={seed}:abc:r0"
            f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "metric": metric,
                        "value": value,
                        "provenance": {"seed": seed, "git_sha": "0" * 40},
                    }
                )
                + "\n"
            )


def _store_with_seeds(tmp_path, seeds, name="runs.jsonl"):
    """A base/tuned store whose base `full` cell varies across `seeds`.

    Base `full` takes 0.50 / 0.60 / 0.70 across the three seeds — mean 0.60, sample std
    0.10, so Δ = 2·0.10 = 0.20 and the default k=3 puts the admissibility floor at 0.60.
    The headroom is 0.60 - 0.20 = 0.40, comfortably above the 0.05 `min_span` fallback and
    comfortably below the measured floor. The tuned model shows textbook parity, so the
    fallback reads SUCCESS on a cell whose headroom is well under two noise floors.
    """
    store = tmp_path / name
    full_by_seed = {0: 0.50, 1: 0.60, 2: 0.70}
    rows = []
    for seed in seeds:
        rows.append(("BASE", "hotpot_qa", "full", seed, "f1", full_by_seed[seed]))
        rows.append(("BASE", "hotpot_qa", "ddi_qf", seed, "f1", 0.40))
        rows.append(("BASE", "hotpot_qa", "blind", seed, "f1", 0.20))
    for mask, v in (("full", 0.60), ("ddi_qf", 0.60), ("blind", 0.20)):
        rows.append(("TUNED", "hotpot_qa", mask, 0, "f1", v))
    _write_quality_rows(store, rows)
    return store


def test_cli_delta_changes_the_admissibility_outcome(tmp_path, capsys) -> None:
    """A measured Δ must be able to overturn the verdict the 0.05 fallback would give.

    Same store, same rows, same `--k`: with no Δ the cell is admissible and reads SUCCESS,
    because 0.40 of headroom clears a 0.05 floor. With the Δ the seeds actually measured
    (0.20), the floor is k·Δ = 0.60 and the cell carries less than one noise floor of
    signal, so no parity claim can be made from it in either direction.
    """
    store = _store_with_seeds(tmp_path, seeds=(0, 1, 2))

    # What the code did before Δ was wired in — every caller took this path.
    without_delta = compute_parity(store, "BASE", "TUNED", metrics=["f1"])
    assert len(without_delta) == 1
    assert without_delta[0].admissible
    assert without_delta[0].verdict.startswith("SUCCESS")

    rc = parity_main([
        "--store", str(store),
        "--base-model", "BASE",
        "--tuned-model", "TUNED",
        "--metrics", "f1",
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert "INADMISSIBLE" in out
    assert "SUCCESS" not in out
    # The floor printed is k*delta, not min_span: 3 * (2 * 0.10).
    assert "0.6000" in out
    assert "k*delta" in out


def test_cli_refuses_to_fall_back_when_delta_is_uncomputable(tmp_path, capsys) -> None:
    """Fewer than 3 seeds must stop the run, not quietly re-enable the constants.

    A silent fallback here is indistinguishable, in the output, from a Δ that was measured
    and cleared — which is exactly how the gate came to have never run.
    """
    store = _store_with_seeds(tmp_path, seeds=(0, 1))

    rc = parity_main([
        "--store", str(store),
        "--base-model", "BASE",
        "--tuned-model", "TUNED",
        "--metrics", "f1",
    ])
    captured = capsys.readouterr()

    assert rc == 2
    assert "SUCCESS" not in captured.out
    assert "|" not in captured.out  # no table at all, not even an inadmissible one
    assert "FATAL" in captured.err
    assert "hotpot_qa/f1" in captured.err
    assert ">= 3 seeds" in captured.err
    assert "--allow-missing-delta" in captured.err


def test_cli_missing_delta_fallback_is_opt_in_and_says_so(tmp_path, capsys) -> None:
    """The escape hatch exists, but it is explicit and it labels what it did."""
    store = _store_with_seeds(tmp_path, seeds=(0, 1))

    rc = parity_main([
        "--store", str(store),
        "--base-model", "BASE",
        "--tuned-model", "TUNED",
        "--metrics", "f1",
        "--allow-missing-delta",
    ])
    captured = capsys.readouterr()

    assert rc == 0
    assert "SUCCESS" in captured.out  # the old, constant-driven verdict
    assert "WARNING" in captured.err
    assert "not on a measured noise floor" in captured.err
    assert "hotpot_qa/f1" in captured.err


def test_cli_delta_store_can_be_separate(tmp_path, capsys) -> None:
    """Δ may live in another store — the seed grid is a different run from the parity eval."""
    parity_store = _store_with_seeds(tmp_path, seeds=(0,), name="parity.jsonl")
    delta_store = _store_with_seeds(tmp_path, seeds=(0, 1, 2), name="delta.jsonl")

    rc = parity_main([
        "--store", str(parity_store),
        "--delta-store", str(delta_store),
        "--base-model", "BASE",
        "--tuned-model", "TUNED",
        "--metrics", "f1",
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert "INADMISSIBLE" in out


def test_cli_delta_k_reaches_compute_delta(tmp_path, capsys) -> None:
    """`--delta-k` scales Δ itself; `--k` scales the finished Δ into a floor.

    Two knobs, both live. At delta_k=0.5 the margin is 0.5·0.10 = 0.05 and the floor is
    3·0.05 = 0.15, which 0.40 of headroom clears — so the same store that is inadmissible
    at the default k=2 becomes admissible.
    """
    store = _store_with_seeds(tmp_path, seeds=(0, 1, 2))

    rc = parity_main([
        "--store", str(store),
        "--base-model", "BASE",
        "--tuned-model", "TUNED",
        "--metrics", "f1",
        "--delta-k", "0.5",
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert "INADMISSIBLE" not in out
    assert "SUCCESS" in out


def test_compute_deltas_reports_failures_instead_of_raising(tmp_path) -> None:
    """Every uncovered cell in one pass, so a run is not fixed one missing seed at a time."""
    store = _store_with_seeds(tmp_path, seeds=(0, 1, 2))

    deltas, failures = compute_deltas(
        store, "BASE", [("hotpot_qa", "f1"), ("hotpot_qa", "em")]
    )

    assert deltas == {("hotpot_qa", "f1"): pytest.approx(0.20, abs=1e-9)}
    assert list(failures) == [("hotpot_qa", "em")]
    assert ">= 3 seeds" in failures[("hotpot_qa", "em")]


def test_compute_deltas_uses_only_the_full_variant(tmp_path) -> None:
    """Ground rule 9 survives the wiring: `ddi_qf` / `blind` rows must not widen Δ."""
    store = _store_with_seeds(tmp_path, seeds=(0, 1, 2))
    # Wildly dispersed masked rows on the base model; if they fed Δ the margin would be
    # dominated by them and the floor would move.
    _write_quality_rows(store, [
        ("BASE", "hotpot_qa", mask, seed, "f1", v)
        for mask in ("ddi_qf", "blind")
        for seed, v in ((0, 0.0), (1, 5.0), (2, 10.0))
    ])

    deltas, failures = compute_deltas(store, "BASE", [("hotpot_qa", "f1")])

    assert not failures
    assert deltas[("hotpot_qa", "f1")] == pytest.approx(0.20, abs=1e-9)
