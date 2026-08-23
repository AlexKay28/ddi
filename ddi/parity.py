"""Parity decision rule for the DDI claim (VIBEQUEUE-2 review).

The research question is *"can a model be fine-tuned so that it performs the same
under `ddi_qf` as under `full`?"*. That claim has two halves, and reporting only
the first one is how a broken model passes:

1. **Parity** — the tuned model scores the same under `ddi_qf` as under `full`.
2. **Preservation** — the tuned model under `full` scores the same as the *base*
   model under `full`. Without this, a model that collapses to its `ddi_qf` level
   scores perfect parity and is worthless.

Both are measured as a fraction of the headroom the documents are worth at all,
so numbers are comparable across datasets and metrics::

    S = full_base - blind_base                      # headroom (the denominator)
    P = (full_tuned  - ddi_qf_tuned) / S            # parity gap        -> 0
    R = (full_base   - full_tuned)   / S            # preservation gap  -> 0
    E = (full_base   - ddi_qf_tuned) / S            # end-to-end gap    -> 0
    L = (blind_tuned - blind_base)   / S            # leakage           -> 0

``E == P + R`` identically, so a failure is always attributable to one half or
the other. Equivalently as ratios: ``closure = 1 - P``, ``preservation = 1 - R``,
both targeting 1.0.

**The admissibility gate matters more than the ratios.** When ``S`` is at the
noise floor every ratio above is noise/noise and will swing wildly while looking
like signal. A cell whose headroom is not comfortably larger than the measurement
resolution cannot support a parity claim in either direction, and this module
reports it as ``inadmissible`` rather than emitting a number. On the numbers that
prompted this review, every quality-metric cell was inadmissible: `S(f1)` on
hotpot_qa was 0.0017 and `S(em)` was exactly 0.0.

Metric orientation is handled here: `em`/`f1`/`presence` are higher-is-better,
`loss` is lower-is-better and is negated before any of the formulas above.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ddi.delta import compute_delta
from ddi.runids import QualityRunId

# Higher-is-better metrics. Anything not listed is treated as lower-is-better and
# negated so a single set of formulas covers both orientations.
_HIGHER_IS_BETTER = frozenset({"em", "f1", "presence"})

REQUIRED_MASKS = ("full", "ddi_qf", "blind")

# Default admissibility: the headroom must be at least this many times Δ. With no Δ
# available, `min_span` is the absolute fallback floor.
DEFAULT_K = 3.0
DEFAULT_MIN_SPAN = 0.05

# `Δ = DEFAULT_DELTA_K · std`, mirroring `ddi.delta.compute_delta`'s own default. Distinct
# from `DEFAULT_K`, which multiplies the finished Δ to get the admissibility floor: one is
# how many sigmas make a noise floor, the other is how many noise floors make a claim.
DEFAULT_DELTA_K = 2.0

# The metrics a parity run covers when none are named. Hoisted out of `compute_parity` so
# `main` can require a Δ for exactly the set it is about to judge.
DEFAULT_METRICS = ("em", "f1", "presence", "loss")


@dataclass(frozen=True)
class ParityCell:
    """Parity verdict for one (dataset, metric) pair."""

    dataset: str
    metric: str
    span: float  # S — headroom on the base model
    parity_gap: float | None  # P
    preservation_gap: float | None  # R
    end_to_end_gap: float | None  # E
    leakage: float | None  # L
    admissible: bool
    verdict: str
    values: dict[str, float]  # raw oriented values that fed the formulas

    @property
    def closure(self) -> float | None:
        return None if self.parity_gap is None else 1.0 - self.parity_gap

    @property
    def preservation(self) -> float | None:
        return None if self.preservation_gap is None else 1.0 - self.preservation_gap


def orient(metric: str, value: float) -> float:
    """Return `value` flipped so that larger is always better."""
    return float(value) if metric in _HIGHER_IS_BETTER else -float(value)


def _verdict(p: float, r: float, tol: float) -> str:
    """Verdict from the parity gap `p` and the preservation gap `r`, in span units.

    `r = (full_base - full_tuned) / S`, so its SIGN carries the meaning and only the
    positive direction is a failure:

    * ``r > tol``  — `full` got worse. This is the regression the check exists to catch.
    * ``r < -tol`` — `full` got BETTER. Not a failure; the finetune improved the ordinary
      causal path as well as the masked one.

    Testing ``abs(r) <= tol`` conflates the two and reports an improvement as
    "full degraded — not bankable". VERIFIED 2026-08-07: row 2 on Qwen3.5-4B improved
    `full` from 0.5092 to 0.5699 f1 on musique (r = -0.134) and the old logic called it
    PARITY BY REGRESSION, which is the opposite of what happened. Read the sign.
    """
    # "within tolerance", never "free": clearing the gate means the gap did not exceed Δ,
    # NOT that it is zero. Third label in this function to say more than the arithmetic
    # supports, after `abs(R)` reporting improvement as regression and "NO EFFECT" naming
    # a 57 %-closure cell. Verdict strings name gates; magnitudes live in P, R, E and L.
    parity_ok = abs(p) <= tol
    regressed = r > tol
    improved = r < -tol

    if parity_ok and not regressed:
        if improved:
            return "SUCCESS: parity gap within tolerance, `full` improved too"
        return "SUCCESS: parity gap within tolerance, model intact"
    if parity_ok and regressed:
        return "PARITY BY REGRESSION: full degraded — not bankable"
    if not parity_ok and not regressed:
        # NOT "NO EFFECT". That string named the parity gate but read as "training did
        # nothing", which can be flatly false: VIBEQUEUE-2 row 2 on musique moved `ddi_qf`
        # em from 0.1680 to 0.3660 — closing 57 % of the gap — and still failed parity,
        # because 57 % is not 100 %. A cell can carry an enormous effect and fail this gate.
        # The verdict names the gate, so it must not editorialise about the effect; the
        # magnitude lives in P and in the raw values. Same failure mode as the `abs(R)` bug
        # above: a label that says more than the arithmetic supports.
        return "PARITY NOT REACHED: model intact, mask still costs quality"
    return "HARMFUL: full degraded and the mask still costs quality"


def compute_cell(
    dataset: str,
    metric: str,
    base: dict[str, float],
    tuned: dict[str, float],
    *,
    delta: float | None = None,
    k: float = DEFAULT_K,
    min_span: float = DEFAULT_MIN_SPAN,
) -> ParityCell:
    """Compute one parity cell from base/tuned per-mask means.

    `base` and `tuned` map mask name -> metric value in the metric's *native*
    orientation (this function handles the flip). Both must carry every mask in
    `REQUIRED_MASKS`; a missing mask raises rather than silently producing a
    partial verdict.
    """
    for name, table in (("base", base), ("tuned", tuned)):
        missing = [m for m in REQUIRED_MASKS if m not in table]
        if missing:
            raise ValueError(
                f"parity[{dataset}/{metric}]: {name} model is missing mask(s) {missing}; "
                f"have {sorted(table)}"
            )

    o = {f"{who}_{mask}": orient(metric, table[mask])
         for who, table in (("base", base), ("tuned", tuned))
         for mask in REQUIRED_MASKS}

    span = o["base_full"] - o["base_blind"]

    # Admissibility: the documents must be worth something on the base model before
    # "the mask costs nothing" means anything at all.
    floor = max(min_span, k * delta) if delta is not None else min_span
    admissible = span > floor

    if not admissible:
        reason = (
            f"INADMISSIBLE: headroom S={span:.4f} <= floor {floor:.4f} "
            f"({'k*delta' if delta is not None else 'min_span'}) — "
            "documents carry no measurable signal here, so parity is undefined"
        )
        return ParityCell(
            dataset=dataset,
            metric=metric,
            span=span,
            parity_gap=None,
            preservation_gap=None,
            end_to_end_gap=None,
            leakage=None,
            admissible=False,
            verdict=reason,
            values=o,
        )

    p = (o["tuned_full"] - o["tuned_ddi_qf"]) / span
    r = (o["base_full"] - o["tuned_full"]) / span
    e = (o["base_full"] - o["tuned_ddi_qf"]) / span
    leak = (o["tuned_blind"] - o["base_blind"]) / span

    # Tolerance in span units. With a Δ we use it directly; otherwise fall back to a
    # fixed 5% of the headroom, which is a *convention*, not a significance test —
    # `ddi.delta.compute_delta` is the real bar and needs >= 3 seeds. `main` therefore
    # refuses to reach this branch unless `--allow-missing-delta` says so out loud; the
    # `delta is None` path stays only for callers that are deliberately not making a claim.
    tol = (delta / span) if delta is not None else 0.05
    verdict = _verdict(p, r, tol)
    if abs(leak) > tol:
        # Names the OBSERVATION, not a presumed cause. The old string said "audit the
        # split", which points at train/test contamination — VERIFIED 2026-08-07 not to be
        # the cause here: hotpot's train pool is the seed-independent HF train split and
        # never intersects val/test. The real mechanism was the model acquiring in-domain
        # answering ability from the finetune, which raises `blind` legitimately and still
        # inflates apparent closure. A gate that names the wrong cause sends the reader to
        # audit something that is already clean. Fourth label in this module to assert more
        # than the arithmetic supports.
        #
        # The gate is `abs(leak) > tol`, so it fires in BOTH directions, and the string
        # above asserted only one of them. VERIFIED 2026-08-08 on the trainer-consistent
        # read of `__all4_s0`: musique/loss has `leak = -0.104` — `blind` loss went 3.8887
        # -> 4.1755, i.e. the tuned model got *worse* without documents — and the cell was
        # reported as "improved WITHOUT documents". Fifth label in this module to assert
        # more than the arithmetic supports, and the second to state the sign backwards.
        #
        # Both directions invalidate the cell, for different reasons, so the branch is on
        # what to tell the reader rather than on whether to flag it.
        if leak > 0:
            verdict = (
                f"INVALID (leakage {leak:+.3f}): the tuned model improved WITHOUT "
                "documents, so closure is partly parametric — subtract the blind gain "
                "before reading it"
            )
        else:
            verdict = (
                f"INVALID (blind moved {leak:+.3f}): the tuned model got WORSE without "
                "documents, so the span this cell is normalised by no longer describes "
                "the tuned model — closure here is inflated by a shrunken denominator"
            )

    return ParityCell(
        dataset=dataset,
        metric=metric,
        span=span,
        parity_gap=p,
        preservation_gap=r,
        end_to_end_gap=e,
        leakage=leak,
        admissible=True,
        verdict=verdict,
        values=o,
    )


# ---- store reading ---------------------------------------------------------------------


def read_means(store: Path) -> dict[tuple[str, str, str, str], float]:
    """Return {(model, dataset, mask, metric): mean value} from a quality store.

    The run_id shape is `ddi.runids.QualityRunId` — the same parser
    `ddi.aggregate`'s quality table and `ddi.delta.compute_delta` use. This function keeps
    the model in the key instead of collapsing it, because telling the base and tuned models
    apart is the entire point here.

    Before `ddi.runids` this function re-split the run_id itself and accepted any row with
    seven segments, while the other two required a well-formed ``seed=<int>`` — so a
    malformed row that the aggregator and Δ both dropped was still folded into a parity
    mean. They now agree by construction; `tests/test_runids.py` asserts it.
    """
    sums: dict[tuple[str, str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    with open(store, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = QualityRunId.parse(str(row.get("run_id", "")))
            if parsed is None:
                continue
            metric = row.get("metric")
            value = row.get("value")
            if not isinstance(metric, str) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, bool):
                continue
            key = (parsed.model, parsed.dataset, parsed.variant, metric)
            sums[key] += float(value)
            counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums}


def compute_parity(
    store: Path,
    base_model: str,
    tuned_model: str,
    *,
    metrics: list[str] | None = None,
    datasets: list[str] | None = None,
    deltas: dict[tuple[str, str], float] | None = None,
    k: float = DEFAULT_K,
    min_span: float = DEFAULT_MIN_SPAN,
    tuned_store: Path | None = None,
) -> list[ParityCell]:
    """Compute every admissible (dataset, metric) parity cell from one or two stores."""
    base_means = read_means(store)
    tuned_means = read_means(tuned_store) if tuned_store is not None else base_means

    if base_model == tuned_model:
        raise ValueError(
            "parity: base_model and tuned_model are the same string. The store keys the "
            "model off the config's `model:` field, so a run that reused it silently "
            "averaged both checkpoints into one cell — set a unique `model:` per row."
        )

    wanted_metrics = metrics or list(DEFAULT_METRICS)
    if datasets is None:
        datasets = sorted({d for (m, d, _v, _mt) in base_means if m == base_model})

    cells: list[ParityCell] = []
    for dataset in datasets:
        for metric in wanted_metrics:
            base = {
                mask: base_means[(base_model, dataset, mask, metric)]
                for mask in REQUIRED_MASKS
                if (base_model, dataset, mask, metric) in base_means
            }
            tuned = {
                mask: tuned_means[(tuned_model, dataset, mask, metric)]
                for mask in REQUIRED_MASKS
                if (tuned_model, dataset, mask, metric) in tuned_means
            }
            if len(base) < len(REQUIRED_MASKS) or len(tuned) < len(REQUIRED_MASKS):
                continue
            delta = (deltas or {}).get((dataset, metric))
            cells.append(
                compute_cell(
                    dataset, metric, base, tuned,
                    delta=delta, k=k, min_span=min_span,
                )
            )
    return cells


def compute_deltas(
    delta_store: Path,
    base_model: str,
    cells: Iterable[tuple[str, str]],
    *,
    delta_k: float = DEFAULT_DELTA_K,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], str]]:
    """Measure Δ per (dataset, metric) for the cells about to be judged.

    Returns ``(deltas, failures)``: `deltas` is exactly the mapping `compute_parity` takes,
    and `failures` maps each (dataset, metric) whose Δ could not be measured to the reason
    — in practice always `ddi.delta`'s "requires >= 3 seeds". Collected rather than raised
    so a caller can report every uncovered cell in one pass instead of one per re-run.

    Δ comes from `base_model`'s `full` rows alone; `ddi.delta.compute_delta` enforces that
    restriction itself and its docstring says why (folding the masked variants in would
    widen the margin exactly when the mask has an effect). This function does not relax it,
    and callers should not pass the tuned model here — the noise floor is a property of the
    measurement, not of the checkpoint under test.
    """
    deltas: dict[tuple[str, str], float] = {}
    failures: dict[tuple[str, str], str] = {}
    # `dict.fromkeys` rather than `set`: keeps the caller's order so the failure report
    # reads in the same order as the table below it.
    for dataset, metric in dict.fromkeys(cells):
        try:
            measured = compute_delta(
                store=delta_store,
                model=base_model,
                dataset=dataset,
                metric=metric,
                k=delta_k,
            )
        except ValueError as exc:
            # Only ValueError. A missing or unreadable store is not "not enough seeds" and
            # must not be reported as one — it propagates.
            failures[(dataset, metric)] = str(exc)
            continue
        deltas[(dataset, metric)] = measured.delta
    return deltas, failures


def format_cells(cells: list[ParityCell]) -> str:
    """Render parity cells as a markdown table, admissible ones first."""
    if not cells:
        return "no parity cells (no (dataset, metric) had all three masks on both models)"
    header = (
        "| dataset | metric | S | closure | preservation | E | leak | verdict |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]

    def fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:+.3f}"

    for c in sorted(cells, key=lambda c: (not c.admissible, c.dataset, c.metric)):
        lines.append(
            f"| {c.dataset} | {c.metric} | {c.span:.4f} | {fmt(c.closure)} | "
            f"{fmt(c.preservation)} | {fmt(c.end_to_end_gap)} | {fmt(c.leakage)} | "
            f"{c.verdict} |"
        )
    n_bad = sum(1 for c in cells if not c.admissible)
    if n_bad:
        lines.append("")
        lines.append(
            f"**{n_bad}/{len(cells)} cells inadmissible** — no parity claim can be made "
            "from them in either direction."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m ddi.parity", description=__doc__)
    ap.add_argument("--store", type=Path, required=True, help="quality store (JSONL)")
    ap.add_argument("--tuned-store", type=Path, default=None,
                    help="second store holding the tuned model's rows (default: --store)")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--tuned-model", required=True)
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=DEFAULT_K,
                    help="admissibility multiplier: the headroom must exceed k*Δ")
    ap.add_argument("--min-span", type=float, default=DEFAULT_MIN_SPAN,
                    help="absolute floor, used only where k*Δ is smaller")
    ap.add_argument("--delta-store", type=Path, default=None,
                    help="store to measure Δ from (default: --store)")
    ap.add_argument("--delta-k", type=float, default=DEFAULT_DELTA_K,
                    help="Δ = delta_k * std across the base model's per-seed `full` means")
    ap.add_argument(
        "--allow-missing-delta",
        action="store_true",
        help="do not stop when a cell has no measurable Δ; judge it against min_span and "
             "the 0.05 tolerance convention instead, and say so on stderr",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    shared = {
        "metrics": args.metrics,
        "datasets": args.datasets,
        "k": args.k,
        "min_span": args.min_span,
        "tuned_store": args.tuned_store,
    }

    # Two passes. The first exists only to learn which (dataset, metric) pairs actually
    # yield a cell — pairs missing a mask on either model produce nothing, and demanding a
    # Δ for those would block a run over a measurement it was never going to report. Its
    # verdicts are discarded; only the keys survive.
    probe = compute_parity(args.store, args.base_model, args.tuned_model, **shared)

    delta_store = args.delta_store if args.delta_store is not None else args.store
    deltas, failures = compute_deltas(
        delta_store,
        args.base_model,
        [(c.dataset, c.metric) for c in probe],
        delta_k=args.delta_k,
    )

    if failures:
        detail = "\n".join(f"  {d}/{m}: {why}" for (d, m), why in failures.items())
        if not args.allow_missing_delta:
            # Loud, and a non-zero exit. Δ *is* the admissibility gate; with none measured,
            # `floor` collapses to `min_span` and `tol` to the 0.05 convention, so SUCCESS
            # and PARITY NOT REACHED get decided by two constants that were never claimed
            # to describe this measurement. Printing that table anyway is the failure this
            # module's docstring calls more important than the ratios: for the whole life of
            # `deltas=` no caller ever passed one, and every verdict the grid has produced
            # so far came off those constants without ever saying so.
            print(
                f"FATAL: no measured Δ for {len(failures)} of {len(probe)} parity cells.\n"
                f"{detail}\n"
                f"Δ is the noise floor the admissibility gate is built from. Without it the "
                f"floor falls back to min_span={args.min_span:g} and the tolerance to the "
                f"0.05 convention, so every verdict below would be decided by constants "
                f"rather than by the measurement. Refusing to print that silently.\n"
                f"Fix: run the missing seeds — Δ needs >= 3 on the base model's `full` "
                f"cells — and check one with\n"
                f"  python -m ddi.delta_cli --store {delta_store} --model "
                f"{args.base_model} --dataset <dataset> --metric <metric>\n"
                f"or pass --allow-missing-delta to accept the constants for those cells.",
                file=sys.stderr,
            )
            return 2
        print(
            f"WARNING: {len(failures)} of {len(probe)} cells have no measured Δ and were "
            f"judged against min_span={args.min_span:g} and the 0.05 tolerance convention "
            f"(--allow-missing-delta). Their verdicts rest on constants, not on a measured "
            f"noise floor, and are not admissible evidence for or against parity:\n"
            f"{detail}",
            file=sys.stderr,
        )

    cells = compute_parity(
        args.store,
        args.base_model,
        args.tuned_model,
        deltas=deltas,
        **shared,
    )
    if args.json:
        print(json.dumps([asdict(c) for c in cells], indent=2))
    else:
        print(format_cells(cells))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ParityCell",
    "REQUIRED_MASKS",
    "compute_cell",
    "compute_deltas",
    "compute_parity",
    "format_cells",
    "orient",
    "read_means",
]
