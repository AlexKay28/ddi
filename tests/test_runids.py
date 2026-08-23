"""Tests for ddi/runids.py — the one place the run_id grammar lives.

Two things are being protected here.

1. **Historical rows keep parsing.** The store is append-only research data going back to
   Phase 1; a shape that stops parsing is a published number that quietly disappears. Every
   shape below is the f-string a producer in `ddi/run.py`, `ddi/finetune.py` or
   `ddi/delta.py` actually writes, copied deliberately rather than paraphrased.

2. **The three former parse sites agree.** `ddi.aggregate`'s quality table,
   `ddi.parity.read_means` and `ddi.delta.compute_delta` each used to re-split the
   phase2_train_free run_id independently, and they did NOT agree: `read_means` accepted any
   row with seven colon-separated segments and folded it into a published mean, while the
   other two required a well-formed ``seed=<int>``. `test_three_parse_sites_agree_row_by_row`
   drives all three end-to-end over the same store and asserts they accept the same rows —
   the failure the duplication risked, rather than the parser's own unit behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.aggregate import collect_table
from ddi.delta import compute_delta
from ddi.parity import read_means
from ddi.runids import (
    CorrectnessRunId,
    FinetuneRunId,
    QualityRunId,
    RestartRunId,
    ThroughputRunId,
    TimingRunId,
    parse_any,
)

# ---- the shapes producers actually write ---------------------------------------------------


def test_quality_shape_from_run_quality_cell() -> None:
    # ddi/run.py: f"phase2_train_free:{model}:{dataset}:{tag}:seed={seed}:{sha[:8]}" + f":{id}"
    parsed = QualityRunId.parse(
        "phase2_train_free:Qwen/Qwen2.5-7B:hotpot_qa:ddi_qf:seed=3:abcd1234:hotpot-0007"
    )
    assert parsed == QualityRunId(
        model="Qwen/Qwen2.5-7B", dataset="hotpot_qa", variant="ddi_qf", seed=3
    )


def test_quality_model_may_contain_a_slash_and_variant_an_equals() -> None:
    """`{model}` is an HF repo id and `{tag}` can be `ddi_qf_bridge_k=2`. Neither has a colon."""
    parsed = QualityRunId.parse(
        "phase2_train_free:results/tuned:musique:ddi_qf_bridge_k=2:seed=0:abcd1234:r0"
    )
    assert parsed is not None
    assert parsed.model == "results/tuned"
    assert parsed.variant == "ddi_qf_bridge_k=2"


def test_quality_record_id_containing_a_colon_still_parses() -> None:
    """Record ids come from the dataset, so the segment count is a minimum, not an equality."""
    parsed = QualityRunId.parse(
        "phase2_train_free:M:hotpot_qa:full:seed=0:abcd1234:rec:with:colons"
    )
    assert parsed is not None
    assert (parsed.dataset, parsed.variant, parsed.seed) == ("hotpot_qa", "full", 0)


def test_timing_shape_from_run_timing_cell() -> None:
    # ddi/run.py: f"phase3_timing:{model}:N={N}:L={L}:Q={Q}:{tag}:{backend}:seed={s}:{sha[:8]}"
    parsed = TimingRunId.parse(
        "phase3_timing:Qwen/Qwen2.5-7B:N=13:L=8:Q=16:ddi_qf:sdpa:seed=1:abcd1234"
    )
    assert parsed == TimingRunId(
        model="Qwen/Qwen2.5-7B", N=13, L=8, Q=16, variant="ddi_qf", backend="sdpa", seed=1
    )


def test_correctness_shape_from_run_cache_correctness() -> None:
    # ddi/run.py: f"phase4_cache:{model}:{mode}:N={N}:L={L}:seed={seed}:{sha[:8]}:q{i}"
    parsed = CorrectnessRunId.parse(
        "phase4_cache:Qwen/Qwen2.5-7B:ddi_cache:N=5:L=2:seed=0:abcd1234:q7"
    )
    assert parsed == CorrectnessRunId(
        model="Qwen/Qwen2.5-7B", mode="ddi_cache", N=5, L=2, query_i=7, seed=0
    )


def test_throughput_shape_from_run_cache_throughput() -> None:
    # ddi/run.py: f"phase4_throughput:{model}:{mode}:{layout}:reuse={tag}:seed={s}:{sha[:8]}"
    parsed = ThroughputRunId.parse(
        "phase4_throughput:Qwen/Qwen2.5-7B:ddi_cache:restart:reuse=zipf(a=1.1):seed=0:abcd1234"
    )
    assert parsed is not None
    assert (parsed.model, parsed.mode, parsed.layout) == (
        "Qwen/Qwen2.5-7B", "ddi_cache", "restart"
    )
    assert parsed.reuse == "zipf(a=1.1)"
    assert parsed.arm == "restart/reuse=zipf(a=1.1)"


def test_throughput_arm_falls_back_but_never_merges_with_a_shaped_one() -> None:
    """A row carrying neither half gets its own `unknown` cell rather than being dropped."""
    unknown = ThroughputRunId.parse("phase4_throughput:M:ddi_cache:seed=0:abcd1234:x:y")
    assert unknown is not None and unknown.arm == ThroughputRunId.UNKNOWN_ARM
    reuse_only = ThroughputRunId.parse(
        "phase4_throughput:M:ddi_cache:reuse=uniform:seed=0:abcd1234:x"
    )
    assert reuse_only is not None and reuse_only.arm == "reuse=uniform"


def test_restart_shape_from_run_restart_quality() -> None:
    # ddi/run.py: f"phase4_restart:{model}:{ds}:{tag}:{layout}:seed={s}:{sha[:8]}:{record_id}"
    parsed = RestartRunId.parse(
        "phase4_restart:Qwen/Qwen2.5-7B:hotpot_qa:ddi_qf:restart:seed=2:abcd1234:rec-9"
    )
    assert parsed == RestartRunId(
        model="Qwen/Qwen2.5-7B", dataset="hotpot_qa", tag="ddi_qf", layout="restart", seed=2
    )


def test_finetune_step_and_epoch_shapes() -> None:
    # ddi/finetune.py: f"finetune:{model}:ddi_qf:seed={seed}:{sha[:8]}" + ":step=N" / ":epoch=E"
    step = FinetuneRunId.parse("finetune:results/tuned:ddi_qf:seed=0:abcd1234:step=150")
    assert step == FinetuneRunId(
        model="results/tuned", tag="ddi_qf", seed=0, step=150, epoch=None
    )
    epoch = FinetuneRunId.parse("finetune:results/tuned:ddi_qf:seed=0:abcd1234:epoch=2")
    assert epoch == FinetuneRunId(
        model="results/tuned", tag="ddi_qf", seed=0, step=None, epoch=2
    )


# ---- non-matching rows return None, never raise ---------------------------------------------


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "no_colons_at_all",
        "phase1_delta:M:hotpot_qa:metric=em:k=2",  # written by ddi.delta.record_delta
        "phase1_baseline:M:nq:seed=0:abcd1234:rec0",  # the prefix nothing ever wrote
        "some_other_prefix:M:hotpot_qa:full:seed=0:abc:r3",
        "phase2_train_free:M:hotpot_qa:full:seed=0",  # too few segments
        "phase2_train_free:M:hotpot_qa:full:seedX:abcd1234:r0",  # seed segment malformed
        "phase2_train_free:M:hotpot_qa:full:seed=NaN:abcd1234:r0",
        "phase3_timing:M:N=x:L=2:Q=16:full:sdpa:seed=0:abcd1234",
        "phase4_cache:M:mode:N=1:L=2:seed=0:abcd1234:notq",
        "finetune:M:ddi_qf:seed=0:abcd1234:other=1",
    ],
)
def test_every_parser_returns_none_and_never_raises(run_id: str) -> None:
    """Callers iterate stores holding several prefixes, so "not mine" is the common case."""
    for cls in (
        QualityRunId, RestartRunId, TimingRunId,
        CorrectnessRunId, ThroughputRunId, FinetuneRunId,
    ):
        assert cls.parse(run_id) is None, cls.__name__
    assert parse_any(run_id) is None


def test_prefixes_are_disjoint_so_parse_any_cannot_be_ambiguous() -> None:
    prefixes = [
        QualityRunId.PREFIX, RestartRunId.PREFIX, TimingRunId.PREFIX,
        CorrectnessRunId.PREFIX, ThroughputRunId.PREFIX, FinetuneRunId.PREFIX,
    ]
    assert len(set(prefixes)) == len(prefixes)
    for a in prefixes:
        for b in prefixes:
            assert a == b or not a.startswith(b)


# ---- the grammar is declared once ------------------------------------------------------------


def test_run_id_prefixes_are_declared_only_in_runids() -> None:
    """`phase2_train_free:` was a string literal in three modules at once.

    That is what let the three parsers drift: each one owned its own copy of the prefix AND
    its own copy of the segment rules. This test fails the moment a fourth copy appears.
    """
    pkg = Path(__file__).resolve().parent.parent / "ddi"
    literals = [
        '"phase2_train_free:"', '"phase3_timing:"', '"phase4_cache:"',
        '"phase4_throughput:"', '"phase4_restart:"', '"finetune:"',
    ]
    # Non-vacuity: the literals must be findable where they DO live, or this test passes by
    # matching nothing the day someone reformats runids.py.
    runids_src = (pkg / "runids.py").read_text(encoding="utf-8")
    assert [lit for lit in literals if lit not in runids_src] == []

    offenders: dict[str, list[str]] = {}
    for path in sorted(pkg.glob("*.py")):
        if path.name == "runids.py":
            continue
        text = path.read_text(encoding="utf-8")
        hits = [lit for lit in literals if lit in text]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}, (
        f"run_id prefixes re-declared outside ddi/runids.py: {offenders}. "
        "Import the dataclass instead — that is the whole point of the module."
    )


# ---- the three former parse sites agree, end to end -----------------------------------------

_MODEL = "Qwen/Qwen2.5-7B"
_DATASET = "hotpot_qa"


def _write(store: Path, rows: list[tuple[str, float, int]]) -> None:
    with store.open("w", encoding="utf-8") as f:
        for run_id, value, seed in rows:
            f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "metric": "em",
                        "value": value,
                        "provenance": {"seed": seed, "git_sha": "abcd1234", "dirty": False},
                    }
                )
                + "\n"
            )


def _baseline_rows() -> list[tuple[str, float, int]]:
    """Three well-formed seeds. `compute_delta` needs >= 3 to return at all."""
    return [
        (f"phase2_train_free:{_MODEL}:{_DATASET}:full:seed={s}:abcd1234:r0", 0.5, s)
        for s in (0, 1, 2)
    ]


# (candidate run_id, provenance seed, is this row well-formed for the quality grammar?)
_CANDIDATES = [
    (f"phase2_train_free:{_MODEL}:{_DATASET}:full:seed=0:abcd1234:r1", 0, True),
    (f"phase2_train_free:{_MODEL}:{_DATASET}:full:seed=0:abcd1234:rec:with:colon", 0, True),
    (f"phase2_train_free:{_MODEL}:{_DATASET}:full:seed=0", 0, False),
    (f"phase2_train_free:{_MODEL}:{_DATASET}:full:seedX:abcd1234:r1", 0, False),
    (f"phase2_train_free:{_MODEL}:{_DATASET}:full:seed=NaN:abcd1234:r1", 0, False),
    (f"phase2_train_free:{_MODEL}:{_DATASET}:full::abcd1234:r1", 0, False),
    (f"phase1_baseline:{_MODEL}:{_DATASET}:full:seed=0:abcd1234:r1", 0, False),
    (f"phase4_restart:{_MODEL}:{_DATASET}:full:restart:seed=0:abcd1234:r1", 0, False),
]


@pytest.mark.parametrize(("candidate", "prov_seed", "well_formed"), _CANDIDATES)
def test_three_parse_sites_agree_row_by_row(
    tmp_path: Path, candidate: str, prov_seed: int, well_formed: bool
) -> None:
    """Each site must accept the candidate row iff the others do.

    Observed end to end rather than through the parser, because the parser agreeing with
    itself proves nothing: what broke before was that three call sites each applied their own
    extra rules on top. The candidate carries a value of 1.0 against a baseline of 0.5, so
    "was it counted" is visible in each site's own output.
    """
    store = tmp_path / "runs.jsonl"
    _write(store, _baseline_rows() + [(candidate, 1.0, prov_seed)])

    table = collect_table({"kind": "quality", "metric": "em"}, store)
    aggregate_took_it = table["backing_rows"] != len(_baseline_rows())

    means = read_means(store)
    parity_took_it = means[(_MODEL, _DATASET, "full", "em")] != pytest.approx(0.5)

    delta = compute_delta(store, _MODEL, _DATASET, "em")
    delta_took_it = delta.per_seed_means[0] != pytest.approx(0.5)

    assert aggregate_took_it == parity_took_it == delta_took_it == well_formed, (
        f"{candidate!r}: aggregate={aggregate_took_it} parity={parity_took_it} "
        f"delta={delta_took_it}, expected all {well_formed}"
    )


def test_delta_alone_still_drops_a_row_whose_provenance_seed_disagrees(tmp_path: Path) -> None:
    """The run_id grammar is shared; `ddi.delta`'s seed cross-check is deliberately not.

    A row whose `provenance.seed` disagrees with its run_id's `seed=` field parses fine — it
    is well-formed, just unresolvable — and only Δ, which groups BY seed, has to drop it.
    Unifying the parser must not have quietly unified this away.
    """
    store = tmp_path / "runs.jsonl"
    poison = f"phase2_train_free:{_MODEL}:{_DATASET}:full:seed=0:abcd1234:poison"
    _write(store, _baseline_rows() + [(poison, 1.0, 1)])

    assert QualityRunId.parse(poison) is not None
    assert read_means(store)[(_MODEL, _DATASET, "full", "em")] != pytest.approx(0.5)
    assert compute_delta(store, _MODEL, _DATASET, "em").per_seed_means == {
        0: 0.5, 1: 0.5, 2: 0.5
    }
