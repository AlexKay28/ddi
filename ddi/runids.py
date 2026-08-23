"""The `run_id` grammar — one module, one parser per prefix (VIBEQUEUE-16, RUNIDS).

Every number this project publishes is a mean over rows in the append-only JSONL store
(`ddi.store.append`), and every row is keyed by a `run_id` string that is parsed
*positionally*. Before this module, three consumers re-split that string independently —
`ddi.aggregate._quality_parse_run_id`, `ddi.parity.read_means` and
`ddi.delta.compute_delta` — each re-deriving model/dataset/variant and each declaring its
own copy of `phase2_train_free:`. They did not agree: `read_means` accepted any row with
seven colon-separated segments and folded it into a mean, while the other two required the
fifth segment to be a well-formed ``seed=<int>``. A row that one of them averaged into a
published number, another silently dropped — and nothing anywhere compared them.

That class of drift has already cost this project once: `compute_delta` filtered on a
`phase1_baseline:` prefix nothing has ever written, so it matched zero rows and raised a
misleading "requires >= 3 seeds" against every real store, while its unit tests built that
prefix by hand and therefore agreed with the bug. A grammar duplicated across modules is a
grammar that is only as correct as the least-read copy of it.

**The shapes here are frozen, not designed.** Old rows exist in stores that are real
research data; a change that makes a historical row unreadable is a regression, not a
cleanup. What each producer writes (see `ddi/run.py`, `ddi/finetune.py`, `ddi/delta.py`):

    phase2_train_free:{model}:{dataset}:{variant}:seed={seed}:{sha8}:{record_id}
    phase3_timing:{model}:N={N}:L={L}:Q={Q}:{variant}:{backend}:seed={seed}:{sha8}
    phase4_cache:{model}:{mode}:N={N}:L={L}:seed={seed}:{sha8}:q{query_i}
    phase4_throughput:{model}:{mode}:{layout}:reuse={reuse}:seed={seed}:{sha8}
    phase4_restart:{model}:{dataset}:{tag}:{layout}:seed={seed}:{sha8}:{record_id}
    finetune:{model}:{tag}:seed={seed}:{sha8}:step={n}      (or ``:epoch={n}``)
    phase1_delta:{model}:{dataset}:metric={metric}:k={k}[:seed={seed}]   (not parsed here)

Rules every parser in this module follows:

- **Never raise.** A caller iterates a store holding several prefixes at once, so "this row
  is not mine" is the common case, not an error. Every ``parse`` returns ``None``.
- **Segment counts are minimums** (``len(parts) < N`` rejects). Record ids and reuse tags
  are free-form and a producer may yet append a segment; a historical row with an extra
  colon must keep parsing.
- **A field the historical producers never validated stays optional** rather than becoming
  a new reason to drop a row. `seed` is `int | None` on the timing / correctness /
  throughput shapes for exactly that reason: those three parsers never looked at it, so
  tightening it now could make an old row unreadable. Where a parser *did* require it
  (quality, restart, finetune) it stays required and typed `int`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "CorrectnessRunId",
    "FinetuneRunId",
    "QualityRunId",
    "RestartRunId",
    "ThroughputRunId",
    "TimingRunId",
    "parse_any",
]


def _int_field(segment: str, prefix: str) -> int | None:
    """``("N=13", "N=") -> 13``; ``None`` when the segment is not that field."""
    if not segment.startswith(prefix):
        return None
    try:
        return int(segment[len(prefix):])
    except ValueError:
        return None


@dataclass(frozen=True)
class QualityRunId:
    """``phase2_train_free:{model}:{dataset}:{variant}:seed={seed}:{sha8}:{record_id}``.

    Written by `ddi.run._quality_cell`, one row per (record, metric). Read by the quality
    table, by `ddi.parity.read_means` and by `ddi.delta.compute_delta` — the three sites
    that used to parse it separately.
    """

    PREFIX: ClassVar[str] = "phase2_train_free:"
    _MIN_SEGMENTS: ClassVar[int] = 7

    model: str
    dataset: str
    variant: str
    seed: int

    @classmethod
    def parse(cls, run_id: str) -> QualityRunId | None:
        parts = run_id.split(":")
        if len(parts) < cls._MIN_SEGMENTS or (parts[0] + ":") != cls.PREFIX:
            return None
        seed = _int_field(parts[4], "seed=")
        if seed is None:
            return None
        return cls(model=parts[1], dataset=parts[2], variant=parts[3], seed=seed)


@dataclass(frozen=True)
class RestartRunId:
    """``phase4_restart:{model}:{dataset}:{tag}:{layout}:seed={seed}:{sha8}:{record_id}``.

    Written by `ddi.run`'s restart-quality task. `tag` is the mask variant, `layout` is the
    axis under test (`sequential` baseline vs `restart`).
    """

    PREFIX: ClassVar[str] = "phase4_restart:"
    _MIN_SEGMENTS: ClassVar[int] = 8

    model: str
    dataset: str
    tag: str
    layout: str
    seed: int

    @classmethod
    def parse(cls, run_id: str) -> RestartRunId | None:
        parts = run_id.split(":")
        if len(parts) < cls._MIN_SEGMENTS or (parts[0] + ":") != cls.PREFIX:
            return None
        seed = _int_field(parts[5], "seed=")
        if seed is None:
            return None
        return cls(
            model=parts[1], dataset=parts[2], tag=parts[3], layout=parts[4], seed=seed
        )


@dataclass(frozen=True)
class TimingRunId:
    """``phase3_timing:{model}:N={N}:L={L}:Q={Q}:{variant}:{backend}:seed={seed}:{sha8}``.

    `seed` is `int | None`: the timing collector keys its cells off `provenance.seed`, and
    the parser this replaced never looked at the run_id's seed segment at all. Requiring it
    now would newly drop rows that have always been collected.
    """

    PREFIX: ClassVar[str] = "phase3_timing:"
    _MIN_SEGMENTS: ClassVar[int] = 9

    model: str
    N: int
    L: int
    Q: int
    variant: str
    backend: str
    seed: int | None

    @classmethod
    def parse(cls, run_id: str) -> TimingRunId | None:
        if not run_id.startswith(cls.PREFIX):
            return None
        parts = run_id.split(":")
        if len(parts) < cls._MIN_SEGMENTS:
            return None
        n = _int_field(parts[2], "N=")
        l_ = _int_field(parts[3], "L=")
        q = _int_field(parts[4], "Q=")
        if n is None or l_ is None or q is None:
            return None
        return cls(
            model=parts[1],
            N=n,
            L=l_,
            Q=q,
            variant=parts[5],
            backend=parts[6],
            seed=_int_field(parts[7], "seed="),
        )


@dataclass(frozen=True)
class CorrectnessRunId:
    """``phase4_cache:{model}:{mode}:N={N}:L={L}:seed={seed}:{sha8}:q{query_i}``.

    One row per query, so the correctness cell reduces with `max` (worst case) rather than
    a mean. `seed` is optional for the same reason as on `TimingRunId`.
    """

    PREFIX: ClassVar[str] = "phase4_cache:"
    _MIN_SEGMENTS: ClassVar[int] = 8

    model: str
    mode: str
    N: int
    L: int
    query_i: int
    seed: int | None

    @classmethod
    def parse(cls, run_id: str) -> CorrectnessRunId | None:
        if not run_id.startswith(cls.PREFIX):
            return None
        parts = run_id.split(":")
        if len(parts) < cls._MIN_SEGMENTS:
            return None
        n = _int_field(parts[3], "N=")
        l_ = _int_field(parts[4], "L=")
        if n is None or l_ is None:
            return None
        # The query index is read off the LAST segment, not a fixed position: it is the one
        # field appended after the sha, and reading it positionally would break the moment a
        # producer added a segment.
        query_i = _int_field(parts[-1], "q")
        if query_i is None:
            return None
        return cls(
            model=parts[1],
            mode=parts[2],
            N=n,
            L=l_,
            query_i=query_i,
            seed=_int_field(parts[5], "seed="),
        )


@dataclass(frozen=True)
class ThroughputRunId:
    """``phase4_throughput:{model}:{mode}:{layout}:reuse={reuse}:seed={seed}:{sha8}``.

    `layout` and `reuse` are the only experiment axes this run_id carries besides the mode,
    and `arm` joins them into the one axis the throughput table renders. ``top_docs``,
    ``docs_per_request`` and ``doc_token_len`` are NOT in the run_id, so two runs differing
    only in those still merge into one cell — fixing that means changing the run_id in
    `ddi/run.py`.

    Both halves of the arm are optional because the run_id is scanned, not indexed: a
    historical row that carries only one of them still gets a cell (named after the half it
    has) rather than being dropped, and a row carrying neither lands in `UNKNOWN_ARM` so it
    is never silently merged with the shaped ones.
    """

    PREFIX: ClassVar[str] = "phase4_throughput:"
    _MIN_SEGMENTS: ClassVar[int] = 7
    UNKNOWN_ARM: ClassVar[str] = "unknown"

    model: str
    mode: str
    layout: str
    reuse: str | None
    seed: int | None

    @property
    def arm(self) -> str:
        """``"{layout}/reuse={reuse}"``, or whichever half exists, or ``UNKNOWN_ARM``."""
        if self.layout and self.reuse is not None:
            return f"{self.layout}/reuse={self.reuse}"
        if self.layout:
            return self.layout
        if self.reuse is not None:
            return f"reuse={self.reuse}"
        return self.UNKNOWN_ARM

    @classmethod
    def parse(cls, run_id: str) -> ThroughputRunId | None:
        if not run_id.startswith(cls.PREFIX):
            return None
        parts = run_id.split(":")
        if len(parts) < cls._MIN_SEGMENTS:
            return None
        layout = parts[3] if not parts[3].startswith(("reuse=", "seed=")) else ""
        reuse_seg = next((seg for seg in parts[3:] if seg.startswith("reuse=")), None)
        seed_seg = next((seg for seg in parts[3:] if seg.startswith("seed=")), None)
        return cls(
            model=parts[1],
            mode=parts[2],
            layout=layout,
            reuse=None if reuse_seg is None else reuse_seg[len("reuse="):],
            seed=None if seed_seg is None else _int_field(seed_seg, "seed="),
        )


@dataclass(frozen=True)
class FinetuneRunId:
    """``finetune:{model}:{tag}:seed={seed}:{sha8}:step={n}`` or ``…:epoch={n}``.

    Exactly one of `step` / `epoch` is set; a tail segment that is neither makes the whole
    run_id unparseable, which is what keeps the loss table (steps) and the perplexity table
    (epochs) from reading each other's rows.
    """

    PREFIX: ClassVar[str] = "finetune:"
    _MIN_SEGMENTS: ClassVar[int] = 6

    model: str
    tag: str
    seed: int
    step: int | None
    epoch: int | None

    @classmethod
    def parse(cls, run_id: str) -> FinetuneRunId | None:
        parts = run_id.split(":")
        if len(parts) < cls._MIN_SEGMENTS or (parts[0] + ":") != cls.PREFIX:
            return None
        seed = _int_field(parts[3], "seed=")
        if seed is None:
            return None
        tail = parts[5]
        step = _int_field(tail, "step=")
        epoch = _int_field(tail, "epoch=")
        if step is None and epoch is None:
            return None
        return cls(model=parts[1], tag=parts[2], seed=seed, step=step, epoch=epoch)


# Prefix-keyed registry. Ordered longest-prefix-first so a future prefix that extends an
# existing one cannot be shadowed by it.
_PARSERS: tuple[type, ...] = (
    QualityRunId,
    RestartRunId,
    TimingRunId,
    CorrectnessRunId,
    ThroughputRunId,
    FinetuneRunId,
)

RunId = (
    QualityRunId
    | RestartRunId
    | TimingRunId
    | CorrectnessRunId
    | ThroughputRunId
    | FinetuneRunId
)


def parse_any(run_id: str) -> RunId | None:
    """Parse a run_id of any known shape, or ``None``.

    Prefixes are disjoint, so at most one parser can match; this is a convenience for
    callers that walk a whole store (and the hook the cross-parser agreement test uses).
    Rows written by `ddi.delta.record_delta` (`phase1_delta:`) have no consumer that parses
    them back and deliberately have no shape here.
    """
    for parser in _PARSERS:
        if run_id.startswith(parser.PREFIX):
            return parser.parse(run_id)
    return None
