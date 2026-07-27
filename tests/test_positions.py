"""Tests for the position-ID builder (issue #22, Phase 4.1).

`build_positions` returns either HF's default sequential layout or the DDI restart layout
in which each document's positions start at 0. The restart layout is the primitive that
unlocks C4 caching, so its shape and offset conventions are load-bearing enough to pin
down explicitly.
"""

from __future__ import annotations

import pytest
import torch

from ddi.positions import build_positions


# ---------------------------------------------------------------------------
# Sequential layout — HF's default, kept as an explicit option
# ---------------------------------------------------------------------------


def test_sequential_matches_arange() -> None:
    doc_lengths = [3, 5, 2]
    query_len, answer_len = 4, 2
    total = sum(doc_lengths) + query_len + answer_len

    got = build_positions(doc_lengths, query_len, answer_len, layout="sequential")

    assert torch.equal(got, torch.arange(total, dtype=torch.long))
    assert got.dtype == torch.long
    assert got.shape == (total,)


def test_sequential_no_docs_matches_arange() -> None:
    got = build_positions([], query_len=4, answer_len=2, layout="sequential")
    assert torch.equal(got, torch.arange(6, dtype=torch.long))


# ---------------------------------------------------------------------------
# Restart layout — the whole point of this module
# ---------------------------------------------------------------------------


def test_restart_per_doc_starts_at_zero() -> None:
    doc_lengths = [3, 5, 2]
    query_len, answer_len = 4, 2

    got = build_positions(doc_lengths, query_len, answer_len, layout="restart")

    # Each document restarts at 0; query+answer continues from max(doc_lengths) = 5.
    expected = torch.tensor(
        [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 5, 6, 7, 8, 9, 10], dtype=torch.long
    )
    assert torch.equal(got, expected)


def test_restart_no_docs() -> None:
    got = build_positions([], query_len=4, answer_len=2, layout="restart")
    # No docs → query starts at 0 (max of an empty list falls back to 0 by design).
    assert torch.equal(got, torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long))


def test_restart_single_doc_still_offsets_query() -> None:
    """With a single doc of length L, the query starts at L (== max(doc_lengths))."""
    got = build_positions([4], query_len=2, answer_len=1, layout="restart")
    assert torch.equal(got, torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long))


def test_restart_uses_max_not_last_doc_length() -> None:
    """Query offset is max(doc_lengths), regardless of whether the last doc is the longest."""
    # Last doc is short; the middle doc is longest at 6 → query starts at 6.
    got = build_positions([3, 6, 2], query_len=2, answer_len=0, layout="restart")
    expected = torch.tensor(
        [0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 6, 7], dtype=torch.long
    )
    assert torch.equal(got, expected)


def test_restart_answer_only_no_query() -> None:
    got = build_positions([2, 3], query_len=0, answer_len=2, layout="restart")
    # Answer picks up right where the query block would have started.
    assert torch.equal(got, torch.tensor([0, 1, 0, 1, 2, 3, 4], dtype=torch.long))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_layout_raises() -> None:
    with pytest.raises(ValueError, match="unknown layout"):
        build_positions([2, 3], query_len=1, answer_len=1, layout="whatever")  # type: ignore[arg-type]


def test_negative_doc_length_raises() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        build_positions([3, 0, 5], query_len=1, answer_len=1, layout="restart")


def test_negative_query_len_raises() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        build_positions([2], query_len=-1, answer_len=0, layout="sequential")


def test_negative_answer_len_raises() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        build_positions([2], query_len=0, answer_len=-1, layout="restart")
