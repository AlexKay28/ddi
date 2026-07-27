"""Tests for the mask builder (issue #3, Phase 0.2).

Acceptance-criterion contract: for every variant at small T, `build_mask` must equal a naive
triple-loop reference written independently of the fast path. `_naive_reference` below is that
reference — it reads as a direct restatement of the spec in `ddi/masks.py`'s docstring and the
variant table in the README. Do NOT share helpers with `ddi.masks`; the whole point is that a
bug in the fast path cannot be masked by a shared subroutine.
"""

from __future__ import annotations

import math

import pytest
import torch

from ddi.masks import build_mask, to_additive


def _naive_reference(
    doc_lengths: list[int],
    query_len: int,
    answer_len: int,
    variant: str,
    bridge_k: int | None = None,
) -> torch.Tensor:
    """Triple-loop restatement of the spec. Kept intentionally verbose."""
    doc_total = sum(doc_lengths)
    total = doc_total + query_len + answer_len
    n_docs = len(doc_lengths)

    # Per-position: region name and document-block id (only meaningful in the doc region).
    region = ["?"] * total
    doc_id = [-1] * total
    off = 0
    for b, length in enumerate(doc_lengths):
        for k in range(length):
            region[off + k] = "doc"
            doc_id[off + k] = b
        off += length
    for p in range(doc_total, doc_total + query_len):
        region[p] = "query"
    for p in range(doc_total + query_len, total):
        region[p] = "answer"

    # N=1 collapse: qf/strict/bridge_k reduce to full — there is no cross-doc leakage to gate.
    collapse_to_full = variant in ("ddi_qf", "ddi_strict", "ddi_bridge_k") and n_docs <= 1
    effective = "full" if collapse_to_full else variant

    last_query_row = doc_total + query_len - 1  # unused when query_len == 0
    first_bridge_row = doc_total + query_len - (bridge_k or 0)

    allow = torch.zeros((total, total), dtype=torch.bool)
    for i in range(total):
        for j in range(total):
            ri, rj = region[i], region[j]

            if effective == "full":
                allow[i, j] = j <= i
                continue

            if ri == "doc" and rj == "doc":
                allow[i, j] = doc_id[i] == doc_id[j] and j <= i

            elif ri == "query" and rj == "doc":
                if effective == "ddi_qf":
                    allow[i, j] = True
                elif effective == "ddi_strict":
                    allow[i, j] = i == last_query_row
                elif effective == "ddi_bridge_k":
                    allow[i, j] = i >= first_bridge_row
                elif effective == "blind":
                    allow[i, j] = False

            elif ri == "query" and rj == "query":
                allow[i, j] = j <= i

            elif ri == "answer" and rj == "doc":
                allow[i, j] = effective != "blind"

            elif ri == "answer" and rj == "query":
                allow[i, j] = True

            elif ri == "answer" and rj == "answer":
                allow[i, j] = j <= i

            # All other (ri, rj) combinations are strictly upper-triangular (col after row)
            # and therefore blocked by causality: doc→query, doc→answer, query→answer.
    return allow


# ---------------------------------------------------------------------------
# Fixtures — a couple of small shapes covering the interesting geometry.
# ---------------------------------------------------------------------------

FIXTURES = [
    # (doc_lengths, query_len, answer_len)
    ([2, 3, 2], 3, 2),
    ([1, 4, 1, 2], 2, 3),
    ([3, 3], 4, 1),
    ([1], 4, 1),  # N=1
    ([5], 3, 2),  # N=1, another shape
    ([], 3, 2),  # N=0 (no info_contexts, e.g. nq) — must collapse to plain causal
]


@pytest.mark.parametrize("doc_lengths,query_len,answer_len", FIXTURES)
def test_full_matches_reference(doc_lengths, query_len, answer_len):
    fast = build_mask(doc_lengths, query_len, answer_len, "full")
    ref = _naive_reference(doc_lengths, query_len, answer_len, "full")
    assert fast.equal(ref)


@pytest.mark.parametrize("doc_lengths,query_len,answer_len", FIXTURES)
def test_ddi_qf_matches_reference(doc_lengths, query_len, answer_len):
    fast = build_mask(doc_lengths, query_len, answer_len, "ddi_qf")
    ref = _naive_reference(doc_lengths, query_len, answer_len, "ddi_qf")
    assert fast.equal(ref)


@pytest.mark.parametrize("doc_lengths,query_len,answer_len", FIXTURES)
def test_ddi_strict_matches_reference(doc_lengths, query_len, answer_len):
    fast = build_mask(doc_lengths, query_len, answer_len, "ddi_strict")
    ref = _naive_reference(doc_lengths, query_len, answer_len, "ddi_strict")
    assert fast.equal(ref)


@pytest.mark.parametrize("doc_lengths,query_len,answer_len", FIXTURES)
def test_blind_matches_reference(doc_lengths, query_len, answer_len):
    fast = build_mask(doc_lengths, query_len, answer_len, "blind")
    ref = _naive_reference(doc_lengths, query_len, answer_len, "blind")
    assert fast.equal(ref)


@pytest.mark.parametrize("doc_lengths,query_len,answer_len", FIXTURES)
def test_ddi_bridge_k_matches_reference(doc_lengths, query_len, answer_len):
    for k in {1, 2, query_len}:
        if not (1 <= k <= query_len):
            continue
        fast = build_mask(doc_lengths, query_len, answer_len, "ddi_bridge_k", bridge_k=k)
        ref = _naive_reference(doc_lengths, query_len, answer_len, "ddi_bridge_k", bridge_k=k)
        assert fast.equal(ref), f"mismatch at k={k}, shape={doc_lengths},{query_len},{answer_len}"


# ---------------------------------------------------------------------------
# Cross-variant equivalences called out in the spec.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc_lengths,query_len,answer_len", FIXTURES)
def test_bridge_k_endpoints(doc_lengths, query_len, answer_len):
    """k=1 ≡ ddi_strict, k=query_len ≡ ddi_qf, per the interpolation definition."""
    if query_len < 1:
        pytest.skip("bridge_k needs query_len >= 1")
    k1 = build_mask(doc_lengths, query_len, answer_len, "ddi_bridge_k", bridge_k=1)
    strict = build_mask(doc_lengths, query_len, answer_len, "ddi_strict")
    assert k1.equal(strict)

    kq = build_mask(doc_lengths, query_len, answer_len, "ddi_bridge_k", bridge_k=query_len)
    qf = build_mask(doc_lengths, query_len, answer_len, "ddi_qf")
    assert kq.equal(qf)


@pytest.mark.parametrize("doc_lengths", [[1], [4], [7]])
def test_n1_collapses_to_full(doc_lengths):
    """The Phase 0.5 correctness gate at the mask level: single-doc → indistinguishable from full."""
    query_len, answer_len = 3, 2
    full = build_mask(doc_lengths, query_len, answer_len, "full")
    for variant in ("ddi_qf", "ddi_strict"):
        assert build_mask(doc_lengths, query_len, answer_len, variant).equal(full)
    for k in (1, 2, query_len):
        assert build_mask(
            doc_lengths, query_len, answer_len, "ddi_bridge_k", bridge_k=k
        ).equal(full)


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_query_len_zero():
    doc_lengths = [2, 3]
    for variant in ("full", "ddi_qf", "ddi_strict", "blind"):
        fast = build_mask(doc_lengths, 0, 2, variant)
        ref = _naive_reference(doc_lengths, 0, 2, variant)
        assert fast.equal(ref), f"query_len=0 mismatch for {variant}"


def test_answer_len_zero():
    doc_lengths = [2, 3]
    for variant in ("full", "ddi_qf", "ddi_strict", "blind"):
        fast = build_mask(doc_lengths, 3, 0, variant)
        ref = _naive_reference(doc_lengths, 3, 0, variant)
        assert fast.equal(ref), f"answer_len=0 mismatch for {variant}"


def test_single_document():
    fast = build_mask([4], 2, 1, "ddi_qf")
    ref = _naive_reference([4], 2, 1, "ddi_qf")
    assert fast.equal(ref)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        build_mask([1], 1, 1, "nope")  # type: ignore[arg-type]


def test_bridge_k_required_for_bridge_variant():
    with pytest.raises(ValueError, match="requires bridge_k"):
        build_mask([2, 2], 3, 1, "ddi_bridge_k")


def test_bridge_k_out_of_range():
    with pytest.raises(ValueError, match="bridge_k must satisfy"):
        build_mask([2, 2], 3, 1, "ddi_bridge_k", bridge_k=0)
    with pytest.raises(ValueError, match="bridge_k must satisfy"):
        build_mask([2, 2], 3, 1, "ddi_bridge_k", bridge_k=4)


def test_bridge_k_rejected_for_non_bridge_variant():
    with pytest.raises(ValueError, match="only valid for variant='ddi_bridge_k'"):
        build_mask([2, 2], 3, 1, "ddi_qf", bridge_k=2)


def test_zero_length_doc_rejected():
    with pytest.raises(ValueError, match="doc_lengths"):
        build_mask([2, 0, 3], 2, 1, "full")


def test_negative_lengths_rejected():
    with pytest.raises(ValueError):
        build_mask([2, 2], -1, 1, "full")
    with pytest.raises(ValueError):
        build_mask([2, 2], 1, -1, "full")


# ---------------------------------------------------------------------------
# to_additive
# ---------------------------------------------------------------------------


def test_to_additive_values_and_dtype():
    allow = torch.tensor([[True, False], [False, True]])
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        add = to_additive(allow, dtype=dtype)
        assert add.dtype == dtype
        assert add[0, 0].item() == 0.0
        assert add[1, 1].item() == 0.0
        assert math.isinf(add[0, 1].item()) and add[0, 1].item() < 0
        assert math.isinf(add[1, 0].item()) and add[1, 0].item() < 0


def test_to_additive_fully_blocked_row_is_representable_in_bf16():
    """A row that blocks everything must be storable as bf16 without producing NaN in the mask."""
    allow = torch.zeros((4, 4), dtype=torch.bool)  # nothing allowed
    add = to_additive(allow, dtype=torch.bfloat16)
    assert add.dtype == torch.bfloat16
    assert torch.isneginf(add).all()
    assert not torch.isnan(add).any()


def test_to_additive_rejects_non_bool():
    with pytest.raises(ValueError, match="expected bool"):
        to_additive(torch.zeros((2, 2), dtype=torch.float32))


def test_to_additive_matches_build_mask():
    """End-to-end: build a mask, convert, sanity-check both branches agree on shape/positions."""
    allow = build_mask([2, 3], 2, 2, "ddi_qf")
    add = to_additive(allow, dtype=torch.float32)
    assert add.shape == allow.shape
    assert (add == 0.0).equal(allow)
    assert torch.isneginf(add).equal(~allow)
