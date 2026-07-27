"""Evaluation metrics for DDI generation quality (issue #11, Phase 1.3).

SQuAD-style normalization plus exact match, token-F1, and answer presence over that
normalization; and a `-100`-masked labels tensor for answer-only cross-entropy under HF's
own `forward(labels=...)`. Definitions are frozen here as the single source of truth so
the old notebook variants cannot silently disagree with the harness.
"""

from __future__ import annotations

import re
import string

import torch

# SQuAD-style articles. Matched as whole tokens after punctuation strip so we do not eat
# substrings like "the" inside "theatre".
_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """SQuAD-style: lowercase, strip punctuation, drop articles, collapse whitespace."""
    lowered = text.lower()
    depunct = lowered.translate(_PUNCT_TABLE)
    tokens = [t for t in depunct.split() if t not in _ARTICLES]
    return _WS_RE.sub(" ", " ".join(tokens)).strip()


def _check_candidates(candidates: list[str]) -> None:
    if len(candidates) == 0:
        raise ValueError("candidates must be non-empty; a metric with no target is a bug")


def exact_match(pred: str, candidates: list[str]) -> float:
    """1.0 if any normalized candidate equals the normalized prediction, else 0.0."""
    _check_candidates(candidates)
    if len(pred) == 0:
        return 0.0
    npred = normalize(pred)
    return 1.0 if any(npred == normalize(c) for c in candidates) else 0.0


def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # SQuAD convention: if either side is empty after normalization, F1 is 1.0 only
        # when both are empty; here pred is guaranteed non-empty by the caller and an
        # empty gold means we cannot overlap with anything meaningful.
        return 0.0
    # Multiset overlap — repeated tokens count up to min(count_pred, count_gold), matching
    # the original SQuAD implementation rather than a set-intersection shortcut.
    from collections import Counter

    common = Counter(pred_tokens) & Counter(gold_tokens)
    matched = sum(common.values())
    if matched == 0:
        return 0.0
    precision = matched / len(pred_tokens)
    recall = matched / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, candidates: list[str]) -> float:
    """Max token-F1 (SQuAD formula) over candidates after normalization."""
    _check_candidates(candidates)
    if len(pred) == 0:
        return 0.0
    pred_tokens = normalize(pred).split()
    if len(pred_tokens) == 0:
        return 0.0
    return max(_f1(pred_tokens, normalize(c).split()) for c in candidates)


def answer_presence(pred: str, candidates: list[str]) -> float:
    """1.0 if any normalized candidate is a contiguous substring of the normalized pred."""
    _check_candidates(candidates)
    if len(pred) == 0:
        return 0.0
    npred = normalize(pred)
    for c in candidates:
        ncand = normalize(c)
        # An empty normalized candidate would trivially match every prediction; treat it
        # as a non-signal rather than a spurious hit.
        if len(ncand) > 0 and ncand in npred:
            return 1.0
    return 0.0


def make_answer_labels(
    input_ids: torch.Tensor, answer_start: int, answer_end: int
) -> torch.Tensor:
    """Build a `[batch, seq]` labels tensor with `-100` outside `[answer_start, answer_end)`.

    HF's `forward(labels=...)` computes cross-entropy over positions where the label is
    not `-100`, so this restricts the training signal to answer tokens only (§9).
    """
    if input_ids.dim() != 2:
        raise ValueError(f"expected [batch, seq] input_ids, got shape {tuple(input_ids.shape)}")
    seq_len = input_ids.shape[1]
    if answer_start < 0 or answer_end > seq_len:
        raise ValueError(
            f"answer range [{answer_start}, {answer_end}) out of bounds for seq_len={seq_len}"
        )
    if answer_start >= answer_end:
        raise ValueError(
            f"answer_start ({answer_start}) must be strictly less than answer_end ({answer_end})"
        )
    labels = torch.full_like(input_ids, -100, dtype=torch.long)
    labels[:, answer_start:answer_end] = input_ids[:, answer_start:answer_end].to(torch.long)
    return labels
