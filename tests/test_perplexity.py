"""Tests for `ddi.perplexity` (issue #120).

Pure-unit tests only: a hand-rolled fake model returns deterministic logits
so we can compute the expected perplexity in closed form and check the
implementation matches to floating-point tolerance. No HF download.
"""

from __future__ import annotations

import math
import types

import pytest
import torch

from ddi.perplexity import compute_perplexity  # noqa: I001
from ddi.prompt import build_query_text

# ---------------------------------------------------------------------------
# Minimal fakes: tokenizer + model that behave like the ones _quality_cell sees.
# ---------------------------------------------------------------------------


class _FakeTokenizerCall:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = torch.tensor([ids], dtype=torch.long)


class _FakeTokenizer:
    """Whitespace-split tokenizer: each token maps to (hash % vocab_size) + 1."""

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None, add_special_tokens=None):  # noqa: ARG002
        ids = [(abs(hash(w)) % (self.vocab_size - 1)) + 1 for w in text.split()]
        if not ids:
            ids = [1]
        return _FakeTokenizerCall(ids)

    def apply_chat_template(self, dialog, tokenize=False, add_generation_prompt=False):  # noqa: ARG002
        return " ".join(m["content"] for m in dialog)


class _FakeModel(torch.nn.Module):
    """Returns a deterministic-per-position logits tensor of shape [1, T, V].

    Logits are ``log_probs = log_softmax(random_but_seeded)`` so `log p(target)` is
    computable directly by the test. The model's ``__call__`` ignores every
    kwarg except ``input_ids`` — that is exactly what we want since we're
    testing perplexity, not attention.
    """

    def __init__(self, vocab_size: int = 32, seed: int = 0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self._seed = seed
        # Register a dummy parameter so `.device` and `.dtype` are defined.
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(vocab_size=vocab_size)

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dummy.dtype

    def _logits_for(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[-1]
        # Deterministic per-(seed, T) so the test can be reproduced.
        g = torch.Generator(device="cpu")
        g.manual_seed(self._seed * 1_000_003 + T)
        raw = torch.randn(1, T, self.vocab_size, generator=g)
        return raw

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):  # noqa: ARG002
        logits = self._logits_for(input_ids)
        return types.SimpleNamespace(logits=logits)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def _record(docs: list[str], question: str, answer: str, rid: str = "r0") -> dict:
    return {
        "id": rid,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
    }


def _expected_perplexity(model: _FakeModel, tok: _FakeTokenizer, records: list[dict]) -> float:
    """Re-derive the expected perplexity by hand from the fake model's logits.

    Mirrors what compute_perplexity should do: build the same input_ids, take
    the shift-by-one logits for the answer window, cross-entropy over the
    answer token targets.
    """
    total_nll = 0.0
    total_tokens = 0
    for record in records:
        docs = record["info_contexts"]
        # Go through the shared builder rather than re-deriving the prompt: the query
        # block now carries the answer-format instruction (ddi.prompt), and an oracle
        # that rebuilt the old bare-question text would be testing a prompt the
        # harness no longer produces.
        query = build_query_text(tok, record)
        answer = record["dialog"][-1]["content"]

        doc_ids = [tok(d).input_ids for d in docs]
        q_ids = tok(query).input_ids
        a_ids = tok(answer).input_ids
        a_len = a_ids.shape[-1]
        full_ids = torch.cat(doc_ids + [q_ids, a_ids], dim=-1)
        logits = model._logits_for(full_ids).float()
        T = full_ids.shape[-1]
        # Match compute_perplexity: labels are answer positions [T-a_len, T); the
        # predictive logits live at [T-a_len-1, T-1) after the standard shift.
        lo = T - a_len - 1
        hi = T - 1
        pred_logits = logits[0, lo:hi, :]
        target = full_ids[0, T - a_len:T]
        per_tok = torch.nn.functional.cross_entropy(
            pred_logits, target, reduction="sum"
        )
        total_nll += float(per_tok.item())
        total_tokens += int(target.numel())
    return math.exp(total_nll / total_tokens)


def test_perplexity_matches_hand_computed_value() -> None:
    tok = _FakeTokenizer(vocab_size=32)
    model = _FakeModel(vocab_size=32, seed=17)
    records = [
        _record(["alpha bravo", "charlie delta"], "what?", "echo foxtrot", rid="r0"),
        _record(["golf hotel"], "again?", "india", rid="r1"),
    ]

    got = compute_perplexity(model, tok, records, mask_variant="full", layout="sequential")
    expected = _expected_perplexity(model, tok, records)
    assert got == pytest.approx(expected, rel=1e-4)


def test_perplexity_skips_records_without_answers() -> None:
    tok = _FakeTokenizer(vocab_size=32)
    model = _FakeModel(vocab_size=32, seed=1)
    records = [
        _record(["alpha"], "?", "beta", rid="r0"),
        # No possible_answers -> must be skipped, not raise.
        {
            "id": "r_bad",
            "info_contexts": ["gamma"],
            "dialog": [
                {"role": "user", "content": "?"},
                {"role": "assistant", "content": "delta"},
            ],
            "possible_answers": [],
        },
    ]
    got = compute_perplexity(model, tok, records, mask_variant="full")
    expected = _expected_perplexity(model, tok, [records[0]])
    assert got == pytest.approx(expected, rel=1e-4)


def test_perplexity_empty_records_raises() -> None:
    tok = _FakeTokenizer(vocab_size=32)
    model = _FakeModel(vocab_size=32, seed=1)
    with pytest.raises(ValueError, match="no answer tokens"):
        compute_perplexity(model, tok, [], mask_variant="full")


def test_perplexity_ddi_qf_variant_coerces_to_full_on_no_docs() -> None:
    """No-doc records: DDI variants collapse to `full` — matches _quality_cell semantics."""
    tok = _FakeTokenizer(vocab_size=32)
    model = _FakeModel(vocab_size=32, seed=5)
    records = [_record([], "?", "beta", rid="r0")]
    got = compute_perplexity(model, tok, records, mask_variant="ddi_qf")
    expected = _expected_perplexity(model, tok, records)
    assert got == pytest.approx(expected, rel=1e-4)
