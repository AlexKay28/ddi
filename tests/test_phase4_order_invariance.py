"""Tests for the Phase 4.3 order-invariance runner (issue #24).

End-to-end runner behaviour needs a real model and is gated behind `DDI_INTEGRATION=1`.
The unit tests here cover the small, model-free parts that own the interesting invariants:
registration, deterministic permutation generation, and the last-token slice helper.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch


def test_runner_is_registered() -> None:
    """Side-effect import of the runners package must register `phase4_order_invariance`."""
    import ddi.runners  # noqa: F401 — force side-effect registration
    from ddi.cli import _RUNNERS

    assert "phase4_order_invariance" in _RUNNERS


def test_reordered_run_ids_are_deterministic() -> None:
    """Given a seed, the (seed + i)-derived permutations are stable across runs.

    The runner uses `np.random.default_rng(seed + i).permutation(N)` for the i-th
    permutation — this is what makes the "reference vs permutation" comparison
    reproducible from the run_id + seed alone.
    """
    n_docs = 5
    seed = 42
    n_permutations = 4

    def _gen() -> list[list[int]]:
        return [
            np.random.default_rng(seed + i).permutation(n_docs).tolist()
            for i in range(n_permutations - 1)
        ]

    a = _gen()
    b = _gen()
    assert a == b
    assert all(sorted(p) == list(range(n_docs)) for p in a)
    # And the seed actually matters — different seed → different permutations for at least one i.
    other = [
        np.random.default_rng((seed + 1000) + i).permutation(n_docs).tolist()
        for i in range(n_permutations - 1)
    ]
    assert a != other


def test_last_query_logits_helper_returns_correct_slice() -> None:
    """`_last_query_logits(logits, doc_total, query_len)` must return the last query token row."""
    from ddi.runners.phase4_order_invariance import _last_query_logits

    torch.manual_seed(0)
    batch, seq, vocab = 1, 10, 7
    logits = torch.randn(batch, seq, vocab)
    doc_total = 6
    query_len = 3
    # Last query row lives at index doc_total + query_len - 1 = 8.
    got = _last_query_logits(logits, doc_total, query_len)
    assert got.shape == (batch, vocab)
    assert torch.equal(got, logits[:, 8, :].cpu())


def test_last_query_logits_rejects_zero_query() -> None:
    from ddi.runners.phase4_order_invariance import _last_query_logits

    logits = torch.zeros(1, 4, 3)
    with pytest.raises(ValueError, match="query_len must be >= 1"):
        _last_query_logits(logits, doc_total=2, query_len=0)


def test_last_query_logits_rejects_out_of_bounds() -> None:
    from ddi.runners.phase4_order_invariance import _last_query_logits

    logits = torch.zeros(1, 4, 3)
    with pytest.raises(ValueError, match="out of bounds"):
        _last_query_logits(logits, doc_total=4, query_len=2)


def test_config_missing_mask_key_raises() -> None:
    from ddi.runners.phase4_order_invariance import _run

    with pytest.raises(ValueError, match="missing required key: mask"):
        _run({"dataset": "nq", "model": "irrelevant"}, "cfg", 0)


def test_config_missing_variant_raises() -> None:
    from ddi.runners.phase4_order_invariance import _run

    with pytest.raises(ValueError, match="missing required key: variant"):
        _run({"dataset": "nq", "model": "irrelevant", "mask": {}}, "cfg", 0)


def test_config_rejects_non_ddi_variant() -> None:
    from ddi.runners.phase4_order_invariance import _run

    with pytest.raises(ValueError, match="only supports 'ddi_qf' and 'ddi_bridge_k'"):
        _run(
            {
                "dataset": "nq",
                "model": "irrelevant",
                "mask": {"variant": "full"},
            },
            "cfg",
            0,
        )


def test_config_rejects_empty_layouts() -> None:
    from ddi.runners.phase4_order_invariance import _run

    with pytest.raises(ValueError, match="layouts must be a non-empty list"):
        _run(
            {
                "dataset": "nq",
                "model": "irrelevant",
                "mask": {"variant": "ddi_qf"},
                "layouts": [],
            },
            "cfg",
            0,
        )


def test_config_rejects_unknown_layout() -> None:
    from ddi.runners.phase4_order_invariance import _run

    with pytest.raises(ValueError, match="unknown layout"):
        _run(
            {
                "dataset": "nq",
                "model": "irrelevant",
                "mask": {"variant": "ddi_qf"},
                "layouts": ["bogus"],
            },
            "cfg",
            0,
        )


def test_config_rejects_too_few_permutations() -> None:
    from ddi.runners.phase4_order_invariance import _run

    with pytest.raises(ValueError, match="n_permutations must be >= 2"):
        _run(
            {
                "dataset": "nq",
                "model": "irrelevant",
                "mask": {"variant": "ddi_qf"},
                "n_permutations": 1,
            },
            "cfg",
            0,
        )


def test_config_bridge_k_requires_bridge_variant() -> None:
    from ddi.runners.phase4_order_invariance import _run

    with pytest.raises(ValueError, match="only valid for variant='ddi_bridge_k'"):
        _run(
            {
                "dataset": "nq",
                "model": "irrelevant",
                "mask": {"variant": "ddi_qf", "bridge_k": 4},
            },
            "cfg",
            0,
        )


def test_config_bridge_variant_requires_bridge_k() -> None:
    from ddi.runners.phase4_order_invariance import _run

    with pytest.raises(ValueError, match="requires mask.bridge_k"):
        _run(
            {
                "dataset": "nq",
                "model": "irrelevant",
                "mask": {"variant": "ddi_bridge_k"},
            },
            "cfg",
            0,
        )


@pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="integration test; set DDI_INTEGRATION=1 to enable",
)
def test_restart_is_order_invariant_on_tiny_model() -> None:
    """On a tiny model, `ddi_qf + restart` must produce identical logits under permutation."""
    from ddi.model import load_model
    from ddi.runners.phase4_order_invariance import _prefill_last_logits

    model, tokenizer = load_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
    model.eval()

    docs = ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india"]
    query_ids = tokenizer("What?", return_tensors="pt", add_special_tokens=False).input_ids
    query_len = query_ids.shape[-1]

    ref = _prefill_last_logits(
        model, tokenizer, docs, query_ids, query_len, "ddi_qf", None, "restart"
    )
    perm = _prefill_last_logits(
        model, tokenizer, list(reversed(docs)), query_ids, query_len, "ddi_qf", None, "restart"
    )
    assert torch.allclose(ref, perm, atol=1e-3)
