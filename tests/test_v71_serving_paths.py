"""V71 fast serving paths must mirror the V70 harness path byte-for-byte."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddi.cache import DocCache
from ddi.canonical_cache import (
    CanonicalCache,
    ExecutionContract,
    model_config_identity,
    runtime_identity,
)
from scripts.v70_harness import (
    DocumentPayload,
    EncodedRequest,
    generate_independent_ids,
    independent_forward_from_entries,
    install_fixed_link_tokens,
)
from scripts.v71_serving_paths import PrecomputedRequest, generate_fast, ttft_fast
from tests.test_v70_harness import TinyTokenizer, tiny_model


def _canonical_contract(model):
    return ExecutionContract(
        weights_identity="tiny-v71-paths",
        adapters_identity="none",
        tokenizer_identity="tiny-v71-paths",
        model_config_identity=model_config_identity(model),
        runtime_identity=runtime_identity(model),
        backend="eager",
        dtype=str(model.dtype),
    )


def _setup(tmp_path):
    model = tiny_model()
    tokenizer = TinyTokenizer()
    state = install_fixed_link_tokens(model, tokenizer)
    model.eval()
    request = EncodedRequest(
        (DocumentPayload((2, 3, 4)), DocumentPayload((5, 6, 7, 8))), (9, 10)
    )
    contract = _canonical_contract(model)
    store = DocCache(tmp_path / "store")
    CanonicalCache(model, contract, store).documents(
        [torch.tensor([list(document.token_ids)]) for document in request.documents]
    )
    cached = CanonicalCache(model, contract, DocCache(tmp_path / "store"))
    entries = cached.documents(
        [torch.tensor([list(document.token_ids)]) for document in request.documents]
    )
    return model, state, request, entries


def test_fast_path_matches_v70_harness_ids_and_ttft(tmp_path):
    model, state, request, entries = _setup(tmp_path)
    pre = PrecomputedRequest(model, state.contract, request, k=5, max_answer_tokens=16)

    slow_ttft_out = independent_forward_from_entries(
        model, state.contract, request, entries, k=5
    )
    slow_first = int(slow_ttft_out.logits[0, -1].argmax())
    assert ttft_fast(model, request, entries, pre) == slow_first

    for budget in (1, 3, 8):
        slow_ids = generate_independent_ids(
            model, state.contract, request, entries, k=5,
            max_new_tokens=budget, eos_token_id=None,
        )
        fast_ids = generate_fast(
            model, request, entries, pre,
            max_new_tokens=budget, eos_token_id=None,
        )
        assert fast_ids == slow_ids


def test_precomputed_prefix_rows_are_budget_independent(tmp_path):
    model, state, request, _entries = _setup(tmp_path)
    pre_small = PrecomputedRequest(model, state.contract, request, k=5, max_answer_tokens=2)
    pre_big = PrecomputedRequest(model, state.contract, request, k=5, max_answer_tokens=16)
    prefix = pre_small.prefix_len
    assert pre_big.prefix_len == prefix
    assert torch.equal(pre_small.mask, pre_big.mask)
    assert torch.equal(pre_small.positions, pre_big.positions)


def test_step_rejects_out_of_budget(tmp_path):
    model, state, request, _entries = _setup(tmp_path)
    pre = PrecomputedRequest(model, state.contract, request, k=5, max_answer_tokens=2)
    try:
        pre.step(2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
