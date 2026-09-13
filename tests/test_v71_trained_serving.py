"""V71 trained-contract serving tests: pure glue and tiny-model identity."""

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
from scripts.run_v71_trained_serving import (
    K_LINKS,
    _MissCounter,
    _payloads,
    _trained_request,
    _trained_ttft,
)
from scripts.v70_harness import (
    assert_original_inputs,
    generate_independent_ids,
    install_fixed_link_tokens,
)
from tests.test_v70_harness import TinyTokenizer, tiny_model


def _canonical_contract(model):
    return ExecutionContract(
        weights_identity="tiny-v71",
        adapters_identity="none",
        tokenizer_identity="tiny-v71",
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
    docs = [torch.tensor([[2, 3, 4]]), torch.tensor([[5, 6, 7, 8]])]
    query = (9, 10)
    payloads = _payloads(docs)
    request = _trained_request(payloads, query)
    contract = _canonical_contract(model)
    return model, state, request, docs, contract, tmp_path


def test_payloads_and_request_pass_the_reserved_id_check(tmp_path):
    _model, state, request, _docs, _contract, _path = _setup(tmp_path)
    assert [len(document.token_ids) for document in request.documents] == [3, 4]
    assert request.query_token_ids == (9, 10)
    assert_original_inputs(state, request)


def test_miss_counter_counts_real_encodes_and_resets(tmp_path):
    _model, _state, _request, docs, contract, path = _setup(tmp_path)
    store = DocCache(path / "store")
    cache = CanonicalCache(_model, contract, store)
    counter = _MissCounter(cache)
    assert counter.reset() == 0
    cache.documents(docs)
    assert counter.reset() == 2
    cache.documents(docs)
    assert counter.reset() == 0


def test_cp_bp_generated_ids_are_byte_identical_through_store_roundtrip(tmp_path):
    model, state, request, docs, contract, path = _setup(tmp_path)
    store = DocCache(path / "store")
    CanonicalCache(model, contract, store).documents(docs)
    reopened = DocCache(path / "store")
    cached = CanonicalCache(model, contract, reopened)
    cached.encode_document = lambda _: (_ for _ in ()).throw(
        AssertionError("all-hit CP attempted an encode")
    )
    fresh = CanonicalCache(model, contract)
    for budget in (1, 3):
        cp_ids = generate_independent_ids(
            model, state.contract, request, cached.documents(docs),
            k=K_LINKS, max_new_tokens=budget, eos_token_id=None,
        )
        bp_ids = generate_independent_ids(
            model, state.contract, request, fresh.documents(docs),
            k=K_LINKS, max_new_tokens=budget, eos_token_id=None,
        )
        assert cp_ids == bp_ids


def test_trained_ttft_is_deterministic_and_integer(tmp_path):
    model, state, request, docs, contract, _path = _setup(tmp_path)
    entries = CanonicalCache(model, contract).documents(docs)
    first, _out = _trained_ttft(model, state.contract, request, entries)
    second, _out2 = _trained_ttft(model, state.contract, request, entries)
    assert isinstance(first, int) and first == second


def test_k_is_the_frozen_link_count():
    assert K_LINKS == 5
