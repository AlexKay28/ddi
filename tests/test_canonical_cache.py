"""CPU-only canonical B/C gates, not combined A or checkpoint equivalence tests."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from ddi.cache import DocCache, assemble_dynamic_cache
from ddi.canonical_cache import (
    STORE_VALIDATION_POLICY,
    CanonicalCache,
    ExecutionContract,
    _LoadedEntryLru,
    model_config_identity,
    runtime_identity,
)


class FakeModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = []

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        past = kwargs.get("past_key_values")
        cache = DynamicCache() if past is None else past
        values = ids.to(self.dtype)[:, None, :, None].expand(1, 1, -1, 8).contiguous()
        for i in range(2):
            cache.update(values + i, values + i + 1, i)
        score = cache.value_cache[0].sum() + kwargs["position_ids"].sum()
        logits = ids[..., None].to(self.dtype) + torch.arange(64) + score
        return SimpleNamespace(logits=logits, past_key_values=cache)


@pytest.fixture(params=["mock", "qwen-fp32", "qwen-bf16"])
def model(request):
    config = Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
        pad_token_id=0, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    if request.param == "mock":
        return FakeModel(config).eval()
    with torch.random.fork_rng():
        torch.manual_seed(60)
        result = Qwen2ForCausalLM(config).cpu().eval()
    return result.to(torch.bfloat16 if request.param == "qwen-bf16" else torch.float32)


def contract_for(model):
    return ExecutionContract(
        weights_identity="random-seed-60-test-only", adapters_identity="none",
        tokenizer_identity="test-token-vocabulary-v1",
        model_config_identity=model_config_identity(model),
        runtime_identity=runtime_identity(model),
        backend="eager", dtype=str(model.dtype),
    )


def test_contract_admits_sdpa_backend_and_keys_identity():
    base = dict(
        weights_identity="w", adapters_identity="none",
        tokenizer_identity="t", model_config_identity="c", runtime_identity="r",
        dtype="torch.float32",
    )
    eager = ExecutionContract(backend="eager", **base)
    sdpa = ExecutionContract(backend="sdpa", **base)
    assert sdpa.identity != eager.identity
    with pytest.raises(ValueError, match="unsupported"):
        replace(eager, backend="fa2")


def ids(*values):
    return torch.tensor([values], dtype=torch.long)


def assert_kv_equal(a, b):
    assert a.key == b.key
    for left, right in zip(a.k_per_layer + a.v_per_layer, b.k_per_layer + b.v_per_layer,
                           strict=True):
        assert torch.equal(left, right)


def test_repeat_and_live_cold_persistence(model, tmp_path, monkeypatch):
    store = DocCache(tmp_path)
    api = CanonicalCache(model, contract_for(model), store)
    first = api.encode_document(ids(2, 3, 4))
    assert_kv_equal(first, api.encode_document(ids(2, 3, 4)))
    encode = Mock(wraps=api.encode_document)
    put = Mock(wraps=store.put)
    monkeypatch.setattr(api, "encode_document", encode)
    monkeypatch.setattr(store, "put", put)
    cold = api.documents([ids(2, 3, 4)])[0]
    assert encode.call_count == 1
    persisted = put.call_args.args[0]
    assert cold is not persisted
    for live, cpu in zip(cold.k_per_layer, persisted.k_per_layer, strict=True):
        assert live.device == model.device
        assert cpu.device.type == "cpu"
        assert live.data_ptr() != cpu.data_ptr()
        assert not live.requires_grad
    warm = api.documents([ids(2, 3, 4)])[0]
    assert encode.call_count == 1
    assert_kv_equal(cold, warm)


def test_request_matrix_exact(model, tmp_path, monkeypatch):
    contract = contract_for(model)
    fresh = CanonicalCache(model, contract)
    cached = CanonicalCache(model, contract, DocCache(tmp_path))
    encode = Mock(wraps=cached.encode_document)
    monkeypatch.setattr(cached, "encode_document", encode)
    a, b, c = ids(2, 3), ids(4, 5, 6), ids(7, 8, 9, 10)
    cases = [
        ([a, b], ids(11, 12), 2),       # cold
        ([a, b], ids(11, 12), 0),       # warm
        ([b, a], ids(11, 12), 0),       # reorder
        ([a, c], ids(11, 12), 1),       # partial and changed neighbor length
        ([a, a, b], ids(11, 12), 0),    # duplicate occupies two physical blocks
        ([a, b], ids(13), 0),           # changed query
        ([ids(2, 14), b], ids(13), 1),   # one exact ID changed
    ]
    original = fresh.encode_document(a)
    for docs, query, misses in cases:
        before = encode.call_count
        expected = fresh.request(docs, query)
        actual = cached.request(docs, query)
        assert torch.equal(expected.logits, actual.logits)
        assert encode.call_count - before == misses
        assert fresh.request(docs, query, max_new_tokens=3) == cached.request(
            docs, query, max_new_tokens=3
        )
        assert_kv_equal(original, cached.documents([a])[0])
    reopened = CanonicalCache(model, contract, DocCache(tmp_path))
    assert torch.equal(reopened.request([a, b], ids(11)).logits,
                       fresh.request([a, b], ids(11)).logits)
    assert cached.request([a], ids(11), max_new_tokens=0) == []
    first_token = cached.request([a], ids(11), max_new_tokens=1)[0]
    assert cached.request([a], ids(11), max_new_tokens=4, eos_token_id=first_token) == [
        first_token
    ]


def test_logical_positions_vs_physical_slots(model, tmp_path, monkeypatch):
    calls = []
    forward = model.forward

    def record(**kwargs):
        calls.append(kwargs)
        return forward(**kwargs)

    monkeypatch.setattr(model, "forward", record)
    api = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    a, b = ids(2, 3), ids(4, 5, 6)
    api.request([a, b, a], ids(7, 8), max_new_tokens=2)
    assert calls[0]["position_ids"].tolist() == [[0, 1]]
    assert calls[0]["cache_position"].tolist() == [0, 1]
    assert calls[1]["position_ids"].tolist() == [[0, 1, 2]]
    query = calls[2]
    assert query["position_ids"].tolist() == [[3, 4]]
    assert query["cache_position"].tolist() == [7, 8]
    assert query["attention_mask"].shape == (1, 1, 2, 9)
    assert query["attention_mask"][0, 0, 0, :8].eq(0).all()
    assert query["attention_mask"][0, 0, 0, 8] <= torch.finfo(query["attention_mask"].dtype).min
    assert calls[3]["position_ids"].tolist() == [[5]]
    assert calls[3]["cache_position"].tolist() == [9]
    entries = api.documents([b, a, a])
    assembled = assemble_dynamic_cache(entries, model.device)
    assert assembled.get_seq_length() == 7
    assert torch.equal(assembled.key_cache[0][..., 3:5, :], entries[1].k_per_layer[0])
    assert torch.equal(assembled.key_cache[0][..., 5:7, :], entries[2].k_per_layer[0])


@pytest.mark.parametrize("field", list(ExecutionContract.__dataclass_fields__))
def test_contract_invalidation(model, tmp_path, field):
    contract = contract_for(model)
    api = CanonicalCache(model, contract, DocCache(tmp_path))
    original = api.documents([ids(2, 3)])[0]
    changes = {field: getattr(contract, field) + "-changed"}
    if field == "dtype":
        dtype = "torch.float32" if model.dtype != torch.float32 else "torch.bfloat16"
        changed = replace(contract, dtype=dtype)
        assert changed.identity != contract.identity
        with pytest.raises(ValueError, match="contract"):
            CanonicalCache(model, changed, api.store)
    elif field in {"backend", "layout", "system_prefix", "padding", "encoding"}:
        with pytest.raises(ValueError, match="unsupported"):
            replace(contract, **changes)
    else:
        changed = replace(contract, **changes)
        assert changed.identity != contract.identity
        if field in {"model_config_identity", "runtime_identity"}:
            with pytest.raises(ValueError, match="contract"):
                CanonicalCache(model, changed, api.store)
        else:
            other = CanonicalCache(model, changed, api.store).documents([ids(2, 3)])[0]
            assert other.key != original.key
            assert api.store.size() == 2


@pytest.mark.parametrize("damage", [
    "key", "cache_identity", "token_ids_sha256", "layout", "tensor_dtype", "doc_len",
    "schema_version", "schema_string", "missing_meta", "extra_meta", "missing_k",
    "layers", "heads", "batch", "length", "head_dim", "rank", "dtype", "value_shape",
    "nan", "inf", "not_tensor", "bool_len",
])
def test_persisted_payload_rejected(model, tmp_path, damage):
    api = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    entry = api.documents([ids(2, 3)])[0]
    path = tmp_path / f"{entry.key}.pt"
    blob = torch.load(path, weights_only=True)
    if damage in {"key", "cache_identity", "token_ids_sha256", "layout", "tensor_dtype",
                  "validated_by", "validated_sha256"}:
        blob["meta"][damage] = "wrong"
    elif damage in {"doc_len", "schema_version"}:
        blob["meta"][damage] += 1
    elif damage == "schema_string":
        blob["meta"]["schema_version"] = "60"
    elif damage == "bool_len":
        blob["meta"]["doc_len"] = True
    elif damage == "missing_meta":
        del blob["meta"]["tensor_dtype"]
    elif damage == "extra_meta":
        blob["meta"]["extra"] = 1
    elif damage == "missing_k":
        del blob["k"]
    elif damage == "layers":
        blob["k"].pop()
    elif damage in {"heads", "batch", "length", "head_dim", "rank"}:
        shapes = {"heads": (1, 2, 2, 8), "batch": (2, 1, 2, 8), "length": (1, 1, 3, 8),
                  "head_dim": (1, 1, 2, 4), "rank": (1, 2, 8)}
        blob["k"][0] = torch.zeros(shapes[damage], dtype=model.dtype)
    elif damage == "value_shape":
        blob["v"][0] = blob["v"][0][..., :1, :]
    elif damage == "dtype":
        blob["k"][0] = blob["k"][0].double()
    elif damage == "not_tensor":
        blob["k"][0] = None
    elif damage in {"nan", "inf"}:
        # V65: finite-value integrity is checked at store time, so a marker-carrying
        # payload is trusted at load and post-store tampering is not re-scanned. Strip
        # the markers to model a pre-V65 payload, which keeps the full load-time check.
        blob["v"][0][0, 0, 0, 0] = float(damage)
        for field in ("validated_by", "validated_sha256"):
            del blob["meta"][field]
    else:  # pragma: no cover - parametrize guard
        raise AssertionError(f"unknown damage {damage}")
    torch.save(blob, path)
    # The in-process LRU tier would otherwise serve the untouched in-memory copy; these
    # damages target the disk payload, so resolve them through a cold memory tier.
    api._loaded = _LoadedEntryLru()
    with pytest.raises(ValueError, match="payload"):
        api.request([ids(2, 3)], ids(4))


@pytest.mark.parametrize("bad", [
    [], torch.empty(1, 0, dtype=torch.long), torch.tensor([1, 2]),
    torch.ones(2, 2, dtype=torch.long), torch.ones(1, 2), ids(-1), ids(64), ids(2, 0),
])
def test_input_validation(model, tmp_path, bad):
    api = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    with pytest.raises(ValueError):
        api.request([ids(2), bad], ids(3))
    with pytest.raises(ValueError):
        api.request([ids(2)], bad)
    assert api.store.size() == 0


def test_eval_and_execution_validation(model, tmp_path):
    contract = contract_for(model)
    api = CanonicalCache(model, contract, DocCache(tmp_path))
    with pytest.raises(ValueError):
        replace(contract, weights_identity="")
    model.train()
    with pytest.raises(ValueError, match="eval"):
        api.request([ids(2)], ids(3))
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            api.request([ids(2)], ids(3))
    model.config.rope_theta += 1
    with pytest.raises(ValueError, match="contract"):
        api.request([ids(2)], ids(3))
    model.config.rope_theta -= 1
    for kwargs in ({"max_new_tokens": -1}, {"max_new_tokens": True},
                   {"max_new_tokens": 1.5}, {"eos_token_id": -1}):
        with pytest.raises(ValueError):
            api.request([ids(2)], ids(3), **kwargs)
    with pytest.raises(ValueError):
        api.request([], ids(3))
    with pytest.raises(ValueError, match="position"):
        api.request([ids(2)], ids(3), max_new_tokens=128)
    assert api.store.size() == 0
    assert asdict(contract)["system_prefix"] == "none"


def test_fresh_reencodes_and_duplicate_cold_hits(model, tmp_path, monkeypatch):
    fresh = CanonicalCache(model, contract_for(model))
    encode = Mock(wraps=fresh.encode_document)
    monkeypatch.setattr(fresh, "encode_document", encode)
    fresh.request([ids(2), ids(2)], ids(3))
    fresh.request([ids(2), ids(2)], ids(3))
    assert encode.call_count == 4
    cached = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    encode = Mock(wraps=cached.encode_document)
    monkeypatch.setattr(cached, "encode_document", encode)
    entries = cached.documents([ids(2), ids(2)])
    assert encode.call_count == 1
    assert cached.store.size() == 1
    assert_kv_equal(*entries)


@pytest.mark.parametrize("field,value", [
    ("model_type", "gemma"), ("rope_scaling", {"rope_type": "dynamic", "factor": 2.0}),
    ("use_sliding_window", True), ("_attn_implementation", "sdpa"),
])
def test_unsupported_model_rejected(model, field, value):
    contract = contract_for(model)
    setattr(model.config, field, value)
    with pytest.raises(ValueError):
        CanonicalCache(model, contract)


def test_bad_live_payload_not_persisted(model, tmp_path, monkeypatch):
    api = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    forward = model.forward

    def corrupt(**kwargs):
        out = forward(**kwargs)
        out.past_key_values.value_cache[0][0, 0, 0, 0] = float("nan")
        return out

    monkeypatch.setattr(model, "forward", corrupt)
    with pytest.raises(ValueError, match="payload"):
        api.documents([ids(2)])
    assert api.store.size() == 0


def test_runtime_drift_rejected_on_warm_hit(model, tmp_path, monkeypatch):
    api = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    api.documents([ids(2)])
    threads = torch.get_num_threads()
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads + 1)
    with pytest.raises(ValueError, match="runtime"):
        api.request([ids(2)], ids(3))


def test_store_time_validation_marker(model, tmp_path):
    """V65: stored payloads carry the store-time validation record in their metadata."""
    api = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    entry = api.documents([ids(2, 3)])[0]
    assert entry.validated_by == STORE_VALIDATION_POLICY
    assert entry.validated_sha256
    blob = torch.load(tmp_path / f"{entry.key}.pt", weights_only=True)
    assert blob["meta"]["validated_by"] == entry.validated_by
    assert blob["meta"]["validated_sha256"] == entry.validated_sha256
    # The validation record is a deterministic function of (policy, key, geometry,
    # dtype): a cold instance reading the same store recomputes the same marker, so the
    # load-time metadata equality check binds it without extra cost.
    reopened = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    assert reopened.documents([ids(2, 3)])[0].validated_sha256 == entry.validated_sha256
    fresh = CanonicalCache(model, contract_for(model))
    assert fresh.encode_document(ids(2, 3)).validated_sha256 == entry.validated_sha256


def test_store_time_integrity_policy_trusts_marked_payloads(model, tmp_path):
    """V65: finite-value checking moved to store time; load keeps cheap checks only."""
    api = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    key = api.documents([ids(2, 3)])[0].key
    path = tmp_path / f"{key}.pt"
    blob = torch.load(path, weights_only=True)
    marked = {**blob, "v": [tensor.clone() for tensor in blob["v"]]}
    marked["v"][0][0, 0, 0, 0] = float("nan")
    torch.save(marked, path)
    # A marker-carrying payload is trusted at load: no finite-value rescan, so post-store
    # tampering is served as-is (cache directories are trusted; the store-time gate is
    # the only writer of these payloads).
    served = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    tampered = served.documents([ids(2, 3)])[0]
    assert torch.isnan(tampered.v_per_layer[0][0, 0, 0, 0])
    # The same damage on a marker-less (pre-V65) payload keeps the full load-time check.
    legacy = {
        **marked,
        "meta": {
            name: value for name, value in marked["meta"].items()
            if name not in {"validated_by", "validated_sha256"}
        },
    }
    torch.save(legacy, path)
    cold = CanonicalCache(model, contract_for(model), DocCache(tmp_path))
    with pytest.raises(ValueError, match="payload"):
        cold.documents([ids(2, 3)])


def test_documents_lru_hit_miss_and_eviction(model, tmp_path, monkeypatch):
    contract = contract_for(model)
    disk_loads = []
    real_load = torch.load

    def counting_load(path, *args, **kwargs):
        disk_loads.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    a, b = ids(2, 3), ids(4, 5)  # equal doc_len: equal LRU byte cost

    api = CanonicalCache(model, contract, DocCache(tmp_path))
    entry_a = api.documents([a])[0]
    entry_b = api.documents([b])[0]
    assert disk_loads == []  # cold requests encode; the disk is never read
    for got, expected in zip(
        api.documents([b, a, b]), [entry_b, entry_a, entry_b], strict=True
    ):
        assert_kv_equal(got, expected)
    assert disk_loads == []  # repeats, reorders and duplicates all hit memory

    # A budget below one entry disables the memory tier: every resolve hits the disk.
    monkeypatch.setattr(api, "_loaded", _LoadedEntryLru(byte_budget=1))
    api.documents([a])
    assert disk_loads == [f"{entry_a.key}.pt"]
    api.documents([a])
    assert disk_loads == [f"{entry_a.key}.pt", f"{entry_a.key}.pt"]

    # A one-entry budget evicts least-recently-used on each new insert.
    disk_loads.clear()
    cost = sum(
        tensor.numel() * tensor.element_size()
        for tensor in entry_a.k_per_layer + entry_a.v_per_layer
    )
    one_slot = CanonicalCache(model, contract, DocCache(tmp_path))
    one_slot._loaded = _LoadedEntryLru(byte_budget=cost)
    one_slot.documents([a, b, a])
    # a fills the slot, b evicts it, the trailing a evicts b back.
    assert disk_loads == [
        f"{entry_a.key}.pt", f"{entry_b.key}.pt", f"{entry_a.key}.pt",
    ]
    one_slot.documents([b])
    assert len(disk_loads) == 4  # b was evicted by the trailing a
    one_slot.documents([b])
    assert len(disk_loads) == 4  # most recent insert (b) is resident again
