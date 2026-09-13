"""Tests for the per-document KV cache primitives (issue #23, Phase 4.2).

Runner end-to-end behaviour is exercised separately — these tests pin down the DocCache
guarantees the acceptance criterion leans on: content-only keying, atomic writes, and safe
concurrent access from multiple workers.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
import torch

from ddi.cache import CacheEntry, DocCache, assemble_dynamic_cache, compute_key


def _fake_entry(key: str, doc_len: int = 4, num_layers: int = 2) -> CacheEntry:
    """Small deterministic CacheEntry with distinguishable K/V per layer."""
    k = [torch.full((1, 2, doc_len, 8), float(i)) for i in range(num_layers)]
    v = [torch.full((1, 2, doc_len, 8), float(i) + 0.5) for i in range(num_layers)]
    return CacheEntry(
        key=key, k_per_layer=k, v_per_layer=v, doc_len=doc_len, layout="restart"
    )


def test_compute_key_stable() -> None:
    a = compute_key("Qwen/Qwen2.5-7B", torch.bfloat16, "hello world")
    b = compute_key("Qwen/Qwen2.5-7B", torch.bfloat16, "hello world")
    assert a == b
    assert len(a) == 16

    different_model = compute_key("meta-llama/Llama-3-8B", torch.bfloat16, "hello world")
    assert different_model != a

    different_dtype = compute_key("Qwen/Qwen2.5-7B", torch.float16, "hello world")
    assert different_dtype != a

    different_text = compute_key("Qwen/Qwen2.5-7B", torch.bfloat16, "hello world!")
    assert different_text != a


def test_cache_get_returns_none_on_miss(tmp_path: Path) -> None:
    cache = DocCache(tmp_path)
    assert cache.get("no-such-key") is None
    assert cache.has("no-such-key") is False
    assert cache.size() == 0


def test_cache_put_then_get_roundtrips(tmp_path: Path) -> None:
    cache = DocCache(tmp_path)
    entry = _fake_entry(key="abc123", num_layers=2)
    cache.put(entry)

    got = cache.get("abc123")
    assert got is not None
    assert got.key == entry.key
    assert got.doc_len == entry.doc_len
    assert got.layout == entry.layout
    assert len(got.k_per_layer) == 2
    for a, b in zip(got.k_per_layer, entry.k_per_layer, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(got.v_per_layer, entry.v_per_layer, strict=True):
        assert torch.equal(a, b)


def test_cache_rejects_payload_under_wrong_key(tmp_path: Path) -> None:
    cache = DocCache(tmp_path)
    cache.put(_fake_entry(key="original"))
    (tmp_path / "original.pt").rename(tmp_path / "requested.pt")
    with pytest.raises(ValueError, match="payload key"):
        cache.get("requested")


def test_cache_identity_and_token_digest_are_validated(tmp_path: Path) -> None:
    cache = DocCache(tmp_path)
    base = _fake_entry(key="identified")
    entry = CacheEntry(
        key=base.key,
        k_per_layer=base.k_per_layer,
        v_per_layer=base.v_per_layer,
        doc_len=base.doc_len,
        layout=base.layout,
        cache_identity="checkpoint-tokenizer-config",
        token_ids_sha256="tokens",
        tensor_dtype="torch.float32",
    )
    cache.put(entry)
    assert cache.get(
        entry.key,
        expected_identity=entry.cache_identity,
        expected_token_ids_sha256=entry.token_ids_sha256,
    ) is not None
    assert cache.get(entry.key, expected_identity="stale") is None
    assert cache.get(entry.key, expected_token_ids_sha256="different") is None


def test_cache_size_counts_files(tmp_path: Path) -> None:
    cache = DocCache(tmp_path)
    cache.put(_fake_entry(key="k1"))
    cache.put(_fake_entry(key="k2"))
    cache.put(_fake_entry(key="k3"))
    assert cache.size() == 3


def test_has_matches_get(tmp_path: Path) -> None:
    cache = DocCache(tmp_path)
    assert cache.has("k1") == (cache.get("k1") is not None)
    cache.put(_fake_entry(key="k1"))
    assert cache.has("k1") is True
    assert cache.has("k1") == (cache.get("k1") is not None)


def test_assemble_dynamic_cache_preserves_request_order() -> None:
    first = _fake_entry("first", doc_len=2)
    second = _fake_entry("second", doc_len=3)
    second = CacheEntry(
        key=second.key,
        k_per_layer=[tensor + 10 for tensor in second.k_per_layer],
        v_per_layer=[tensor + 10 for tensor in second.v_per_layer],
        doc_len=second.doc_len,
        layout=second.layout,
    )
    cache = assemble_dynamic_cache([second, first], torch.device("cpu"))
    assert cache.get_seq_length() == 5
    assert torch.equal(cache.key_cache[0][..., :3, :], second.k_per_layer[0])
    assert torch.equal(cache.key_cache[0][..., 3:, :], first.k_per_layer[0])


def test_assemble_dynamic_cache_requires_restart_entries() -> None:
    entry = _fake_entry("sequential")
    entry = CacheEntry(
        key=entry.key,
        k_per_layer=entry.k_per_layer,
        v_per_layer=entry.v_per_layer,
        doc_len=entry.doc_len,
        layout="sequential",
    )
    with pytest.raises(ValueError, match="layout"):
        assemble_dynamic_cache([entry], torch.device("cpu"))


def test_assemble_dynamic_cache_rejects_empty_sentinels() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        assemble_dynamic_cache(
            [CacheEntry("empty", [], [], 0, "restart")], torch.device("cpu")
        )


def test_atomic_write_via_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash mid-write must leave no partial file at the target key path."""
    cache = DocCache(tmp_path)

    import ddi.cache as cache_module

    def _boom(*_a: object, **_kw: object) -> None:
        raise RuntimeError("simulated disk failure mid-write")

    monkeypatch.setattr(cache_module.torch, "save", _boom)

    entry = _fake_entry(key="doomed")
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        cache.put(entry)

    target = tmp_path / "doomed.pt"
    tmp = tmp_path / "doomed.pt.tmp"
    assert not target.exists(), "target key file must not appear after a failed write"
    assert not tmp.exists(), ".tmp scratch file must be cleaned up after a failed write"


def _put_worker(cache_dir: str, key_prefix: str, n: int) -> None:
    cache = DocCache(Path(cache_dir))
    for i in range(n):
        cache.put(_fake_entry(key=f"{key_prefix}-{i:03d}"))


def test_cache_concurrent_writes(tmp_path: Path) -> None:
    """Two processes writing 20 distinct keys each must all land readable, no torn files."""
    n = 20
    ctx = mp.get_context("fork")
    p1 = ctx.Process(target=_put_worker, args=(str(tmp_path), "a", n))
    p2 = ctx.Process(target=_put_worker, args=(str(tmp_path), "b", n))
    p1.start()
    p2.start()
    p1.join(timeout=60)
    p2.join(timeout=60)
    assert p1.exitcode == 0
    assert p2.exitcode == 0

    cache = DocCache(tmp_path)
    assert cache.size() == 2 * n
    for prefix in ("a", "b"):
        for i in range(n):
            key = f"{prefix}-{i:03d}"
            got = cache.get(key)
            assert got is not None, f"missing key {key}"
            assert got.key == key
            assert len(got.k_per_layer) == 2


def _reference_cat(tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Inline pre-V65 assemble reference: per-entry .to(device), then one cat."""
    return torch.cat([tensor.to(device) for tensor in tensors], dim=-2)


def _synthetic_entries() -> list[CacheEntry]:
    torch.manual_seed(7)
    entries = []
    for index, doc_len in enumerate((5, 1, 9)):
        k = [torch.randn(1, 2, doc_len, 8) for _ in range(3)]
        v = [torch.randn(1, 2, doc_len, 8) for _ in range(3)]
        entries.append(
            CacheEntry(
                key=f"e{index}", k_per_layer=k, v_per_layer=v, doc_len=doc_len,
                layout="restart",
            )
        )
    return entries


def test_assemble_matches_reference_cat_on_cpu() -> None:
    entries = _synthetic_entries()
    device = torch.device("cpu")
    assembled = assemble_dynamic_cache(entries, device)
    for layer in range(3):
        expected_k = _reference_cat([e.k_per_layer[layer] for e in entries], device)
        expected_v = _reference_cat([e.v_per_layer[layer] for e in entries], device)
        assert torch.equal(assembled.key_cache[layer], expected_k)
        assert torch.equal(assembled.value_cache[layer], expected_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused staging triggers on CUDA only")
def test_assemble_fused_staging_bit_identical_to_reference_cat() -> None:
    device = torch.device("cuda")
    entries = _synthetic_entries()
    assembled = assemble_dynamic_cache(entries, device)
    for layer in range(3):
        expected_k = _reference_cat([e.k_per_layer[layer] for e in entries], device)
        expected_v = _reference_cat([e.v_per_layer[layer] for e in entries], device)
        assert assembled.key_cache[layer].device.type == "cuda"
        assert torch.equal(assembled.key_cache[layer], expected_k)
        assert torch.equal(assembled.value_cache[layer], expected_v)

    # Mixed CPU/device blocks also route through the pinned staging buffer and must
    # stay bit-identical to the reference cat of the original CPU tensors.
    first = entries[0]
    resident = CacheEntry(
        key=first.key,
        k_per_layer=[tensor.to(device) for tensor in first.k_per_layer],
        v_per_layer=[tensor.to(device) for tensor in first.v_per_layer],
        doc_len=first.doc_len,
        layout=first.layout,
    )
    mixed = assemble_dynamic_cache([resident, *entries[1:]], device)
    for layer in range(3):
        expected_k = _reference_cat([e.k_per_layer[layer] for e in entries], device)
        expected_v = _reference_cat([e.v_per_layer[layer] for e in entries], device)
        assert torch.equal(mixed.key_cache[layer], expected_k)
        assert torch.equal(mixed.value_cache[layer], expected_v)
