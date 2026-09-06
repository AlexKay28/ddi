"""Per-document KV cache (issue #23, Phase 4.2).

C4's win: with `layout=restart` (Phase 4.1) each document's KV block is a pure function of its
content, so the block can be computed once and reused across every request that references the
same document. This module is the disk-backed store that keys per-document K/V by content and
serves them back on hit.

Cache entries are keyed by `sha256(model_id, dtype, doc_text)[:16]`. Including model_id + dtype
prevents cross-model / cross-precision collisions (RoPE-encoded KVs from Qwen bf16 must not be
handed back on a Llama fp16 query).

Storage layout and the atomic-write contract are stated where they are implemented, on
`DocCache` and `DocCache.put`.

The `kind: cache task: correctness` cell in `ddi/run.py` compares query logits from assembled
cache entries with fresh restart-layout DDI recomputation. Behavioral generation equivalence
is covered separately because bf16 execution paths need not produce numerically identical
logits.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from ddi.masks import build_mask, to_additive
from ddi.positions import Layout, build_positions


@dataclass(frozen=True)
class CacheEntry:
    """Per-document KV block. `layout` is stored so callers can refuse cross-layout hits."""

    key: str
    k_per_layer: list[torch.Tensor]
    v_per_layer: list[torch.Tensor]
    doc_len: int
    layout: str


def compute_key(model_id: str, dtype: torch.dtype, doc_text: str) -> str:
    """Deterministic 16-hex-char content key. Ordering: model_id \\0 dtype \\0 doc_text.

    The \\0 separator prevents collisions between, e.g., ("Qwen", "bf16", "foo") and
    ("Qwenbf16", "", "foo"). str(dtype) is stable across torch versions for the dtypes we use.
    """
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\0")
    h.update(str(dtype).encode("utf-8"))
    h.update(b"\0")
    h.update(doc_text.encode("utf-8"))
    return h.hexdigest()[:16]


class DocCache:
    """Directory-of-files KV cache: one `<key>.pt` per document under a caller-supplied
    directory, each a `torch.save`-serialised dict
    `{"k": [layer_tensors...], "v": [layer_tensors...], "meta": {...}}`. Writes are atomic
    (`.tmp` + `os.replace`); see `put`.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def _lock_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.lock"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> CacheEntry | None:
        path = self._path(key)
        if not path.exists():
            return None
        # WHY weights_only=False: we serialise our own dict (tensors + str meta), not
        # untrusted third-party checkpoints. weights_only=True refuses arbitrary dicts.
        blob: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        # Fail loudly on structural damage: a torn file is a bug, not a cache miss, so a
        # truncated payload must raise rather than degrade into a silent recompute.
        if not isinstance(blob, dict) or "k" not in blob or "v" not in blob or "meta" not in blob:
            raise ValueError(f"cache file {path} is malformed: missing k/v/meta keys")
        meta = blob["meta"]
        return CacheEntry(
            key=meta["key"],
            k_per_layer=blob["k"],
            v_per_layer=blob["v"],
            doc_len=meta["doc_len"],
            layout=meta["layout"],
        )

    def put(self, entry: CacheEntry) -> None:
        target = self._path(entry.key)
        tmp = target.with_suffix(target.suffix + ".tmp")
        # WHY per-key flock: two workers computing the same document must not corrupt each
        # other's writes even inside the .tmp → os.replace dance. The lock file is a separate
        # sentinel so we don't recursively lock the target itself.
        lock_path = self._lock_path(entry.key)
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            payload = {
                "k": entry.k_per_layer,
                "v": entry.v_per_layer,
                "meta": {
                    "key": entry.key,
                    "doc_len": entry.doc_len,
                    "layout": entry.layout,
                },
            }
            # WHY .tmp + os.replace: writes to the target path must be atomic so a concurrent
            # reader never sees a partially serialised file. torch.save can be interrupted
            # mid-stream; if the interruption hits the .tmp, we clean it up and the target is
            # untouched. os.replace is atomic on POSIX and Windows.
            try:
                torch.save(payload, tmp)
                os.replace(tmp, target)
            except BaseException:
                if tmp.exists():
                    tmp.unlink()
                raise

    def size(self) -> int:
        return sum(1 for _ in self.cache_dir.glob("*.pt"))


def assemble_dynamic_cache(
    entries: list[CacheEntry], device: torch.device
) -> DynamicCache:
    """Concatenate restart-layout document K/V blocks in request order."""
    if not entries:
        raise ValueError("at least one cache entry is required")
    if any(entry.layout != "restart" for entry in entries):
        raise ValueError("document KV reuse requires restart-layout cache entries")
    layer_count = len(entries[0].k_per_layer)
    if layer_count == 0 or len(entries[0].v_per_layer) != layer_count:
        raise ValueError("cache entries must contain matching non-empty K/V layers")

    layers = []
    for layer in range(layer_count):
        keys = []
        values = []
        reference_shape = None
        for entry in entries:
            if len(entry.k_per_layer) != layer_count or len(entry.v_per_layer) != layer_count:
                raise ValueError("cache entries have different layer counts")
            key = entry.k_per_layer[layer]
            value = entry.v_per_layer[layer]
            if key.shape != value.shape or key.ndim != 4:
                raise ValueError("each cache layer must contain shape-matched rank-4 K/V")
            if key.shape[-2] != entry.doc_len:
                raise ValueError("cache entry doc_len does not match its K/V sequence length")
            shape = (*key.shape[:-2], key.shape[-1])
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                raise ValueError("cache entries have incompatible K/V shapes")
            keys.append(key.to(device))
            values.append(value.to(device))
        layers.append((torch.cat(keys, dim=-2), torch.cat(values, dim=-2)))
    return DynamicCache.from_legacy_cache(tuple(layers))


def forward_query_with_doc_cache(
    model: PreTrainedModel,
    query_ids: torch.Tensor,
    entries: list[CacheEntry],
) -> Any:
    """Prefill a query against real assembled per-document K/V blocks."""
    doc_lengths = [entry.doc_len for entry in entries]
    query_len = int(query_ids.shape[-1])
    doc_total = sum(doc_lengths)
    cache = assemble_dynamic_cache(entries, model.device)
    positions = build_positions(doc_lengths, query_len, 0, "restart")
    query_positions = positions[doc_total:].unsqueeze(0).to(model.device)
    cache_position = torch.arange(
        doc_total, doc_total + query_len, dtype=torch.long, device=model.device
    )
    allow = build_mask(doc_lengths, query_len, 0, "ddi_qf")
    additive = (
        to_additive(allow[doc_total:, :], dtype=model.dtype)
        .to(model.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    with torch.no_grad():
        return model(
            input_ids=query_ids.to(model.device),
            attention_mask=additive,
            position_ids=query_positions,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )


def generate_tokens_with_doc_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    query_ids: torch.Tensor,
    entries: list[CacheEntry],
    max_new_tokens: int,
) -> list[int]:
    """Greedy-decode from independently cached restart-layout documents."""
    doc_lengths = [entry.doc_len for entry in entries]
    query_len = int(query_ids.shape[-1])
    prefix_len = sum(doc_lengths) + query_len
    allow = build_mask(doc_lengths, query_len, max_new_tokens, "ddi_qf")
    additive = to_additive(allow, dtype=model.dtype).to(model.device)
    positions = build_positions(doc_lengths, query_len, 0, "restart")
    decode_start = int(positions[-1].item()) + 1
    out = forward_query_with_doc_cache(model, query_ids, entries)
    past = out.past_key_values
    next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
    generated = []
    with torch.no_grad():
        for i in range(max_new_tokens):
            token_id = int(next_token.item())
            generated.append(token_id)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            kv_len = prefix_len + i + 1
            step_mask = additive[prefix_len + i, :kv_len].clone().reshape(1, 1, 1, kv_len)
            step = model(
                input_ids=next_token,
                attention_mask=step_mask,
                position_ids=torch.tensor(
                    [[decode_start + i]], dtype=torch.long, device=model.device
                ),
                cache_position=torch.tensor(
                    [prefix_len + i], dtype=torch.long, device=model.device
                ),
                past_key_values=past,
                use_cache=True,
            )
            past = step.past_key_values
            next_token = step.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
    return generated


def compute_doc_kv(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    doc_text: str,
    layout: Layout,
) -> CacheEntry:
    """Single-doc prefill, extract per-layer K/V from past_key_values into a CacheEntry.

    Uses `variant='full'` because a single doc attending only to itself IS causal-within-block
    under DDI's block-diagonal geometry (the qf/strict/bridge_k gating only matters when there
    are multiple docs to isolate). Uses `build_positions([L], 0, 0, layout)` so that under
    `restart` the positions are exactly [0..L-1] — that is what makes the returned KV block a
    pure function of the document's content.
    """
    doc_ids = tokenizer(doc_text, return_tensors="pt", add_special_tokens=False).input_ids
    doc_len = int(doc_ids.shape[-1])
    if doc_len < 1:
        raise ValueError(f"doc_text tokenises to zero tokens: {doc_text!r}")

    input_ids = doc_ids.to(model.device)
    positions = build_positions([doc_len], 0, 0, layout).unsqueeze(0).to(model.device)

    allow = build_mask([doc_len], query_len=0, answer_len=0, variant="full")
    additive = to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # Bypass forward_with_mask because we need past_key_values back; the mask/positions
        # validation there is duplicated here implicitly via build_mask/build_positions.
        out = model(
            input_ids=input_ids,
            attention_mask=additive,
            position_ids=positions,
            use_cache=True,
        )

    past = out.past_key_values
    # HF's DynamicCache exposes .key_cache / .value_cache as parallel lists per layer;
    # the legacy tuple-of-tuples format ((k, v), (k, v), ...) is also still supported.
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        k_list = [k.detach().cpu().contiguous() for k in past.key_cache]
        v_list = [v.detach().cpu().contiguous() for v in past.value_cache]
    else:
        k_list = [layer[0].detach().cpu().contiguous() for layer in past]
        v_list = [layer[1].detach().cpu().contiguous() for layer in past]

    model_id = getattr(model.config, "_name_or_path", "") or type(model).__name__
    key = compute_key(model_id, model.dtype, doc_text)
    return CacheEntry(
        key=key,
        k_per_layer=k_list,
        v_per_layer=v_list,
        doc_len=doc_len,
        layout=layout,
    )
