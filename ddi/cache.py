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

The Phase 4.2 acceptance criterion — a cache hit yields KV blocks identical to the cold
compute within bf16 tolerance — is checked by the `kind: cache task: correctness` cell in
`ddi/run.py`, which drives this module alongside `compute_doc_kv`.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
