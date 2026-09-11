"""Opt-in V60 slice: token-native per-document B (fresh) / C (disk) execution.

This is not the combined-prefill A path and does not replace any production default.
Only single-device, unquantized, dense Qwen2 with static RoPE and eager attention is
supported here. Callers attest immutable weights/adapter/tokenizer identities (use
content digests, not mutable paths), and must rebuild the contract after changes.
No tokenizer, system prefix, padding, batching, or implicit special tokens are added.
Cache directories must be trusted; metadata validation is not artifact authentication.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch
import transformers

from ddi.cache import (
    CacheEntry,
    DocCache,
    forward_query_with_doc_cache,
    generate_tokens_with_doc_cache,
)
from ddi.masks import build_mask, to_additive


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def model_config_identity(model: Any) -> str:
    """Digest the actual config, including architecture and RoPE settings."""
    return _digest(model.config.to_dict())


def runtime_identity(model: Any) -> str:
    """Observed runtime fingerprint; not a promise of cross-host bitwise equality."""
    return _digest({
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_build": torch.__config__.show(),
        "device": str(model.device),
        "accelerator": (
            torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "none"
        ),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_version": torch.version.cuda,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "mkldnn": torch.backends.mkldnn.enabled,
        "fp16_reduced_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduced_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
    })


@dataclass(frozen=True)
class ExecutionContract:
    weights_identity: str
    adapters_identity: str
    tokenizer_identity: str
    model_config_identity: str
    runtime_identity: str
    backend: str
    dtype: str
    layout: str = "restart"
    system_prefix: str = "none"
    padding: str = "none"
    encoding: str = "per-document-eval-v1"

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("contract fields must be explicit non-empty strings")
        if (
            self.backend not in {"eager", "sdpa"}
            or self.dtype not in {"torch.float32", "torch.float16", "torch.bfloat16"}
            or self.layout != "restart"
            or self.system_prefix != "none"
            or self.padding != "none"
            or self.encoding != "per-document-eval-v1"
        ):
            raise ValueError("unsupported canonical execution contract")

    @property
    def identity(self) -> str:
        return _digest({"canonical_schema": 60, **asdict(self)})


# V65 integrity policy: full finite-value + geometry validation runs exactly once, at
# STORE time, inside encode_document before a payload is ever written to the DocCache
# (the store-time gate is the only writer of `<key>.pt` payloads in this module).
# Integrity checking has therefore moved from load time to store time: loaded payloads
# that carry the store-time marker below are only cheap-checked (metadata equality,
# shape/dtype/device/contiguity), because per-request isfinite over every stored layer
# dominated the C-path latency profile (results/v65_profile_20260910T094553Z). This is
# sound because cache directories must be trusted (module docstring) — a payload that
# was finite and well-formed when written stays so on disk. Payloads persisted before
# V65 carry no marker and keep the original full load-time check, so pre-existing
# stores behave exactly as before. `validated_sha256` binds the validation record
# (policy, key, geometry, dtype) — it is a deterministic, cheaply recomputable digest
# of that record, not a digest of the tensor bytes; byte-level re-verification at load
# time is intentionally omitted under the same trusted-store policy.
STORE_VALIDATION_POLICY = "canonical-store-validate-v65"
_VALIDATION_FIELDS = ("validated_by", "validated_sha256")
_LRU_BYTE_BUDGET = 2 * 1024**3


class _LoadedEntryLru:
    """Byte-budgeted in-process LRU in front of disk `torch.load` (V65).

    Holds only CPU-side CacheEntry payloads that already passed validation, so repeated
    requests for the same document skip the synchronous disk read. Purely an accelerator
    tier: the disk payload remains the source of truth — a cached entry is served only
    while its backing file still exists — so eviction and external store clears change
    latency, never values. Single-threaded like the rest of this module.
    """

    def __init__(self, byte_budget: int = _LRU_BYTE_BUDGET):
        self._budget = byte_budget
        self._entries: OrderedDict[str, tuple[int, CacheEntry]] = OrderedDict()
        self._bytes = 0

    @staticmethod
    def _payload_bytes(entry: CacheEntry) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in entry.k_per_layer + entry.v_per_layer
        )

    def get(self, key: str) -> CacheEntry | None:
        item = self._entries.get(key)
        if item is None:
            return None
        self._entries.move_to_end(key)
        return item[1]

    def remember(self, entry: CacheEntry) -> None:
        cost = self._payload_bytes(entry)
        if cost > self._budget:
            return
        existing = self._entries.pop(entry.key, None)
        if existing is not None:
            self._bytes -= existing[0]
        while self._entries and self._bytes + cost > self._budget:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted[0]
        self._entries[entry.key] = (cost, entry)
        self._bytes += cost


class CanonicalCache:
    """Explicit opt-in request API. ``store=None`` is B; a DocCache is C.

    IDs must be nonempty [1, L] int64 tensors, with no pad token occurrences.
    Even a pad ID used as content is rejected in this bounded API. Prefix absence
    and immutable external identities are caller attestations, not inferable from IDs.
    Freshly encoded entries stay on device for the current request only. Disk-resident
    entries are served through a small byte-budgeted in-process LRU of CPU copies in
    front of the store (V65); LRU hits return exactly what the disk path returns, and
    the disk payload stays authoritative.
    """

    def __init__(self, model: Any, contract: ExecutionContract, store: DocCache | None = None):
        self.model = model
        self.contract = contract
        self.store = store
        self._loaded = _LoadedEntryLru()
        self._check_model()

    def _check_model(self) -> None:
        model, contract = self.model, self.contract
        config = model.config
        if any(module.training for module in model.modules()):
            raise ValueError("canonical execution requires model.eval() on all modules")
        if (
            config.model_type != "qwen2"
            or getattr(config, "rope_scaling", None)
            or getattr(config, "use_sliding_window", False)
            or getattr(model, "is_quantized", False)
            or getattr(model, "hf_device_map", None)
        ):
            raise ValueError("requires dense single-device Qwen2 with static RoPE")
        if (
            config._attn_implementation != contract.backend
            or str(model.dtype) != contract.dtype
            or model_config_identity(model) != contract.model_config_identity
            or runtime_identity(model) != contract.runtime_identity
        ):
            raise ValueError("model/config/runtime does not match execution contract")
        if torch.is_autocast_enabled(model.device.type):
            raise ValueError("autocast is outside the canonical execution contract")
        if any(
            p.device != model.device or (p.is_floating_point() and p.dtype != model.dtype)
            for p in model.parameters()
        ):
            raise ValueError("requires uniform model device and dtype")

    def _ids(self, ids: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(ids, torch.Tensor)
            or ids.dtype != torch.long
            or ids.ndim != 2
            or ids.shape[0] != 1
            or ids.shape[1] == 0
            or ids.layout != torch.strided
            or ids.device.type == "meta"
        ):
            raise ValueError("IDs must be nonempty dense [1, L] int64 tensors")
        if bool(((ids < 0) | (ids >= self.model.config.vocab_size)).any()):
            raise ValueError("token ID outside vocabulary")
        pad = self.model.config.pad_token_id
        if pad is not None and bool((ids == pad).any()):
            raise ValueError("padded inputs / pad token IDs are not supported")
        if ids.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("IDs exceed supported position range")
        return ids.detach().contiguous()

    def _metadata(self, ids: torch.Tensor) -> dict[str, Any]:
        # JSON integer arrays avoid text retokenization and native-endian byte keys.
        token_digest = _digest(ids.cpu().tolist()[0])
        key = _digest([self.contract.identity, token_digest])
        return {
            "key": key,
            "doc_len": ids.shape[1],
            "layout": "restart",
            "cache_identity": self.contract.identity,
            "token_ids_sha256": token_digest,
            "tensor_dtype": self.contract.dtype,
            "schema_version": 60,
            "validated_by": STORE_VALIDATION_POLICY,
            "validated_sha256": _digest(
                [STORE_VALIDATION_POLICY, key, ids.shape[1], self.contract.dtype]
            ),
        }

    def _validate(
        self,
        entry: CacheEntry,
        meta: dict[str, Any],
        device: torch.device,
        *,
        check_finite: bool = True,
    ) -> None:
        """Full validation also scans finite values; cheap load-time checks do not.

        `check_finite=False` implements the V65 policy: payloads carrying the store-time
        validation marker were fully integrity-checked when written (see the policy
        comment above), so loading only re-verifies cheap structural properties.
        """
        for name, expected in meta.items():
            actual = getattr(entry, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"canonical payload metadata mismatch: {name}")
        config = self.model.config
        shape = (1, config.num_key_value_heads, meta["doc_len"],
                 getattr(config, "head_dim", None)
                 or config.hidden_size // config.num_attention_heads)
        for tensors in (entry.k_per_layer, entry.v_per_layer):
            if not isinstance(tensors, list) or len(tensors) != config.num_hidden_layers:
                raise ValueError("canonical payload layer count mismatch")
            for tensor in tensors:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.layout != torch.strided
                    or tuple(tensor.shape) != shape
                    or tensor.dtype != self.model.dtype
                    or tensor.device != device
                    or not tensor.is_contiguous()
                    or (check_finite and not bool(torch.isfinite(tensor).all()))
                ):
                    raise ValueError(
                        "canonical payload tensor geometry/dtype/device/finite mismatch"
                    )

    def encode_document(self, token_ids: torch.Tensor) -> CacheEntry:
        """Always encode one document, retaining detached, contiguous live-device KVs."""
        self._check_model()
        ids = self._ids(token_ids)
        length = ids.shape[1]
        slots = torch.arange(length, device=self.model.device)
        mask = to_additive(build_mask([length], 0, 0, "full"), self.model.dtype)
        with torch.no_grad():
            out = self.model(
                input_ids=ids.to(self.model.device),
                attention_mask=mask.to(self.model.device)[None, None],
                position_ids=slots[None],
                cache_position=slots,
                use_cache=True,
            )
        past = out.past_key_values
        if hasattr(past, "key_cache"):
            layers = list(zip(past.key_cache, past.value_cache, strict=True))
        else:
            layers = list(past)
        meta = self._metadata(ids)
        entry = CacheEntry(
            **meta,
            k_per_layer=[k.detach().contiguous() for k, _ in layers],
            v_per_layer=[v.detach().contiguous() for _, v in layers],
        )
        self._validate(entry, meta, self.model.device)
        return entry

    def documents(self, documents: list[torch.Tensor]) -> list[CacheEntry]:
        """Resolve in request order, preserving duplicates; invalid disk entries raise."""
        self._check_model()
        if not isinstance(documents, list) or not documents:
            raise ValueError("at least one document is required")
        ids_list = [self._ids(ids) for ids in documents]
        entries = []
        for ids in ids_list:
            meta = self._metadata(ids)
            entry = self._load(meta)
            if entry is None:
                entry = self.encode_document(ids)
                if self.store is not None:
                    # Persistence owns CPU copies, never replacing the live cold-path KVs.
                    # The CPU copy is also the LRU-resident form, identical to what a
                    # later disk load would hand back.
                    cpu_entry = replace(
                        entry,
                        k_per_layer=[t.detach().to("cpu", copy=True) for t in entry.k_per_layer],
                        v_per_layer=[t.detach().to("cpu", copy=True) for t in entry.v_per_layer],
                    )
                    self.store.put(cpu_entry)
                    self._loaded.remember(cpu_entry)
            entries.append(entry)
        return entries

    def _load(self, meta: dict[str, Any]) -> CacheEntry | None:
        """Return a validated CPU entry from the LRU tier or the disk store, else None.

        Full finite-value + geometry integrity checking happens once at store time,
        inside encode_document, before DocCache.put writes the payload (see the V65
        integrity-policy comment). Marker-carrying payloads are therefore only
        cheap-checked here; pre-V65 payloads without the marker keep the original
        full load-time check.
        """
        if self.store is None:
            return None
        key = meta["key"]
        path = self.store.cache_dir / f"{key}.pt"
        cached = self._loaded.get(key)
        # The disk payload stays authoritative: an LRU hit is served only while its
        # backing file still exists, so an externally cleared store re-encodes and
        # re-persists exactly as it would without the memory tier. The existence stat
        # is nanoseconds next to the millisecond disk read the hit avoids.
        if cached is not None and path.exists():
            self._validate(cached, meta, torch.device("cpu"), check_finite=False)
            return cached
        if not path.exists():
            return None
        # DocCache.get coerces schema_version and defaults missing fields.
        # Read its wire format strictly instead, without pickle object loading.
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict) or set(blob) != {"k", "v", "meta"}:
            raise ValueError("malformed canonical payload schema")
        stored = blob["meta"]
        legacy_meta = {
            name: value for name, value in meta.items() if name not in _VALIDATION_FIELDS
        }
        if (
            not isinstance(stored, dict)
            or set(stored) not in (set(meta), set(legacy_meta))
        ):
            raise ValueError("malformed canonical payload schema")
        entry = CacheEntry(**stored, k_per_layer=blob["k"], v_per_layer=blob["v"])
        if set(stored) == set(legacy_meta):
            # Pre-V65 payload: no store-time marker, so retain the original full
            # load-time integrity check including the finite-value scan.
            self._validate(entry, legacy_meta, torch.device("cpu"))
        else:
            self._validate(entry, meta, torch.device("cpu"), check_finite=False)
        self._loaded.remember(entry)
        return entry

    def request(
        self, documents: list[torch.Tensor], query_ids: torch.Tensor, *,
        max_new_tokens: int | None = None, eos_token_id: int | None = None,
    ) -> Any:
        """Return query ModelOutput, or greedy token IDs when max_new_tokens is set."""
        self._check_model()
        query_ids = self._ids(query_ids)
        if not isinstance(documents, list) or not documents:
            raise ValueError("at least one document is required")
        documents = [self._ids(ids) for ids in documents]
        if max_new_tokens is not None and (
            type(max_new_tokens) is not int or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if eos_token_id is not None and (
            type(eos_token_id) is not int or not 0 <= eos_token_id < self.model.config.vocab_size
        ):
            raise ValueError("invalid eos_token_id")
        end = max(ids.shape[1] for ids in documents) + query_ids.shape[1]
        if end + (max_new_tokens or 0) > self.model.config.max_position_embeddings:
            raise ValueError("request exceeds supported logical position range")
        if max_new_tokens == 0:
            return []
        entries = self.documents(documents)
        if max_new_tokens is None:
            return forward_query_with_doc_cache(self.model, query_ids, entries)
        return generate_tokens_with_doc_cache(
            self.model, SimpleNamespace(eos_token_id=eos_token_id), query_ids, entries,
            max_new_tokens,
        )
