"""Minimal V70 link-treatment harness.

This module is intentionally separate from the V67 training scripts.  Link markers use
fixed deterministic embeddings (not learned link tokens).  Documents are either encoded
in the monolithic request or freshly and independently with autograd intact; a cache miss
always means a fresh encoding and never removes evidence.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from transformers import DynamicCache

from ddi.cache import CacheEntry, DocCache
from ddi.canonical_cache import CanonicalCache, ExecutionContract
from ddi.link_token_contract import (
    DocumentPayload,
    LinkTokenContract,
    RequestLayout,
    build_request_layout,
)
from ddi.metrics import answer_presence, exact_match, token_f1

LINK_TOKENS = tuple(f"<|ddi_link_{i}|>" for i in range(5))
EMBEDDING_POLICY = "fixed-sha256-sine-v1"
BOOTSTRAP_SEED = 47
BOOTSTRAP_RESAMPLES = 10_000
NONINFERIORITY_MARGIN = -0.02


@dataclass(frozen=True)
class TokenContractState:
    contract: LinkTokenContract
    base_vocab_size: int
    embedding_policy: str = EMBEDDING_POLICY


@dataclass(frozen=True)
class EncodedRequest:
    documents: tuple[DocumentPayload, ...]
    query_token_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ForwardResult:
    logits: torch.Tensor
    answer_logits: torch.Tensor
    layout: RequestLayout
    past_key_values: Any
    document_entries: tuple[CacheEntry, ...] = ()


def _fixed_row(token: str, width: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256((EMBEDDING_POLICY + "\0" + token).encode()).digest()
    phase = int.from_bytes(digest[:8], "big") / 2**64 * (2 * math.pi)
    values = torch.arange(width, dtype=torch.float64, device=device)
    return (0.02 * torch.sin(values * math.sqrt(2.0) + phase)).to(dtype)


def _token_id(tokenizer: Any, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if len(ids) != 1:
        raise ValueError(f"V70 link token must encode to one ID: {token!r} -> {ids!r}")
    return int(ids[0])


def install_fixed_link_tokens(model: Any, tokenizer: Any) -> TokenContractState:
    """Add/validate five special tokens and freeze deterministic marker rows.

    A restored V70 checkpoint carries the policy and base vocabulary in its config.  In
    that case this function validates dimensions and leaves checkpoint rows untouched.
    """
    restored_policy = getattr(model.config, "ddi_link_embedding_policy", None)
    restored_base = getattr(model.config, "ddi_base_vocab_size", None)
    if restored_policy is not None:
        if restored_policy != EMBEDDING_POLICY or type(restored_base) is not int:
            raise ValueError("unsupported or incomplete V70 link-token checkpoint metadata")
        base_vocab_size = restored_base
    else:
        base_vocab_size = len(tokenizer)
        added = tokenizer.add_special_tokens({"additional_special_tokens": list(LINK_TOKENS)})
        if added != len(LINK_TOKENS):
            raise ValueError("all V70 link tokens must be newly added to the base tokenizer")
        model.resize_token_embeddings(len(tokenizer))

    link_ids = tuple(_token_id(tokenizer, token) for token in LINK_TOKENS)
    if len(set(link_ids)) != 5 or any(token_id < base_vocab_size for token_id in link_ids):
        raise ValueError("V70 link tokens must have five distinct IDs outside the base vocabulary")
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if (
        input_embeddings.num_embeddings != len(tokenizer)
        or model.config.vocab_size != len(tokenizer)
        or output_embeddings is None
        or output_embeddings.weight.shape[0] != len(tokenizer)
    ):
        raise ValueError("model/tokenizer vocabulary dimensions do not match")

    if restored_policy is None:
        with torch.no_grad():
            for token, token_id in zip(LINK_TOKENS, link_ids, strict=True):
                input_embeddings.weight[token_id].copy_(_fixed_row(
                    token,
                    input_embeddings.weight.shape[1],
                    dtype=input_embeddings.weight.dtype,
                    device=input_embeddings.weight.device,
                ))
                if output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr():
                    output_embeddings.weight[token_id].copy_(_fixed_row(
                        "output\0" + token,
                        output_embeddings.weight.shape[1],
                        dtype=output_embeddings.weight.dtype,
                        device=output_embeddings.weight.device,
                    ))
        model.config.ddi_link_embedding_policy = EMBEDDING_POLICY
        model.config.ddi_base_vocab_size = base_vocab_size
        model.config.ddi_link_tokens = list(LINK_TOKENS)

    input_embeddings.weight.requires_grad_(False)
    output_embeddings.weight.requires_grad_(False)
    contract = LinkTokenContract(
        document_cache_identity=f"v70-{EMBEDDING_POLICY}-base-{base_vocab_size}",
        link_token_ids=link_ids,  # type: ignore[arg-type]
    )
    return TokenContractState(contract=contract, base_vocab_size=base_vocab_size)


def assert_original_inputs(state: TokenContractState, request: EncodedRequest) -> None:
    _assert_no_reserved_ids(state.contract, request)


def _assert_no_reserved_ids(contract: LinkTokenContract, request: EncodedRequest) -> None:
    link_ids = set(contract.link_token_ids)
    original = [
        *(token for document in request.documents for token in document.token_ids),
        *request.query_token_ids,
        *request.answer_token_ids,
    ]
    if any(token in link_ids for token in original):
        raise ValueError("a reserved V70 link ID appears in original document/query/answer input")


def reset_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_adamw8bit(parameters: Sequence[torch.nn.Parameter], **kwargs: Any) -> Any:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    try:
        import bitsandbytes as bnb
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("V70 frozen training requires bitsandbytes AdamW8bit") from error
    return bnb.optim.AdamW8bit(params, **kwargs)


def layout_tensors(
    layout: RequestLayout, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.tensor([layout.physical_token_ids], dtype=torch.long, device=device)
    positions = torch.tensor([layout.positions], dtype=torch.long, device=device)
    allow = torch.tensor(layout.allow, dtype=torch.bool, device=device)
    additive = torch.zeros(allow.shape, dtype=dtype, device=device).masked_fill_(~allow, -torch.inf)
    return ids, additive[None, None], positions


def _answer_bounds(
    layout: RequestLayout, query_len: int, answer_len: int, *, tail: bool
) -> tuple[int, int]:
    if answer_len < 1:
        raise ValueError("answer logits require a non-empty answer")
    if tail:
        lo = len(layout.document_identities) * layout.k + query_len - 1
        return lo, lo + answer_len
    total = len(layout.physical_token_ids)
    return total - answer_len - 1, total - 1


def monolithic_forward(
    model: Any,
    contract: LinkTokenContract,
    request: EncodedRequest,
    *,
    k: int,
    full_causal: bool = False,
    use_cache: bool = False,
) -> ForwardResult:
    _assert_no_reserved_ids(contract, request)
    layout = build_request_layout(
        contract, request.documents, request.query_token_ids, request.answer_token_ids, k=k
    )
    ids, mask, positions = layout_tensors(layout, model.dtype, model.device)
    if full_causal:
        length = ids.shape[1]
        causal = torch.ones((length, length), dtype=torch.bool, device=model.device).tril()
        mask = torch.zeros_like(causal, dtype=model.dtype).masked_fill_(~causal, -torch.inf)[
            None, None
        ]
    out = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        cache_position=torch.arange(ids.shape[1], device=model.device),
        use_cache=use_cache,
    )
    answer_logits = out.logits[:, 0:0]
    if request.answer_token_ids:
        lo, hi = _answer_bounds(
            layout,
            len(request.query_token_ids),
            len(request.answer_token_ids),
            tail=False,
        )
        answer_logits = out.logits[:, lo:hi]
    return ForwardResult(out.logits, answer_logits, layout, out.past_key_values)


def encode_documents_with_grad(
    model: Any, documents: tuple[DocumentPayload, ...]
) -> tuple[CacheEntry, ...]:
    """Independently encode documents without detach/no_grad, preserving autograd."""
    entries = []
    for index, document in enumerate(documents):
        ids = torch.tensor([document.token_ids], dtype=torch.long, device=model.device)
        length = ids.shape[1]
        causal = torch.ones((length, length), dtype=torch.bool, device=model.device).tril()
        mask = torch.zeros_like(causal, dtype=model.dtype).masked_fill_(
            ~causal, -torch.inf
        )[None, None]
        slots = torch.arange(length, device=model.device)
        out = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=slots[None],
            cache_position=slots,
            use_cache=True,
        )
        past = out.past_key_values
        if hasattr(past, "key_cache"):
            keys, values = list(past.key_cache), list(past.value_cache)
        else:
            keys, values = zip(*past, strict=True)
        entries.append(CacheEntry(
            key=f"training-only-{index}",
            k_per_layer=list(keys),
            v_per_layer=list(values),
            doc_len=length,
            layout="restart",
        ))
    return tuple(entries)


def _assemble_with_grad(entries: Sequence[CacheEntry], device: torch.device) -> DynamicCache:
    if not entries:
        raise ValueError("at least one document entry is required")
    layers = []
    for layer in range(len(entries[0].k_per_layer)):
        layers.append((
            torch.cat([entry.k_per_layer[layer].to(device) for entry in entries], dim=-2),
            torch.cat([entry.v_per_layer[layer].to(device) for entry in entries], dim=-2),
        ))
    return DynamicCache.from_legacy_cache(tuple(layers))


def independent_forward_from_entries(
    model: Any,
    contract: LinkTokenContract,
    request: EncodedRequest,
    entries: Sequence[CacheEntry],
    *,
    k: int,
    use_cache: bool = True,
) -> ForwardResult:
    _assert_no_reserved_ids(contract, request)
    layout = build_request_layout(
        contract, request.documents, request.query_token_ids, request.answer_token_ids, k=k
    )
    doc_total = sum(len(document.token_ids) for document in request.documents)
    ids, mask, positions = layout_tensors(layout, model.dtype, model.device)
    cache = _assemble_with_grad(entries, model.device)
    out = model(
        input_ids=ids[:, doc_total:],
        attention_mask=mask[:, :, doc_total:, :],
        position_ids=positions[:, doc_total:],
        cache_position=torch.arange(doc_total, ids.shape[1], device=model.device),
        past_key_values=cache,
        use_cache=use_cache,
    )
    answer_logits = out.logits[:, 0:0]
    if request.answer_token_ids:
        lo, hi = _answer_bounds(
            layout,
            len(request.query_token_ids),
            len(request.answer_token_ids),
            tail=True,
        )
        answer_logits = out.logits[:, lo:hi]
    return ForwardResult(out.logits, answer_logits, layout, out.past_key_values, tuple(entries))


def independent_forward(
    model: Any,
    contract: LinkTokenContract,
    request: EncodedRequest,
    *,
    k: int,
) -> ForwardResult:
    entries = encode_documents_with_grad(model, request.documents)
    return independent_forward_from_entries(model, contract, request, entries, k=k)


def teacher_answer_logits(model: Any, request: EncodedRequest) -> torch.Tensor:
    """No-link, full-causal, sequential teacher target shared by every k arm."""
    if not request.answer_token_ids:
        raise ValueError("teacher target requires a non-empty answer")
    tokens = tuple(
        token for document in request.documents for token in document.token_ids
    ) + request.query_token_ids + request.answer_token_ids
    length = len(tokens)
    ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
    causal = torch.ones((length, length), dtype=torch.bool, device=model.device).tril()
    mask = torch.zeros_like(causal, dtype=model.dtype).masked_fill_(~causal, -torch.inf)[None, None]
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=torch.arange(length, device=model.device)[None],
            cache_position=torch.arange(length, device=model.device),
            use_cache=False,
        )
    lo = length - len(request.answer_token_ids) - 1
    return out.logits[:, lo : length - 1].detach()


def training_loss(
    teacher: Any,
    student: Any,
    contract: LinkTokenContract,
    request: EncodedRequest,
    *,
    k: int,
    encoding: Literal["monolithic", "independent"],
) -> torch.Tensor:
    """KL to one arm-independent no-link teacher target for a scheduled treatment."""
    _assert_no_reserved_ids(contract, request)
    target = teacher_answer_logits(teacher, request).float()
    if encoding == "monolithic":
        student_logits = monolithic_forward(student, contract, request, k=k).answer_logits
    elif encoding == "independent":
        student_logits = independent_forward(student, contract, request, k=k).answer_logits
    else:
        raise ValueError(f"unknown V70 encoding treatment: {encoding!r}")
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits.float(), dim=-1),
        torch.nn.functional.softmax(target, dim=-1),
        reduction="batchmean",
    )
    if loss.grad_fn is None or not bool(torch.isfinite(loss)):
        raise RuntimeError("V70 training loss is detached or non-finite")
    return loss


@torch.no_grad()
def _decode_from_prefill(
    model: Any,
    contract: LinkTokenContract,
    request: EncodedRequest,
    prefill: ForwardResult,
    *,
    k: int,
    max_new_tokens: int,
    eos_token_id: int | None,
    independent_entries: Sequence[CacheEntry] | None = None,
    full_causal: bool = False,
) -> list[int]:
    if max_new_tokens < 0 or request.answer_token_ids:
        raise ValueError("generation requires nonnegative budget and no supplied answer")
    if max_new_tokens == 0:
        return []
    next_token = prefill.logits[:, -1].argmax(-1)
    past = prefill.past_key_values
    generated: list[int] = []
    prefix_len = len(prefill.layout.physical_token_ids)
    for index in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == eos_token_id or index + 1 == max_new_tokens:
            break
        step_request = EncodedRequest(request.documents, request.query_token_ids, tuple(generated))
        layout = build_request_layout(
            contract, step_request.documents, step_request.query_token_ids,
            step_request.answer_token_ids, k=k,
        )
        ids, mask, positions = layout_tensors(layout, model.dtype, model.device)
        row = ids.shape[1] - 1
        step_mask = (
            torch.zeros((1, 1, 1, ids.shape[1]), dtype=model.dtype, device=model.device)
            if full_causal
            else mask[:, :, row : row + 1, :]
        )
        out = model(
            input_ids=ids[:, -1:],
            attention_mask=step_mask,
            position_ids=positions[:, -1:],
            cache_position=torch.tensor([prefix_len + index], device=model.device),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = out.logits[:, -1].argmax(-1)
    return generated


@torch.no_grad()
def generate_monolithic_ids(
    model: Any,
    contract: LinkTokenContract,
    request: EncodedRequest,
    *,
    k: int,
    max_new_tokens: int,
    eos_token_id: int | None,
    full_causal: bool = False,
) -> list[int]:
    prefill = monolithic_forward(
        model, contract, request, k=k, full_causal=full_causal, use_cache=True
    )
    return _decode_from_prefill(
        model, contract, request, prefill, k=k, max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id, full_causal=full_causal,
    )


@torch.no_grad()
def generate_independent_ids(
    model: Any,
    contract: LinkTokenContract,
    request: EncodedRequest,
    entries: Sequence[CacheEntry],
    *,
    k: int,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    """The one tail generator used by both fresh B and persisted C."""
    prefill = independent_forward_from_entries(model, contract, request, entries, k=k)
    return _decode_from_prefill(
        model, contract, request, prefill, k=k, max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id, independent_entries=entries,
    )


def _kv_equal(left: Sequence[CacheEntry], right: Sequence[CacheEntry]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.detach().cpu(), b.detach().cpu())
        for le, re in zip(left, right, strict=True)
        for a, b in zip(
            le.k_per_layer + le.v_per_layer, re.k_per_layer + re.v_per_layer, strict=True
        )
    )


def _quality(tokenizer: Any, generated_ids: list[int], gold: list[str]) -> dict[str, Any]:
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "generated_ids": generated_ids,
        "prediction": prediction,
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "presence": answer_presence(prediction, gold),
    }


def evaluate_abcd(
    model: Any,
    tokenizer: Any,
    token_state: TokenContractState,
    records: Sequence[tuple[str, EncodedRequest, list[str]]],
    canonical_contract: ExecutionContract,
    cache_dir: Path,
    *,
    k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Evaluate A/B/C/D after persisting and reopening all C document entries."""
    model.eval()
    writer_store = DocCache(cache_dir)
    writer = CanonicalCache(model, canonical_contract, writer_store)
    all_documents = [
        torch.tensor([document.token_ids], dtype=torch.long)
        for _, request, _ in records
        for document in request.documents
    ]
    writer.documents(all_documents)
    expected_keys = {writer._metadata(document)["key"] for document in all_documents}
    size_before = writer_store.size()
    if size_before != len(expected_keys):
        raise RuntimeError("persisted C store does not contain exactly the expected document keys")

    fresh = CanonicalCache(model, canonical_contract)
    reopened_store = DocCache(cache_dir)
    cached = CanonicalCache(model, canonical_contract, reopened_store)

    def fail_encode(_: torch.Tensor) -> CacheEntry:
        raise AssertionError("C attempted to encode a document instead of hitting reopened storage")

    cached.encode_document = fail_encode  # type: ignore[method-assign]
    rows = []
    for record_id, request, gold in records:
        assert_original_inputs(token_state, request)
        tensors = [torch.tensor([doc.token_ids], dtype=torch.long) for doc in request.documents]
        b_entries = fresh.documents(tensors)
        c_entries = cached.documents(tensors)
        if not _kv_equal(b_entries, c_entries):
            raise RuntimeError(f"record {record_id}: B/C document K/V differ")
        arms = {
            "A": generate_monolithic_ids(
                model, token_state.contract, request, k=k, max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            ),
            "B": generate_independent_ids(
                model, token_state.contract, request, b_entries, k=k,
                max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id,
            ),
            "C": generate_independent_ids(
                model, token_state.contract, request, c_entries, k=k,
                max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id,
            ),
            "D": generate_monolithic_ids(
                model, token_state.contract, request, k=k, max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, full_causal=True,
            ),
        }
        rows.append({
            "record_id": record_id,
            "gold": gold,
            "arms": {name: _quality(tokenizer, ids, gold) for name, ids in arms.items()},
        })
    if reopened_store.size() != size_before:
        raise RuntimeError("C store grew during hit-only evaluation")
    if {path.stem for path in cache_dir.glob("*.pt")} != expected_keys:
        raise RuntimeError("C store keys changed during evaluation")
    return {
        "schema": "v70-abcd-v1",
        "k": k,
        "fixed_marker_conditioning": True,
        "rows": rows,
        "cache_validation": {
            "persisted_before_evaluation": True,
            "reopened_store": True,
            "encode_on_c_miss": "fail_closed",
            "hit_only": True,
            "entry_count_before": size_before,
            "entry_count_after": reopened_store.size(),
            "keys_unchanged": True,
            "b_c_kv_equal_after_detached_cpu_normalization": True,
        },
        "gates": {
            "C-B": paired_quality_gate(rows, candidate="C", reference="B"),
            "C-D": paired_quality_gate(rows, candidate="C", reference="D"),
        },
    }


def paired_bootstrap(deltas: Sequence[float], *, seed: int = BOOTSTRAP_SEED,
                     resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or resamples < 1 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap needs finite, non-empty 1D deltas and resamples >= 1")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draws[index] = values[rng.integers(0, values.size, values.size)].mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "n": int(values.size),
        "seed": seed,
        "resamples": resamples,
        "method": "paired-percentile-mean-delta",
    }


def paired_quality_gate(
    rows: Sequence[dict[str, Any]], *, candidate: str, reference: str
) -> dict[str, Any]:
    analyses = {}
    for metric in ("em", "f1"):
        deltas = [row["arms"][candidate][metric] - row["arms"][reference][metric] for row in rows]
        analyses[metric] = paired_bootstrap(deltas)
    passed = all(analyses[metric]["ci95"][0] >= NONINFERIORITY_MARGIN for metric in analyses)
    return {
        "candidate": candidate,
        "reference": reference,
        "margin": NONINFERIORITY_MARGIN,
        "metrics": analyses,
        "pass": passed,
    }
