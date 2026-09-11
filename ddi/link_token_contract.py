"""V62 request-time link-token contract.

This module is deliberately tokenizer- and model-free.  It specifies token IDs,
positions, physical slots, and mask reachability for the V62 k=0/1/5 ablation;
it does not add tokens to a tokenizer or execute a model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

SplitLabel = Literal["train", "development", "confirmation"]
_SPLITS: tuple[SplitLabel, ...] = ("train", "development", "confirmation")
_LINK_COUNTS = (0, 1, 5)


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def _token_ids(name: str, values: tuple[int, ...], *, allow_empty: bool = False) -> None:
    if not isinstance(values, tuple) or (not values and not allow_empty):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValueError(f"{name} must be a {qualifier} tuple")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")


@dataclass(frozen=True)
class DocumentPayload:
    """Context-independent document input eligible for a persisted document cache."""

    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _token_ids("document token_ids", self.token_ids)

    def identity(self, document_cache_identity: str) -> str:
        """Identity intentionally excludes request links, order, query, and split."""
        if not isinstance(document_cache_identity, str) or not document_cache_identity:
            raise ValueError("document_cache_identity must be a non-empty string")
        return _identity({
            "schema": "v62-document-payload-v1",
            "document_cache_identity": document_cache_identity,
            "token_ids": self.token_ids,
        })


@dataclass(frozen=True)
class LinkTokenContract:
    """Immutable V62 interface identity, separate from document-cache identity."""

    document_cache_identity: str
    link_token_ids: tuple[int, int, int, int, int]
    position_layout: str = "restart-documents-request-tail-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.document_cache_identity, str) or not self.document_cache_identity:
            raise ValueError("document_cache_identity must be a non-empty string")
        _token_ids("link_token_ids", self.link_token_ids)
        if len(set(self.link_token_ids)) != len(self.link_token_ids):
            raise ValueError("link_token_ids must be distinct")
        if self.position_layout != "restart-documents-request-tail-v1":
            raise ValueError("unsupported V62 position layout")

    @property
    def identity(self) -> str:
        return _identity({
            "schema": "v62-link-token-contract-v1",
            "document_cache_identity": self.document_cache_identity,
            "link_token_ids": self.link_token_ids,
            "position_layout": self.position_layout,
            "supported_k": _LINK_COUNTS,
        })


@dataclass(frozen=True)
class SplitManifest:
    """Fixed IDs for V62 selection and one untouched confirmation evaluation."""

    train_ids: frozenset[str]
    development_ids: frozenset[str]
    confirmation_ids: frozenset[str]

    def __post_init__(self) -> None:
        groups = (self.train_ids, self.development_ids, self.confirmation_ids)
        if any(not isinstance(group, frozenset) for group in groups):
            raise ValueError("split IDs must be frozensets")
        if any(not isinstance(item, str) or not item for group in groups for item in group):
            raise ValueError("split IDs must be non-empty strings")
        if self.train_ids & self.development_ids:
            raise ValueError("train and development IDs overlap")
        if self.train_ids & self.confirmation_ids:
            raise ValueError("train and confirmation IDs overlap")
        if self.development_ids & self.confirmation_ids:
            raise ValueError("confirmation data cannot be reused as development")

    def ids_for(self, split: SplitLabel) -> frozenset[str]:
        if split not in _SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return getattr(self, f"{split}_ids")


@dataclass(frozen=True)
class RequestLayout:
    """Pure token-ID layout and boolean attention allow-matrix for one request."""

    k: int
    physical_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    logical_order: tuple[str, ...]
    allow: tuple[tuple[bool, ...], ...]
    document_identities: tuple[str, ...]
    request_identity: str


def build_request_layout(
    contract: LinkTokenContract,
    documents: tuple[DocumentPayload, ...],
    query_token_ids: tuple[int, ...],
    answer_token_ids: tuple[int, ...] = (),
    *,
    k: int,
) -> RequestLayout:
    """Build the V62 logical contract while executing cached documents first.

    Logical order is ``doc0, links0, doc1, links1, ..., query, answer``.
    Physical slots are ``all documents, all links, query, answer`` so document
    K/Vs can be assembled from context-independent cache entries.  The returned
    mask implements logical, not physical, reachability.
    """
    if not isinstance(contract, LinkTokenContract):
        raise ValueError("contract must be a LinkTokenContract")
    if not isinstance(documents, tuple) or not documents:
        raise ValueError("documents must be a non-empty tuple of DocumentPayload")
    if any(not isinstance(document, DocumentPayload) for document in documents):
        raise ValueError("documents must contain DocumentPayload values")
    _token_ids("query_token_ids", query_token_ids)
    _token_ids("answer_token_ids", answer_token_ids, allow_empty=True)
    if type(k) is not int or k not in _LINK_COUNTS:
        raise ValueError(f"k must be one of {_LINK_COUNTS}")

    document_ids = tuple(
        document.identity(contract.document_cache_identity) for document in documents
    )
    doc_lengths = tuple(len(document.token_ids) for document in documents)
    max_doc_length = max(doc_lengths)
    physical_tokens = tuple(token for document in documents for token in document.token_ids)
    physical_positions = tuple(position for length in doc_lengths for position in range(length))
    doc_owner: list[int] = []
    for document_index, document in enumerate(documents):
        doc_owner.extend([document_index] * len(document.token_ids))

    link_owner: list[int] = []
    logical_order: list[str] = []
    for document_index in range(len(documents)):
        logical_order.extend([f"doc:{document_index}"] * doc_lengths[document_index])
        logical_order.extend([f"link:{document_index}"] * k)
        physical_tokens += contract.link_token_ids[:k]
        physical_positions += tuple(max_doc_length + len(link_owner) + slot for slot in range(k))
        link_owner.extend([document_index] * k)
    physical_tokens += query_token_ids + answer_token_ids
    tail_start = max_doc_length + len(link_owner)
    physical_positions += tuple(
        range(tail_start, tail_start + len(query_token_ids) + len(answer_token_ids))
    )
    logical_order.extend(["query"] * len(query_token_ids) + ["answer"] * len(answer_token_ids))

    doc_total = sum(doc_lengths)
    link_total = len(link_owner)
    query_start = doc_total + link_total
    answer_start = query_start + len(query_token_ids)
    total = len(physical_tokens)
    allow: list[tuple[bool, ...]] = []
    for row in range(total):
        values: list[bool] = []
        for column in range(total):
            permitted = False
            if row < doc_total:
                row_owner = doc_owner[row]
                permitted = column < doc_total and doc_owner[column] == row_owner and column <= row
            elif row < query_start:
                row_link = row - doc_total
                owner = link_owner[row_link]
                permitted = (
                    (column < doc_total and doc_owner[column] <= owner)
                    or (doc_total <= column <= row)
                )
            elif row < answer_start:
                permitted = column < query_start or (query_start <= column <= row)
            else:
                permitted = column < answer_start or (answer_start <= column <= row)
            values.append(permitted)
        allow.append(tuple(values))

    return RequestLayout(
        k=k,
        physical_token_ids=physical_tokens,
        positions=physical_positions,
        logical_order=tuple(logical_order),
        allow=tuple(allow),
        document_identities=document_ids,
        request_identity=_identity({
            "schema": "v62-request-layout-v1",
            "contract_identity": contract.identity,
            "k": k,
            "document_identities": document_ids,
            "query_token_ids": query_token_ids,
            "answer_token_ids": answer_token_ids,
        }),
    )
