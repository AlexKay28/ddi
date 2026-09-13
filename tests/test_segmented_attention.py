"""CPU tests for the V61 segmented-attention mathematical reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ddi.masks import build_mask
from ddi.segmented_attention import (
    DocumentSegment,
    merge_lse,
    segmented_attention,
    segmented_attention_decomposed,
)


def _case(dtype: torch.dtype, *, query_len: int = 3, decode_len: int = 2):
    torch.manual_seed(61)
    lengths, hq, hkv, dim = [2, 3, 1], 4, 2, 3
    total = sum(lengths)
    doc_q = torch.randn(total, hq, dim, dtype=dtype)
    # K/V are intentionally in cache-slot order, unlike the logical doc order.
    doc_k = torch.randn(total, hkv, dim, dtype=dtype)
    doc_v = torch.randn(total, hkv, dim, dtype=dtype)
    query_q = torch.randn(query_len, hq, dim, dtype=dtype)
    query_k = torch.randn(query_len, hkv, dim, dtype=dtype)
    query_v = torch.randn(query_len, hkv, dim, dtype=dtype)
    decode_q = torch.randn(decode_len, hq, dim, dtype=dtype)
    decode_k = torch.randn(decode_len, hkv, dim, dtype=dtype)
    decode_v = torch.randn(decode_len, hkv, dim, dtype=dtype)
    # These document identities intentionally repeat.  Segment identity, rather
    # than a de-duplicated logical ID, defines doc-to-doc visibility.
    documents = [
        DocumentSegment(torch.tensor([4, 1]), torch.tensor([10, 11]), "same"),
        DocumentSegment(torch.tensor([0, 5, 2]), torch.tensor([30, 31, 32]), "other"),
        DocumentSegment(torch.tensor([3]), torch.tensor([20]), "same"),
    ]
    return (
        doc_q,
        doc_k,
        doc_v,
        query_q,
        query_k,
        query_v,
        documents,
        torch.arange(100, 100 + query_len),
        decode_q,
        decode_k,
        decode_v,
        torch.arange(100 + query_len, 100 + query_len + decode_len),
    )


def _dense_reference(case, variant: str):
    doc_q, doc_k, doc_v, qq, qk, qv, docs, _, dq, dk, dv, _ = case
    q_docs, k_docs, v_docs = [], [], []
    offset = 0
    for segment in docs:
        length = segment.cache_slots.numel()
        q_docs.append(doc_q[offset : offset + length])
        k_docs.append(doc_k[segment.cache_slots])
        v_docs.append(doc_v[segment.cache_slots])
        offset += length
    q = torch.cat([*q_docs, qq, dq])
    k = torch.cat([*k_docs, qk, dk])
    v = torch.cat([*v_docs, qv, dv])
    allow = build_mask(
        [segment.cache_slots.numel() for segment in docs],
        qq.shape[0],
        dq.shape[0],
        variant,
    )
    doc_total = sum(segment.cache_slots.numel() for segment in docs)
    groups = q.shape[1] // k.shape[1]
    k_expanded = k.repeat_interleave(groups, dim=1)
    v_expanded = v.repeat_interleave(groups, dim=1)
    output = (
        F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k_expanded.transpose(0, 1).unsqueeze(0),
            v_expanded.transpose(0, 1).unsqueeze(0),
            attn_mask=allow.unsqueeze(0).unsqueeze(0),
            is_causal=False,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    return (
        output[:doc_total],
        output[doc_total : doc_total + qq.shape[0]],
        output[doc_total + qq.shape[0] :],
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("variant", ["full", "ddi_qf", "blind"])
def test_segmented_matches_build_mask_dense_reference_for_adversarial_unequal_layout(
    dtype, variant
):
    case = _case(dtype)
    actual = segmented_attention(
        *case[:8],
        variant=variant,
        decode_q=case[8],
        decode_k=case[9],
        decode_v=case[10],
        decode_positions=case[11],
    )
    expected = _dense_reference(case, variant)
    tolerance = 2e-12 if dtype == torch.float64 else 3e-6
    assert torch.allclose(actual.document, expected[0], atol=tolerance, rtol=tolerance)
    assert torch.allclose(actual.query, expected[1], atol=tolerance, rtol=tolerance)
    assert actual.decode is not None
    assert torch.allclose(actual.decode, expected[2], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("variant", ["full", "ddi_qf", "blind"])
def test_empty_query_and_no_decode_are_well_defined(variant):
    case = _case(torch.float64, query_len=0, decode_len=0)
    actual = segmented_attention(*case[:8], variant=variant)
    expected = _dense_reference(case, variant)
    assert actual.query.shape == (0, 4, 3)
    assert actual.decode is None
    assert torch.allclose(actual.document, expected[0], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("variant", ["ddi_qf", "blind"])
def test_decomposed_matches_dense_for_irregular_packed_duplicate_documents(variant):
    case = _case(torch.float64)
    dense = segmented_attention(
        *case[:8],
        variant=variant,
        decode_q=case[8],
        decode_k=case[9],
        decode_v=case[10],
        decode_positions=case[11],
    )
    decomposed = segmented_attention_decomposed(
        *case[:8],
        variant=variant,
        decode_q=case[8],
        decode_k=case[9],
        decode_v=case[10],
        decode_positions=case[11],
    )
    assert torch.allclose(decomposed.document, dense.document, atol=2e-12, rtol=2e-12)
    assert torch.allclose(decomposed.query, dense.query, atol=2e-12, rtol=2e-12)
    assert decomposed.decode is not None and dense.decode is not None
    assert torch.allclose(decomposed.decode, dense.decode, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("variant", ["ddi_qf", "blind"])
def test_decomposed_empty_query_and_decode_match_dense(variant):
    case = _case(torch.float64, query_len=0, decode_len=0)
    dense = segmented_attention(*case[:8], variant=variant)
    decomposed = segmented_attention_decomposed(*case[:8], variant=variant)
    assert decomposed.query.shape == (0, 4, 3)
    assert decomposed.decode is None
    assert torch.allclose(decomposed.document, dense.document, atol=2e-12, rtol=2e-12)


def test_decomposed_rejects_full_causal_attention():
    case = _case(torch.float64)
    with pytest.raises(ValueError, match="attend across documents"):
        segmented_attention_decomposed(*case[:8], variant="full")


def test_full_document_rows_are_not_a_sparse_document_only_path():
    case = _case(torch.float64, query_len=0, decode_len=0)
    changed = list(case)
    changed_k = case[1].clone()
    changed_v = case[2].clone()
    first_document_slots = case[6][0].cache_slots
    changed_k[first_document_slots] += 10.0
    changed_v[first_document_slots] += 10.0
    changed[1], changed[2] = changed_k, changed_v
    full_before = segmented_attention(*case[:8], variant="full").document
    full_after = segmented_attention(*changed[:8], variant="full").document
    qf_before = segmented_attention(*case[:8], variant="ddi_qf").document
    qf_after = segmented_attention(*changed[:8], variant="ddi_qf").document
    second_document_start = case[6][0].cache_slots.numel()
    assert not torch.allclose(
        full_before[second_document_start:], full_after[second_document_start:]
    )
    assert torch.allclose(qf_before[second_document_start:], qf_after[second_document_start:])


def test_lse_merge_matches_direct_attention_and_naive_average_fails():
    q = torch.tensor([[[1.0]]], dtype=torch.float64)
    near_k = torch.tensor([[[0.0]]], dtype=torch.float64)
    near_v = torch.tensor([[[0.0]]], dtype=torch.float64)
    far_k = torch.tensor([[[8.0]]], dtype=torch.float64)
    far_v = torch.tensor([[[10.0]]], dtype=torch.float64)
    # Build the same direct query result with two document components.
    documents = [
        DocumentSegment(torch.tensor([0]), torch.tensor([0])),
        DocumentSegment(torch.tensor([1]), torch.tensor([0])),
    ]
    direct = segmented_attention(
        torch.zeros(2, 1, 1, dtype=torch.float64),
        torch.cat([near_k, far_k]),
        torch.cat([near_v, far_v]),
        q,
        near_k,
        near_v,
        documents,
        torch.tensor([1]),
        variant="ddi_qf",
    ).query
    # The exposed primitive receives independently normalized component outputs and LSEs.
    partials = [
        (near_v, torch.tensor([[0.0]], dtype=torch.float64)),
        (far_v, torch.tensor([[8.0]], dtype=torch.float64)),
        (near_v, torch.tensor([[0.0]], dtype=torch.float64)),
    ]
    merged = merge_lse(partials)
    assert torch.allclose(merged, direct, atol=1e-12)
    assert not torch.allclose((near_v + far_v) / 2, direct, atol=1e-3)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda c: c[:1] + (c[1].unsqueeze(0),) + c[2:], "unpadded"),
        (
            lambda c: (
                c[0][:, :3],
                c[1],
                c[2],
                c[3][:, :3],
                c[4],
                c[5],
                *c[6:],
            ),
            "GQA",
        ),
        (lambda c: (c[0].clone().fill_(float("nan")), *c[1:]), "finite"),
    ],
)
def test_rejects_padding_invalid_gqa_and_nonfinite(mutator, match):
    case = mutator(_case(torch.float32))
    with pytest.raises(ValueError, match=match):
        segmented_attention(*case[:8], variant="ddi_qf")


def test_rejects_masks_and_invalid_positions_or_cache_cover():
    case = _case(torch.float32)
    with pytest.raises(ValueError, match="does not accept masks"):
        segmented_attention(*case[:8], variant="ddi_qf", attention_mask=torch.ones(1))
    bad_positions = [DocumentSegment(torch.tensor([4, 1]), torch.tensor([10, 10])), *case[6][1:]]
    with pytest.raises(ValueError, match="strictly increasing"):
        segmented_attention(*case[:6], bad_positions, case[7], variant="ddi_qf")
    bad_cover = [DocumentSegment(torch.tensor([4, 4]), torch.tensor([10, 11])), *case[6][1:]]
    with pytest.raises(ValueError, match="duplicate-free cover"):
        segmented_attention(*case[:6], bad_cover, case[7], variant="ddi_qf")
