"""Tests for ddi.scoring — ProphetKV-style query-driven token scoring.

Tests are CPU-only and use mock tensors to verify the selection logic
without requiring a real model or GPU.
"""

import pytest
import torch

from ddi.scoring import _fuse_attention_scores, select_top_k


def test_select_top_k_basic():
    """Top-p selection picks the highest-scoring tokens globally."""
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    doc_lengths = [3, 3]
    result = select_top_k(scores, doc_lengths, top_p=0.5)
    # 6 tokens, top_p=0.5 -> k=3 -> tokens 1 (0.9), 3 (0.8), 5 (0.7)
    # token 1 is in doc 0 (offset 0) at intra-doc index 1
    # token 3 is in doc 1 (offset 3) at intra-doc index 0
    # token 5 is in doc 1 (offset 3) at intra-doc index 2
    assert result == [{1}, {0, 2}]


def test_select_top_k_global_not_per_document():
    """Selection is global across all documents (ProphetKV Algorithm 1)."""
    # Doc 1 has all high scores, doc 2 has all low.
    scores = torch.tensor([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    doc_lengths = [3, 3]
    result = select_top_k(scores, doc_lengths, top_p=0.5)
    # k=3, top 3 are 0,1,2 (all in doc 1)
    assert result == [{0, 1, 2}, set()]


def test_select_top_k_min_one_token():
    """At least 1 token is always selected."""
    scores = torch.tensor([0.1, 0.1, 0.1, 0.1])
    doc_lengths = [2, 2]
    result = select_top_k(scores, doc_lengths, top_p=0.01)
    # k = max(1, 0) = 1
    total_selected = sum(len(s) for s in result)
    assert total_selected == 1


def test_select_top_k_all_tokens():
    """top_p=1.0 selects all tokens."""
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    doc_lengths = [2, 2]
    result = select_top_k(scores, doc_lengths, top_p=1.0)
    assert result == [{0, 1}, {0, 1}]


def test_fuse_attention_scores_uniform_fusion():
    """Fusion averages across heads and query positions (ProphetKV Eq. 7)."""
    # 2 layers, 1 head, 2 query tokens, 3 doc tokens
    # Layer 0: uniform attention (each query token attends equally to all docs)
    # Layer 1: same
    uniform = torch.ones(1, 1, 2, 3) / 3  # [batch, heads, query_seq, key_seq]
    # doc_total=3, query_len=2
    scores = _fuse_attention_scores((uniform, uniform), doc_total=3, query_len=2)
    assert scores.shape == (3,)
    # Uniform attention: each token gets 1/3 per query, averaged = 1/3
    assert torch.allclose(scores, torch.tensor([1 / 3, 1 / 3, 1 / 3]))


def test_fuse_attention_scores_concentrated():
    """When query attends only to token 0, that token gets the highest score."""
    # 1 layer, 1 head, 1 query token, 3 doc tokens
    attn = torch.zeros(1, 1, 1, 3)
    attn[0, 0, 0, 0] = 1.0  # query attends only to doc token 0
    scores = _fuse_attention_scores((attn,), doc_total=3, query_len=1)
    assert scores[0].item() == pytest.approx(1.0)
    assert scores[1].item() == pytest.approx(0.0)
    assert scores[2].item() == pytest.approx(0.0)


def test_fuse_attention_scores_multi_head():
    """Averaging across heads gives the mean attention pattern."""
    # 2 heads with different focus
    attn = torch.zeros(1, 2, 1, 2)
    attn[0, 0, 0, 0] = 1.0  # head 0: attends to token 0
    attn[0, 1, 0, 1] = 1.0  # head 1: attends to token 1
    scores = _fuse_attention_scores((attn,), doc_total=2, query_len=1)
    assert scores[0].item() == pytest.approx(0.5)
    assert scores[1].item() == pytest.approx(0.5)
