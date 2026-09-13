"""Tests for ddi.deferred_rope — LazyAttention-style deferred positional encoding.

Tests are CPU-only and verify the rotation math against known RoPE properties
without requiring a real model or GPU.
"""

import pytest
import torch

from ddi.deferred_rope import (
    _rotate_half,
    compute_global_positions,
    rotate_keys,
    rotate_query,
)


def test_rotate_half_shape():
    """rotate_half preserves shape and negates the second half."""
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    result = _rotate_half(x)
    assert result.shape == x.shape
    # rotate_half([1,2,3,4]) = [-3,-4,1,2]
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    assert torch.allclose(result, expected)


def test_rotate_keys_identity_at_zero():
    """Rotation at position 0 is identity (cos(0)=1, sin(0)=0)."""
    keys = torch.randn(1, 2, 3, 4)
    positions = torch.zeros(3, dtype=torch.long)
    rotated = rotate_keys(keys, positions, rope_theta=10000.0, dtype=torch.float32)
    assert torch.allclose(rotated, keys, atol=1e-5)


def test_rotate_keys_changes_values():
    """Rotation at nonzero positions changes key values."""
    keys = torch.randn(1, 2, 3, 4)
    positions = torch.tensor([5, 10, 15], dtype=torch.long)
    rotated = rotate_keys(keys, positions, rope_theta=10000.0, dtype=torch.float32)
    assert not torch.allclose(rotated, keys, atol=1e-3)


def test_rotate_keys_relative_property():
    """RoPE relative rotation: (R_m q)^T (R_n k) = q^T R_{n-m} k.

    If we rotate keys at global positions and query at its position,
    the attention score equals the score with relative offset rotation.
    """
    head_dim = 8
    rope_theta = 10000.0
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Query at position 10, key at position 3 -> relative offset = 7
    q_pos = torch.tensor([10], dtype=torch.long)
    k_pos = torch.tensor([3], dtype=torch.long)

    q_rotated = rotate_query(q, q_pos, rope_theta, dtype=torch.float32)
    k_rotated = rotate_keys(k, k_pos, rope_theta, dtype=torch.float32)

    # Attention score with both rotated
    score_both = (q_rotated * k_rotated).sum()

    # Now compute with un-rotated k and rotation at relative offset 7
    # R_{n-m} = R_{3-10} = R_{-7}
    rel_pos = torch.tensor([7], dtype=torch.long)  # |n-m| = 7
    # Rotate q at position 7, keep k un-rotated
    q_rel = rotate_query(q, rel_pos, rope_theta, dtype=torch.float32)
    score_rel = (q_rel * k).sum()

    # These should be equal (RoPE relative property)
    assert torch.allclose(score_both, score_rel, atol=1e-4)


def test_compute_global_positions_sequential():
    """Sequential layout gives 0, 1, 2, ... for all tokens."""
    positions = compute_global_positions([3, 5], 2, layout="sequential")
    assert positions.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_compute_global_positions_restart():
    """Restart layout gives per-doc positions starting at 0."""
    positions = compute_global_positions([3, 5], 2, layout="restart")
    # Doc 1: [0,1,2], Doc 2: [0,1,2,3,4], Query: [5,6] (max(doc_lengths)=5)
    assert positions.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5, 6]


def test_compute_global_positions_invalid_layout():
    """Invalid layout raises ValueError."""
    with pytest.raises(ValueError, match="unknown layout"):
        compute_global_positions([3], 1, layout="invalid")


def test_rotate_keys_preserves_shape():
    """Rotation preserves the input tensor shape."""
    keys = torch.randn(2, 4, 10, 64)
    positions = torch.arange(10, dtype=torch.long)
    rotated = rotate_keys(keys, positions, rope_theta=10000.0, dtype=torch.float32)
    assert rotated.shape == keys.shape


def test_rotate_query_single_token():
    """rotate_query works on a single decode-step token."""
    q = torch.randn(1, 1, 1, 64)
    pos = torch.tensor([42], dtype=torch.long)
    rotated = rotate_query(q, pos, rope_theta=10000.0, dtype=torch.float32)
    assert rotated.shape == q.shape
    assert not torch.allclose(rotated, q, atol=1e-3)
