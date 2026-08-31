import pytest
import torch

from ddi.flex_attention import make_batched_block_mask, make_block_mask
from ddi.masks import build_mask


@pytest.mark.parametrize("variant", ["full", "ddi_qf", "blind"])
def test_flex_block_mask_matches_canonical_allow_matrix(variant: str) -> None:
    """One-token flex blocks expose the exact mask semantics for regression tests."""
    doc_lengths = [3, 2]
    query_len = 2
    expected = build_mask(doc_lengths, query_len, 0, variant)
    block_mask = make_block_mask(
        doc_lengths, query_len, variant, device=torch.device("cpu"), block_size=1
    )
    positions = torch.arange(expected.shape[0])
    actual = block_mask.mask_mod(0, 0, positions[:, None], positions[None, :])
    assert torch.equal(actual, expected)
    # The sparse block adjacency may contain conservative candidates, but it
    # must never omit a token pair the canonical predicate permits.
    assert torch.all(block_mask.to_dense()[0, 0].to(torch.bool)[expected])


@pytest.mark.parametrize("variant", ["full", "ddi_qf", "blind"])
def test_batched_flex_block_masks_match_each_sample_allow_matrix(variant: str) -> None:
    doc_lengths = [[3, 2], [2, 1, 2]]
    query_lens = [2, 3]
    block_mask = make_batched_block_mask(
        doc_lengths, query_lens, variant, device=torch.device("cpu"), block_size=1
    )
    dense = block_mask.to_dense()
    for batch, (lengths, query_len) in enumerate(zip(doc_lengths, query_lens, strict=True)):
        expected = build_mask(lengths, query_len, 0, variant)
        positions = torch.arange(expected.shape[0])
        actual = block_mask.mask_mod(batch, 0, positions[:, None], positions[None, :])
        assert torch.equal(actual, expected)
        actual_blocks = dense[batch, 0, : expected.shape[0], : expected.shape[0]]
        assert torch.all(actual_blocks[expected])
