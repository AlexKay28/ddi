import pytest
import torch

from ddi.flex_attention import make_batched_block_mask, make_block_mask
from ddi.masks import build_mask


@pytest.mark.parametrize("variant", ["full", "ddi_qf", "blind"])
@pytest.mark.parametrize("block_size", [1, 2, 4, (4, 2), (2, 4), (3, 5)])
@pytest.mark.parametrize("doc_lengths,query_len", [([5], 1), ([3, 2], 2), ([2, 5], 0)])
def test_flex_block_mask_matches_canonical_allow_matrix(
    variant: str, block_size: int | tuple[int, int], doc_lengths: list[int], query_len: int
) -> None:
    expected = build_mask(doc_lengths, query_len, 0, variant)
    block_mask = make_block_mask(
        doc_lengths, query_len, variant, device=torch.device("cpu"), block_size=block_size
    )
    positions = torch.arange(expected.shape[0])
    actual = block_mask.mask_mod(0, 0, positions[:, None], positions[None, :])
    assert torch.equal(actual, expected)
    # The sparse block adjacency may contain conservative candidates, but it
    # must never omit a token pair the canonical predicate permits.
    q_block, kv_block = block_mask.BLOCK_SIZE
    candidates = block_mask.to_dense()[0, 0].to(torch.bool)
    token_candidates = candidates[
        positions[:, None] // q_block, positions[None, :] // kv_block
    ]
    assert torch.all(token_candidates[expected])


@pytest.mark.parametrize("variant", ["full", "ddi_qf", "blind"])
@pytest.mark.parametrize("block_size", [1, 2, 4, (4, 2), (2, 4), (3, 5)])
def test_batched_flex_block_masks_match_each_sample_allow_matrix(
    variant: str, block_size: int | tuple[int, int]
) -> None:
    doc_lengths = [[5], [3, 2], [2, 1, 2], [1], [2, 5]]
    query_lens = [1, 2, 4, 1, 0]
    total = max(
        sum(lengths) + query_len
        for lengths, query_len in zip(doc_lengths, query_lens, strict=True)
    )
    block_mask = make_batched_block_mask(
        doc_lengths, query_lens, variant, device=torch.device("cpu"), block_size=block_size
    )
    dense = block_mask.to_dense()
    q_block, kv_block = block_mask.BLOCK_SIZE
    positions = torch.arange(total)
    for batch, (lengths, query_len) in enumerate(zip(doc_lengths, query_lens, strict=True)):
        sample = build_mask(lengths, query_len, 0, variant)
        # Full stays causal through padding; isolated variants use padding self-attention.
        expected = (
            torch.ones((total, total), dtype=torch.bool).tril()
            if variant == "full"
            else torch.eye(total, dtype=torch.bool)
        )
        expected[: sample.shape[0], : sample.shape[1]] = sample
        actual = block_mask.mask_mod(batch, 0, positions[:, None], positions[None, :])
        assert torch.equal(actual, expected)
        token_candidates = dense[batch, 0][
            positions[:, None] // q_block, positions[None, :] // kv_block
        ].to(torch.bool)
        assert torch.all(token_candidates[expected])


@pytest.mark.parametrize("batched", [False, True])
def test_flex_block_mask_adjusts_non_aligned_sequence_length_when_supported(batched: bool) -> None:
    if batched:
        block_mask = make_batched_block_mask(
            [[3, 2], [2, 1]], [2, 1], "ddi_qf", device=torch.device("cpu"), block_size=4
        )
        expected = 7
    else:
        block_mask = make_block_mask(
            [3, 2], 2, "ddi_qf", device=torch.device("cpu"), block_size=4
        )
        expected = 7
    if hasattr(block_mask, "_adjust"):
        assert block_mask.shape[-2:] == (expected, expected)
