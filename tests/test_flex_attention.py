import pytest
import torch

from ddi.flex_attention import make_block_mask
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
