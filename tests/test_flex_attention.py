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
    actual = make_block_mask(
        doc_lengths, query_len, variant, device=torch.device("cpu"), block_size=1
    ).to_dense()[0, 0].to(torch.bool)
    assert torch.equal(actual, expected)
