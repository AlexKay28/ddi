"""Tests for the metric implementations (issue #11, Phase 1.3).

Contract: every metric behavior called out in the acceptance criteria has a test here, and
the numeric expectations are computed by hand rather than routed back through the module —
a bug in `normalize` cannot cover for a bug in `token_f1` if the F1 golden numbers are
literal.
"""

from __future__ import annotations

import pytest
import torch

from ddi.metrics import (
    answer_presence,
    exact_match,
    make_answer_labels,
    normalize,
    token_f1,
)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_lowercases():
    assert normalize("Paris") == "paris"
    assert normalize("PARIS") == "paris"


def test_normalize_strips_punctuation():
    assert normalize("Paris, France.") == "paris france"
    assert normalize("!?.,;:") == ""


def test_normalize_strips_articles():
    assert normalize("the cat") == "cat"
    assert normalize("a cat") == "cat"
    assert normalize("an cat") == "cat"
    assert normalize("The Cat") == "cat"


def test_normalize_articles_only_as_whole_tokens():
    # "theatre" contains "the" as a substring; it must survive normalization intact.
    assert normalize("theatre") == "theatre"
    assert normalize("another") == "another"


def test_normalize_collapses_whitespace():
    assert normalize("hello   world") == "hello world"
    assert normalize("hello\tworld\n\nfoo") == "hello world foo"
    assert normalize("  leading and trailing  ") == "leading and trailing"


def test_normalize_idempotent():
    for text in [
        "The Quick Brown Fox!",
        "  hello,   world  ",
        "An apple a day",
        "",
        "the",
    ]:
        assert normalize(normalize(text)) == normalize(text)


def test_normalize_empty_string():
    assert normalize("") == ""
    assert normalize("   ") == ""
    assert normalize("the a an") == ""


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


def test_exact_match_identical():
    assert exact_match("paris", ["paris"]) == 1.0


def test_exact_match_case_insensitive():
    assert exact_match("Paris", ["paris"]) == 1.0
    assert exact_match("PARIS", ["Paris"]) == 1.0


def test_exact_match_punctuation_insensitive():
    assert exact_match("Paris.", ["paris"]) == 1.0
    assert exact_match("Paris, France!", ["paris france"]) == 1.0


def test_exact_match_article_insensitive():
    assert exact_match("The cat", ["cat"]) == 1.0
    assert exact_match("a cat", ["the cat"]) == 1.0


def test_exact_match_no_match():
    assert exact_match("paris", ["london"]) == 0.0
    assert exact_match("paris france", ["france paris"]) == 0.0  # order matters for EM


def test_exact_match_multiple_candidates_any_match():
    assert exact_match("paris", ["london", "berlin", "paris"]) == 1.0
    assert exact_match("paris", ["london", "berlin"]) == 0.0


def test_exact_match_order_invariant_across_candidates():
    a = exact_match("paris", ["london", "berlin", "paris"])
    b = exact_match("paris", ["paris", "berlin", "london"])
    c = exact_match("paris", ["berlin", "paris", "london"])
    assert a == b == c == 1.0


def test_exact_match_empty_pred():
    assert exact_match("", ["paris"]) == 0.0


def test_exact_match_empty_candidates_raises():
    with pytest.raises(ValueError):
        exact_match("paris", [])


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------


def test_token_f1_perfect_match():
    assert token_f1("paris france", ["paris france"]) == 1.0


def test_token_f1_perfect_match_after_normalization():
    assert token_f1("The Paris, France!", ["paris france"]) == 1.0


def test_token_f1_zero_overlap():
    assert token_f1("paris", ["london"]) == 0.0
    assert token_f1("cat dog", ["fish bird"]) == 0.0


def test_token_f1_partial_overlap_by_hand():
    # pred tokens after normalization: ["paris", "france", "europe"]  (len 3)
    # gold tokens: ["paris", "france"]                                (len 2)
    # matched = 2; precision = 2/3; recall = 2/2 = 1.0
    # F1 = 2 * (2/3 * 1) / (2/3 + 1) = (4/3) / (5/3) = 4/5 = 0.8
    assert token_f1("paris france europe", ["paris france"]) == pytest.approx(0.8)


def test_token_f1_partial_overlap_symmetric():
    # pred tokens: ["paris", "france"]                (len 2)
    # gold tokens: ["paris", "france", "europe"]      (len 3)
    # matched = 2; precision = 1.0; recall = 2/3; F1 = 0.8
    assert token_f1("paris france", ["paris france europe"]) == pytest.approx(0.8)


def test_token_f1_multiset_semantics():
    # pred: ["a", "a", "b"] -> after article strip: ["b"]  -- articles ARE stripped.
    # Use non-article repeats instead to exercise multiset counting.
    # pred tokens: ["cat", "cat", "dog"] (len 3)
    # gold tokens: ["cat", "dog"]        (len 2)
    # matched = 2 (min(2,1) for cat + min(1,1) for dog); precision = 2/3, recall = 1
    # F1 = 0.8
    assert token_f1("cat cat dog", ["cat dog"]) == pytest.approx(0.8)


def test_token_f1_takes_max_over_candidates():
    # First candidate scores 0.5, second scores 1.0 → max wins.
    assert token_f1("paris france", ["paris berlin", "paris france"]) == 1.0


def test_token_f1_case_and_whitespace_insensitive():
    assert token_f1("Paris   France", ["paris france"]) == 1.0
    assert token_f1("PARIS\tFRANCE", ["paris\nfrance"]) == 1.0


def test_token_f1_empty_pred():
    assert token_f1("", ["paris"]) == 0.0


def test_token_f1_empty_candidates_raises():
    with pytest.raises(ValueError):
        token_f1("paris", [])


def test_token_f1_order_invariant_across_candidates():
    a = token_f1("paris france", ["berlin", "paris france", "london"])
    b = token_f1("paris france", ["paris france", "berlin", "london"])
    c = token_f1("paris france", ["london", "berlin", "paris france"])
    assert a == b == c == 1.0


# ---------------------------------------------------------------------------
# answer_presence
# ---------------------------------------------------------------------------


def test_answer_presence_substring_hit():
    assert answer_presence("the answer is paris", ["paris"]) == 1.0
    assert answer_presence("I think paris france is nice", ["paris france"]) == 1.0


def test_answer_presence_no_hit():
    assert answer_presence("the answer is london", ["paris"]) == 0.0


def test_answer_presence_case_insensitive():
    assert answer_presence("The answer is PARIS", ["paris"]) == 1.0
    assert answer_presence("the answer is paris", ["Paris"]) == 1.0


def test_answer_presence_punctuation_insensitive():
    assert answer_presence("The answer is: Paris!", ["paris"]) == 1.0
    assert answer_presence("Paris, France.", ["paris france"]) == 1.0


def test_answer_presence_multiple_candidates():
    assert answer_presence("The answer is paris", ["london", "paris", "berlin"]) == 1.0
    assert answer_presence("The answer is paris", ["london", "berlin"]) == 0.0


def test_answer_presence_empty_pred():
    assert answer_presence("", ["paris"]) == 0.0


def test_answer_presence_empty_candidates_raises():
    with pytest.raises(ValueError):
        answer_presence("paris", [])


def test_answer_presence_order_invariant_across_candidates():
    a = answer_presence("the answer is paris", ["london", "paris", "berlin"])
    b = answer_presence("the answer is paris", ["paris", "london", "berlin"])
    c = answer_presence("the answer is paris", ["berlin", "london", "paris"])
    assert a == b == c == 1.0


def test_answer_presence_softer_than_exact_match():
    # Wrapping the answer in a sentence: presence should hit even when EM misses.
    pred = "The answer to your question is Paris."
    assert exact_match(pred, ["paris"]) == 0.0
    assert answer_presence(pred, ["paris"]) == 1.0


# ---------------------------------------------------------------------------
# make_answer_labels
# ---------------------------------------------------------------------------


def test_make_answer_labels_masks_outside_range():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long)
    labels = make_answer_labels(input_ids, answer_start=2, answer_end=5)
    expected = torch.tensor([[-100, -100, 12, 13, 14, -100]], dtype=torch.long)
    assert labels.equal(expected)


def test_make_answer_labels_inside_matches_input_ids():
    input_ids = torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.long)
    labels = make_answer_labels(input_ids, 1, 4)
    assert labels[0, 1].item() == 8
    assert labels[0, 2].item() == 9
    assert labels[0, 3].item() == 10


def test_make_answer_labels_dtype_is_long():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = make_answer_labels(input_ids, 0, 2)
    assert labels.dtype == torch.long


def test_make_answer_labels_dtype_long_even_if_input_is_int32():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    labels = make_answer_labels(input_ids, 0, 2)
    assert labels.dtype == torch.long


def test_make_answer_labels_preserves_device():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = make_answer_labels(input_ids, 1, 3)
    assert labels.device == input_ids.device


def test_make_answer_labels_full_range():
    input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    labels = make_answer_labels(input_ids, 0, 3)
    assert labels.equal(input_ids)


def test_make_answer_labels_batch_dim_supported():
    # Phase 1 uses batch=1 in practice, but the shape signature stays 2D.
    input_ids = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long
    )
    labels = make_answer_labels(input_ids, 1, 3)
    expected = torch.tensor(
        [[-100, 11, 12, -100], [-100, 21, 22, -100]], dtype=torch.long
    )
    assert labels.equal(expected)


def test_make_answer_labels_start_negative_raises():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with pytest.raises(ValueError):
        make_answer_labels(input_ids, -1, 2)


def test_make_answer_labels_end_beyond_seq_raises():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with pytest.raises(ValueError):
        make_answer_labels(input_ids, 1, 5)


def test_make_answer_labels_start_ge_end_raises():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with pytest.raises(ValueError):
        make_answer_labels(input_ids, 2, 2)
    with pytest.raises(ValueError):
        make_answer_labels(input_ids, 3, 2)


def test_make_answer_labels_rejects_non_2d():
    with pytest.raises(ValueError):
        make_answer_labels(torch.tensor([1, 2, 3, 4], dtype=torch.long), 0, 2)
    with pytest.raises(ValueError):
        make_answer_labels(torch.zeros((1, 4, 2), dtype=torch.long), 0, 2)
