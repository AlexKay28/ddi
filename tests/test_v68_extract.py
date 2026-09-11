"""CPU-only unit tests for the V68 mechanical extraction pure functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from v68_extract_differing import (  # noqa: E402
    first_divergence_index,
    gold_in_prediction,
    is_token_prefix,
    levenshtein,
    normalize_ws,
    token_ops,
    whitespace_tokens,
)


def test_first_divergence_index_identical():
    assert first_divergence_index([1, 2, 3], [1, 2, 3]) is None


def test_first_divergence_index_at_zero():
    assert first_divergence_index([9, 2, 3], [1, 2, 3]) == 0


def test_first_divergence_index_midway_and_length_diff():
    assert first_divergence_index([1, 2, 9], [1, 2, 3]) == 2
    assert first_divergence_index([1, 2], [1, 2, 3]) == 2
    assert first_divergence_index([1, 2, 3, 4], [1, 2, 3]) == 3


def test_whitespace_tokens():
    assert whitespace_tokens("a  b\nc\t d") == ["a", "b", "c", "d"]
    assert whitespace_tokens("") == []


def test_normalize_ws():
    assert normalize_ws("  Foo BAR\t baz ") == "foo bar baz"
    assert normalize_ws("X") == normalize_ws("x")


def test_gold_in_prediction_containment():
    assert gold_in_prediction("The answer is Women.", ["women"]) is True
    assert gold_in_prediction("Paris", ["London", "paris"]) is True
    assert gold_in_prediction("Paris", ["London"]) is False
    assert gold_in_prediction("anything", []) is False
    # multi-word gold with collapsed whitespace
    assert gold_in_prediction("New\n  York City", ["new york city"]) is True


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("same", "same") == 0


def test_token_ops_prefix_insertion():
    ops = token_ops(["a", "b"], ["a", "b", "c"])
    assert ops["equal"] == 2 and ops["insert"] == 1 and ops["delete"] == 0 and ops["replace"] == 0
    assert ops["n_nonequal_ops"] == 1


def test_token_ops_replacement():
    ops = token_ops(["x", "y"], ["x", "z"])
    assert ops["equal"] == 1 and ops["replace"] == 1
    assert ops["n_nonequal_ops"] == 1


def test_is_token_prefix():
    assert is_token_prefix(["a", "b"], ["a", "b", "c"]) is True
    assert is_token_prefix(["a", "b"], ["a", "b"]) is True
    assert is_token_prefix(["a", "c"], ["a", "b", "c"]) is False
    assert is_token_prefix(["a", "b", "c"], ["a", "b"]) is False
