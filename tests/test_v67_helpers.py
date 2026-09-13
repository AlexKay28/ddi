"""V67 pure-helper tests: metrics, gate logic, link mask shapes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ddi.link_token_contract import (
    DocumentPayload,
    LinkTokenContract,
    build_request_layout,
)
from ddi.metrics import exact_match, token_f1, answer_presence, normalize


# ---------------------------------------------------------------------------
# Metrics pure helper tests
# ---------------------------------------------------------------------------


class TestMetricsHelpers:
    def test_exact_match_basic(self):
        assert exact_match("Paris", ["Paris"]) == 1.0
        assert exact_match("paris", ["Paris"]) == 1.0
        assert exact_match("the Paris", ["Paris"]) == 1.0
        assert exact_match("London", ["Paris"]) == 0.0

    def test_exact_match_empty_pred(self):
        assert exact_match("", ["Paris"]) == 0.0

    def test_exact_match_multiple_candidates(self):
        assert exact_match("Rome", ["Paris", "Rome"]) == 1.0
        assert exact_match("Rome", ["Paris", "London"]) == 0.0

    def test_token_f1_basic(self):
        assert token_f1("Paris", ["Paris"]) == 1.0
        assert token_f1("Paris France", ["Paris"]) == 2/3  # precision=1/2, recall=1/1
        assert 0 < token_f1("Paris", ["Paris France"]) < 1.0

    def test_token_f1_empty_pred(self):
        assert token_f1("", ["Paris"]) == 0.0

    def test_answer_presence_basic(self):
        assert answer_presence("Paris is in France", ["Paris"]) == 1.0
        assert answer_presence("London is in England", ["Paris"]) == 0.0

    def test_normalize_strips_articles_and_punct(self):
        assert normalize("The, Paris!") == "paris"
        assert normalize("a an the") == ""
        assert normalize("Hello World") == "hello world"


# ---------------------------------------------------------------------------
# Gate logic tests
# ---------------------------------------------------------------------------


class TestGateLogic:
    def test_noninferiority_pass(self):
        ci_low = -0.01
        margin = -0.02
        assert ci_low >= margin

    def test_noninferiority_fail(self):
        ci_low = -0.05
        margin = -0.02
        assert not (ci_low >= margin)

    def test_noninferiority_both_pass(self):
        em_ci_low = -0.01
        f1_ci_low = -0.015
        margin = -0.02
        assert em_ci_low >= margin and f1_ci_low >= margin

    def test_noninferiority_one_fails(self):
        em_ci_low = -0.01
        f1_ci_low = -0.03
        margin = -0.02
        assert not (em_ci_low >= margin and f1_ci_low >= margin)

    def test_dev_selection_by_em_then_f1(self):
        arms = {
            "k0": {"em": 0.56, "f1": 0.713},
            "k1": {"em": 0.57, "f1": 0.714},
            "k5": {"em": 0.57, "f1": 0.719},
        }
        best = max(arms.keys(), key=lambda a: (arms[a]["em"], arms[a]["f1"]))
        assert best == "k5"


# ---------------------------------------------------------------------------
# Link mask shape tests
# ---------------------------------------------------------------------------


class TestLinkMaskShapes:
    @pytest.fixture
    def contract(self):
        return LinkTokenContract("test-cache", (90, 91, 92, 93, 94))

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_mask_is_square(self, contract, k):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31),
            (40,),
            k=k,
        )
        T = len(layout.allow)
        assert all(len(row) == T for row in layout.allow)

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_physical_tokens_match_logical_size(self, contract, k):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31),
            (40,),
            k=k,
        )
        assert len(layout.physical_token_ids) == len(layout.allow)
        assert len(layout.positions) == len(layout.allow)
        assert len(layout.logical_order) == len(layout.allow)

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_total_size_formula(self, contract, k):
        docs = (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22)))
        query = (30, 31)
        answer = (40,)
        layout = build_request_layout(contract, docs, query, answer, k=k)
        doc_total = sum(len(d.token_ids) for d in docs)
        link_total = len(docs) * k
        expected = doc_total + link_total + len(query) + len(answer)
        assert len(layout.physical_token_ids) == expected

    def test_k0_has_no_link_tokens(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30,),
            k=0,
        )
        assert 90 not in layout.physical_token_ids
        assert 91 not in layout.physical_token_ids

    def test_k1_has_one_link_per_doc(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30,),
            k=1,
        )
        link_count = sum(1 for t in layout.physical_token_ids if t == 90)
        assert link_count == 2  # 2 docs × k=1

    def test_k5_has_five_links_per_doc(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30,),
            k=5,
        )
        link_tokens = [t for t in layout.physical_token_ids if t in (90, 91, 92, 93, 94)]
        assert len(link_tokens) == 10  # 2 docs × k=5
