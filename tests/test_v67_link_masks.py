"""V67 mask/position reference tests for the link-token mechanism.

2-document hand-checkable cases + causal-leakage tests.
MUST pass before any V67 training.
"""

from __future__ import annotations

import pytest
import torch

from ddi.link_token_contract import (
    DocumentPayload,
    LinkTokenContract,
    build_request_layout,
)


@pytest.fixture
def contract() -> LinkTokenContract:
    return LinkTokenContract("v67-test-cache", (90, 91, 92, 93, 94))


# ---------------------------------------------------------------------------
# 2-document hand-checkable cases
# ---------------------------------------------------------------------------


class TestK1TwoDocumentHandCheck:
    """Hand-checkable mask for k=1 with two documents of length 2 and 1."""

    def test_physical_layout(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        # Physical: doc0(10,11), doc1(20), link0(90), link1(90), query(30,31), answer(40)
        assert layout.physical_token_ids == (10, 11, 20, 90, 90, 30, 31, 40)
        assert len(layout.physical_token_ids) == 8

    def test_positions(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        # max_doc_length = 2. Links start at position 2.
        # doc0: [0,1], doc1: [0], links: [2,3], query: [4,5], answer: [6]
        assert layout.positions == (0, 1, 0, 2, 3, 4, 5, 6)

    def test_logical_order(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        assert layout.logical_order == (
            "doc:0", "doc:0", "link:0",
            "doc:1", "link:1",
            "query", "query",
            "answer",
        )

    def test_doc0_row0_causal_within_doc(self, contract):
        """Doc0 token 0 attends only to itself (causal within doc0)."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        # Row 0 (doc:0 token 0): only itself
        assert layout.allow[0] == (True, False, False, False, False, False, False, False)

    def test_doc0_row1_causal_within_doc(self, contract):
        """Doc0 token 1 attends to doc0 tokens 0,1 (causal within doc0)."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        # Row 1 (doc:0 token 1): doc0 token 0 and itself
        assert layout.allow[1] == (True, True, False, False, False, False, False, False)

    def test_link0_row3(self, contract):
        """Link0 (physical slot 3) attends to doc0 (its owner) and itself.

        Physical order: doc0(0,1), doc1(2), link0(3), link1(4), query(5,6), answer(7).
        Link0 owner = document 0. It sees doc0 tokens (cols 0,1) and itself (col 3).
        """
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        # Row 3 = link0 (physical slot 3, logical_order[3] = 'doc:1' but physical = link)
        # link0 owner = document 0. Sees doc0 (cols 0,1) + earlier links (none) + itself (col 3)
        assert layout.allow[3] == (True, True, False, True, False, False, False, False)

    def test_doc1_row2(self, contract):
        """Doc1 token 0 (physical slot 2) attends only to itself (causal within doc1).

        Physical order: doc0(0,1), doc1(2), link0(3), link1(4), query(5,6), answer(7).
        Doc1 is isolated from doc0 and everything else.
        """
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        # Row 2 = doc1 token 0 (physical slot 2): only itself
        assert layout.allow[2] == (False, False, True, False, False, False, False, False)

    def test_link1_row4(self, contract):
        """Link1 (physical slot 4) attends to doc0, doc1, link0, and itself."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        # Row 4 is link:1 (physical slot 4). owner = document 1.
        # It sees: all docs with owner <= 1 (doc0 cols 0,1 and doc1 col 2),
        # and links up to and including itself (col 3, col 4)
        assert layout.allow[4] == (True, True, True, True, True, False, False, False)

    def test_query_row5(self, contract):
        """Query token 0 sees all docs, all links, itself."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        # Row 5 (query token 0): sees everything up to itself
        assert layout.allow[5] == (True, True, True, True, True, True, False, False)

    def test_query_row6(self, contract):
        """Query token 1 sees all docs, all links, both query tokens."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        assert layout.allow[6] == (True, True, True, True, True, True, True, False)

    def test_answer_row7(self, contract):
        """Answer token sees everything (full context)."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=1,
        )
        assert layout.allow[7] == (True, True, True, True, True, True, True, True)


class TestK5TwoDocumentHandCheck:
    """Hand-checkable mask for k=5 with two documents of length 2 and 1."""

    def test_physical_layout(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=5,
        )
        # Physical: doc0(10,11), doc1(20), links0(90,91,92,93,94), links1(90,91,92,93,94),
        #          query(30,31), answer(40)
        assert layout.physical_token_ids == (
            10, 11, 20,
            90, 91, 92, 93, 94,
            90, 91, 92, 93, 94,
            30, 31, 40,
        )
        assert len(layout.physical_token_ids) == 16

    def test_positions(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=5,
        )
        # max_doc_length = 2. Total links = 10. Query starts at position 2 + 10 = 12.
        # doc0: [0,1], doc1: [0], links0: [2,3,4,5,6], links1: [7,8,9,10,11],
        # query: [12,13], answer: [14]
        assert layout.positions == (0, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    def test_total_size(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=5,
        )
        assert len(layout.allow) == 16
        assert all(len(row) == 16 for row in layout.allow)

    def test_doc_rows_isolated(self, contract):
        """Doc rows never attend other docs or links."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=5,
        )
        # Row 0 (doc:0 token 0): only col 0
        assert layout.allow[0][0] == True
        assert sum(layout.allow[0]) == 1
        # Row 1 (doc:0 token 1): cols 0,1
        assert layout.allow[1][0] == True
        assert layout.allow[1][1] == True
        assert sum(layout.allow[1]) == 2
        # Row 2 (doc:1 token 0): only col 2
        assert layout.allow[2][2] == True
        assert sum(layout.allow[2]) == 1

    def test_link0_sees_only_owner_doc(self, contract):
        """Link0 (owner=doc0) sees doc0 but NOT doc1.

        Physical order: doc0(0,1), doc1(2,3,4), links0(5-9), links1(10-14), query(15-17), answer(18-19).
        Link0 is at physical slot 5, owner = document 0.
        """
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31, 32),
            (40, 41),
            k=5,
        )
        # Row 5 = link:0 slot 0, owner = 0
        # Sees doc0 (cols 0,1), links up to col 5 (only itself)
        assert layout.allow[5][0] == True   # doc0 token 0
        assert layout.allow[5][1] == True   # doc0 token 1
        assert layout.allow[5][2] == False  # doc1 token 0
        assert layout.allow[5][5] == True   # itself

    def test_link1_sees_both_docs(self, contract):
        """Link1 (owner=doc1) sees doc0 and doc1.

        Physical order: doc0(0,1), doc1(2,3,4), links0(5-9), links1(10-14), query(15-17), answer(18-19).
        Link1 is at physical slot 10, owner = document 1.
        """
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31, 32),
            (40, 41),
            k=5,
        )
        # Row 10 = link:1 slot 0, owner = 1
        # Sees doc0 (cols 0,1), doc1 (cols 2,3,4), all links up to col 10
        assert layout.allow[10][0] == True   # doc0 token 0
        assert layout.allow[10][1] == True   # doc0 token 1
        assert layout.allow[10][2] == True   # doc1 token 0
        assert layout.allow[10][5] == True   # link0 slot 0
        assert layout.allow[10][9] == True   # link0 slot 4
        assert layout.allow[10][10] == True  # itself

    def test_answer_sees_everything(self, contract):
        """Answer token sees all docs, all links, all query, itself."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=5,
        )
        # Row 15 = answer
        assert all(layout.allow[15])


class TestK0NoLinks:
    """k=0: no link tokens at all, standard ddi_qf layout."""

    def test_no_link_tokens(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=0,
        )
        # No link tokens in physical stream
        assert 90 not in layout.physical_token_ids
        assert layout.physical_token_ids == (10, 11, 20, 30, 31, 40)
        assert len(layout.physical_token_ids) == 6

    def test_positions_k0(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=0,
        )
        # max_doc_length=2, no links, query starts at 2
        assert layout.positions == (0, 1, 0, 2, 3, 4)

    def test_logical_order_k0(self, contract):
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=0,
        )
        assert layout.logical_order == ("doc:0", "doc:0", "doc:1", "query", "query", "answer")


# ---------------------------------------------------------------------------
# Causal-leakage tests
# ---------------------------------------------------------------------------


class TestCausalLeakage:
    """Verify no causal leakage: no row attends to a column after itself."""

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_lower_triangular(self, contract, k):
        """The allow matrix must be lower-triangular (causal)."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31, 32),
            (40, 41),
            k=k,
        )
        T = len(layout.allow)
        for row in range(T):
            for col in range(T):
                if col > row:
                    assert layout.allow[row][col] == False, (
                        f"Causal leak at k={k}: row={row}, col={col} "
                        f"should be False but is True"
                    )

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_diagonal_always_true(self, contract, k):
        """Every token attends to itself (diagonal is always True)."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31, 32),
            (40, 41),
            k=k,
        )
        T = len(layout.allow)
        for i in range(T):
            assert layout.allow[i][i] == True, (
                f"Diagonal False at k={k}: row={i}, col={i}"
            )

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_doc_isolation(self, contract, k):
        """Document rows never attend other documents."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31, 32),
            (40, 41),
            k=k,
        )
        doc_total = 5  # 2 + 3
        for doc0_row in range(2):  # doc0 is length 2
            # doc0 rows must not see doc1
            assert layout.allow[doc0_row][2] == False  # doc1 token 0
            assert layout.allow[doc0_row][3] == False  # doc1 token 1
            assert layout.allow[doc0_row][4] == False  # doc1 token 2
        for doc1_row in range(2, 5):  # doc1 is length 3
            # doc1 rows must not see doc0
            assert layout.allow[doc1_row][0] == False  # doc0 token 0
            assert layout.allow[doc1_row][1] == False  # doc0 token 1

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_doc_never_attends_links(self, contract, k):
        """Document rows never attend link tokens."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31, 32),
            (40, 41),
            k=k,
        )
        doc_total = 5
        link_total = 2 * k
        for doc_row in range(doc_total):
            for link_col in range(doc_total, doc_total + link_total):
                assert layout.allow[doc_row][link_col] == False, (
                    f"Doc row {doc_row} attends link col {link_col} at k={k}"
                )

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_doc_never_attends_query_or_answer(self, contract, k):
        """Document rows never attend query or answer tokens."""
        layout = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20, 21, 22))),
            (30, 31, 32),
            (40, 41),
            k=k,
        )
        doc_total = 5
        link_total = 2 * k
        non_doc_start = doc_total + link_total
        T = len(layout.allow)
        for doc_row in range(doc_total):
            for col in range(non_doc_start, T):
                assert layout.allow[doc_row][col] == False, (
                    f"Doc row {doc_row} attends non-doc col {col} at k={k}"
                )


# ---------------------------------------------------------------------------
# Document identity independence tests
# ---------------------------------------------------------------------------


class TestDocumentIdentity:
    """Document identities are context-independent (never include links)."""

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_identity_same_across_k(self, contract, k):
        """Same document has the same identity regardless of k."""
        doc = DocumentPayload((10, 11, 12))
        layout = build_request_layout(
            contract, (doc, DocumentPayload((20,))), (30,), k=k
        )
        expected = doc.identity(contract.document_cache_identity)
        assert layout.document_identities[0] == expected

    def test_identity_different_with_cache_id(self):
        """Same document, different cache identity → different identity."""
        doc = DocumentPayload((10, 11))
        id1 = doc.identity("cache-a")
        id2 = doc.identity("cache-b")
        assert id1 != id2

    def test_different_documents_different_identity(self, contract):
        """Different documents have different identities."""
        doc1 = DocumentPayload((10, 11))
        doc2 = DocumentPayload((20, 21))
        assert doc1.identity(contract.document_cache_identity) != doc2.identity(contract.document_cache_identity)

    def test_request_identity_changes_with_k(self, contract):
        """Request identity changes with k, but document identities don't."""
        doc = DocumentPayload((10, 11))
        zero = build_request_layout(contract, (doc, DocumentPayload((20,))), (30,), k=0)
        five = build_request_layout(contract, (doc, DocumentPayload((20,))), (30,), k=5)
        assert zero.request_identity != five.request_identity
        assert zero.document_identities == five.document_identities


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Layouts are deterministic for the same inputs."""

    @pytest.mark.parametrize("k", [0, 1, 5])
    def test_same_inputs_same_layout(self, contract, k):
        layout1 = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=k,
        )
        layout2 = build_request_layout(
            contract,
            (DocumentPayload((10, 11)), DocumentPayload((20,))),
            (30, 31),
            (40,),
            k=k,
        )
        assert layout1.physical_token_ids == layout2.physical_token_ids
        assert layout1.positions == layout2.positions
        assert layout1.allow == layout2.allow
        assert layout1.request_identity == layout2.request_identity

    def test_different_docs_different_layout(self, contract):
        l1 = build_request_layout(
            contract, (DocumentPayload((10, 11)), DocumentPayload((20,))), (30,), k=1
        )
        l2 = build_request_layout(
            contract, (DocumentPayload((10, 11)), DocumentPayload((99,))), (30,), k=1
        )
        assert l1.request_identity != l2.request_identity

    def test_different_query_different_layout(self, contract):
        l1 = build_request_layout(
            contract, (DocumentPayload((10, 11)), DocumentPayload((20,))), (30,), k=1
        )
        l2 = build_request_layout(
            contract, (DocumentPayload((10, 11)), DocumentPayload((20,))), (31,), k=1
        )
        assert l1.request_identity != l2.request_identity
