"""CPU-only tests for the V62 request-time link-token contract."""

import pytest

from ddi.link_token_contract import (
    DocumentPayload,
    LinkTokenContract,
    SplitManifest,
    build_request_layout,
)


@pytest.fixture
def contract() -> LinkTokenContract:
    return LinkTokenContract("canonical-cache-v60-test", (90, 91, 92, 93, 94))


def test_k1_two_document_layout_is_deterministic_and_hand_checkable(contract):
    layout = build_request_layout(
        contract,
        (DocumentPayload((10, 11)), DocumentPayload((20,))),
        (30, 31),
        (40,),
        k=1,
    )

    # Physical slots are cached docs, then request-only links, query, and answer.
    assert layout.physical_token_ids == (10, 11, 20, 90, 90, 30, 31, 40)
    assert layout.positions == (0, 1, 0, 2, 3, 4, 5, 6)
    assert layout.logical_order == (
        "doc:0", "doc:0", "link:0", "doc:1", "link:1", "query", "query", "answer"
    )
    assert layout.allow[0] == (True, False, False, False, False, False, False, False)
    assert layout.allow[2] == (False, False, True, False, False, False, False, False)
    assert layout.allow[3] == (True, True, False, True, False, False, False, False)
    assert layout.allow[4] == (True, True, True, True, True, False, False, False)
    assert layout.allow[5] == (True, True, True, True, True, True, False, False)
    assert layout.allow[7] == (True, True, True, True, True, True, True, True)
    assert layout == build_request_layout(
        contract,
        (DocumentPayload((10, 11)), DocumentPayload((20,))),
        (30, 31),
        (40,),
        k=1,
    )


@pytest.mark.parametrize(
    "k,expected_links", [(0, ()), (1, (90, 90)), (5, (90, 91, 92, 93, 94) * 2)]
)
def test_k_variants_preserve_document_identity_and_assign_link_ids(contract, k, expected_links):
    document = DocumentPayload((10, 11))
    layout = build_request_layout(contract, (document, DocumentPayload((20,))), (30,), k=k)

    assert layout.physical_token_ids[3 : 3 + len(expected_links)] == expected_links
    assert layout.document_identities[0] == document.identity(contract.document_cache_identity)
    assert "link_token_ids" not in document.identity(contract.document_cache_identity)


def test_request_contract_changes_with_k_without_leaking_links_into_document_identity(contract):
    document = DocumentPayload((10, 11))
    zero = build_request_layout(contract, (document, DocumentPayload((20,))), (30,), k=0)
    five = build_request_layout(contract, (document, DocumentPayload((20,))), (30,), k=5)

    assert zero.request_identity != five.request_identity
    assert zero.document_identities == five.document_identities
    assert document.identity("another-cache-identity") != document.identity(
        contract.document_cache_identity
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda contract: build_request_layout(contract, (), (1,), k=0),
        lambda contract: build_request_layout(contract, (DocumentPayload((1,)),), (), k=0),
        lambda contract: build_request_layout(contract, (DocumentPayload((1,)),), (2,), k=2),
        lambda contract: LinkTokenContract("cache", (1, 1, 2, 3, 4)),
        lambda contract: DocumentPayload((1, -1)),
    ],
)
def test_invalid_inputs_are_rejected(contract, factory):
    with pytest.raises(ValueError):
        factory(contract)


def test_confirmation_ids_cannot_be_reused_as_development():
    with pytest.raises(ValueError, match="confirmation data cannot be reused as development"):
        SplitManifest(
            train_ids=frozenset({"train-1"}),
            development_ids=frozenset({"held-out-1"}),
            confirmation_ids=frozenset({"held-out-1"}),
        )


def test_split_labels_are_distinct_and_retrievable():
    manifest = SplitManifest(
        train_ids=frozenset({"train-1"}),
        development_ids=frozenset({"dev-1"}),
        confirmation_ids=frozenset({"confirm-1"}),
    )
    assert manifest.ids_for("train") == frozenset({"train-1"})
    assert manifest.ids_for("development") == frozenset({"dev-1"})
    assert manifest.ids_for("confirmation") == frozenset({"confirm-1"})
