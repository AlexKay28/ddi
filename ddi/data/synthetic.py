"""Synthetic QA dataset (issue #9, Phase 1.1).

A real dataset that runs without network: tests and smoke runs stay hermetic. Deterministic given
`(split, seed)` — same inputs, byte-identical stream.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from ddi.data.loaders import Split, register_loader
from ddi.data.manifest import validate_or_write
from ddi.data.splits import hash_split, split_ids

_N_RECORDS = 100
_VAL_FRAC = 0.20
_TEST_FRAC = 0.20
_DOCS_PER_RECORD = 3
_DOC_WORDS = 12
_VOCAB = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
    "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu"
).split()


def _record_rng(seed: int, record_idx: int) -> random.Random:
    # Two independent axes → mix them into one 64-bit stream.
    return random.Random((seed & 0xFFFFFFFF) * 1_000_003 + record_idx)


def _make_record(record_idx: int, seed: int) -> dict:
    rng = _record_rng(seed, record_idx)
    answer = f"answer_{record_idx:04d}_{rng.randint(0, 9999):04d}"
    question_nonce = f"nonce_{record_idx:04d}"
    gold_doc_idx = rng.randrange(_DOCS_PER_RECORD)

    docs: list[str] = []
    for d in range(_DOCS_PER_RECORD):
        words = [rng.choice(_VOCAB) for _ in range(_DOC_WORDS)]
        if d == gold_doc_idx:
            insert_pos = rng.randrange(len(words) + 1)
            words.insert(insert_pos, answer)
        docs.append(" ".join(words))

    record_id = f"synthetic_qa-{seed}-{record_idx:04d}"
    return {
        "id": record_id,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": f"What is {question_nonce}?"},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
        "hop_type": "single",
        "gold_position": None,
        "meta": {"gold_doc_idx": gold_doc_idx, "question_nonce": question_nonce},
    }


def _all_records(seed: int) -> list[dict]:
    return [_make_record(i, seed) for i in range(_N_RECORDS)]


@register_loader("synthetic_qa")
def load_synthetic_qa(split: Split, seed: int) -> Iterator[dict]:
    records = _all_records(seed)
    all_ids = [r["id"] for r in records]
    by_split = split_ids(all_ids, val_frac=_VAL_FRAC, test_frac=_TEST_FRAC, seed=seed)

    hashes = {name: (len(ids), hash_split(ids)) for name, ids in by_split.items()}
    # Record IDs and split partition are both a function of `seed`; key the manifest per seed
    # so seeds 1/2 don't collide with seed 0's frozen split (issue #40).
    validate_or_write(f"synthetic_qa__seed{seed}", hashes)

    wanted = set(by_split[split])
    by_id = {r["id"]: r for r in records}
    # Deterministic yield order: sorted by id.
    for rid in sorted(wanted):
        yield by_id[rid]
