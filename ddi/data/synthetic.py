"""Synthetic QA dataset (issue #9, Phase 1.1).

A real dataset that runs without network: tests and smoke runs stay hermetic. Deterministic given
`(split, seed)` — same inputs, byte-identical stream.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

from ddi.data.loaders import Split, register_loader
from ddi.data.manifest import validate_or_write
from ddi.data.splits import hash_split, split_ids

_VAL_FRAC = 0.20
_TEST_FRAC = 0.20

# Env-overridable so the document-count axis (VIBEQUEUE-2 rows 14-15) needs no code
# change, and so the val split can be made large enough to measure anything: at the
# hardcoded 100 records, `val_frac=0.20` yields **20 val records**, which is what
# `max_records: 500` was actually evaluating on this dataset. A metric computed over
# 20 records moves in steps of 0.05 and cannot resolve a parity claim
# (VIBEQUEUE-2 review, issue R7).
_ENV_N_RECORDS = "DDI_SYNTHETIC_N_RECORDS"
_ENV_DOCS_PER_RECORD = "DDI_SYNTHETIC_DOCS_PER_RECORD"
_DEFAULT_N_RECORDS = 100
_DEFAULT_DOCS_PER_RECORD = 3
_DOC_WORDS = 12


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def n_records() -> int:
    """Total generated records (train+val+test), default 100."""
    return _env_int(_ENV_N_RECORDS, _DEFAULT_N_RECORDS)


def docs_per_record() -> int:
    """Documents per record, default 3. Raising this is the controlled document-load axis."""
    return _env_int(_ENV_DOCS_PER_RECORD, _DEFAULT_DOCS_PER_RECORD)
_VOCAB = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
    "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu"
).split()


def _record_rng(seed: int, record_idx: int) -> random.Random:
    # Two independent axes → mix them into one 64-bit stream.
    return random.Random((seed & 0xFFFFFFFF) * 1_000_003 + record_idx)


def _make_record(record_idx: int, seed: int) -> dict:
    rng = _record_rng(seed, record_idx)
    answer_prefix = f"answer_{record_idx:04d}_"
    answer = f"{answer_prefix}{rng.randint(0, 9999):04d}"
    n_docs = docs_per_record()
    gold_doc_idx = rng.randrange(n_docs)

    docs: list[str] = []
    for d in range(n_docs):
        words = [rng.choice(_VOCAB) for _ in range(_DOC_WORDS)]
        if d == gold_doc_idx:
            insert_pos = rng.randrange(len(words) + 1)
            words.insert(insert_pos, answer)
        docs.append(" ".join(words))

    # The question must (a) not collide with a real-world concept an instruct model already
    # has a strong prior about — "nonce" made models answer from cryptography knowledge and
    # never look at the passages at all — and (b) explicitly tell the model the passages above
    # are the only source of truth and give it a literal, matchable search key, since the docs
    # are unrelated random-word noise with no natural-language link back to the question.
    question = (
        "The passages above are random words, except exactly one of them also contains a "
        f"single token starting with {answer_prefix!r} followed by 4 more digits. "
        "Reply with only that exact token and nothing else."
    )

    record_id = f"synthetic_qa-{seed}-{record_idx:04d}"
    return {
        "id": record_id,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
        "hop_type": "single",
        "gold_position": None,
        # `self_instructed` tells ddi.prompt not to wrap the question again — the
        # question above already names the passages and demands a bare token.
        "meta": {
            "gold_doc_idx": gold_doc_idx,
            "answer_prefix": answer_prefix,
            "self_instructed": True,
        },
    }


def _all_records(seed: int) -> list[dict]:
    return [_make_record(i, seed) for i in range(n_records())]


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
