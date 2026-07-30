"""Natural Questions loader (issue #9, Phase 1.1).

NQ Open is a question–answer dataset with no retrieved passages, so `info_contexts` is empty here.
A retrieval step (Phase 2.1) must run before this loader is useful for actual DDI eval; it exists
now so Phase 1.4's `full` baseline can run against a real dataset.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator

from ddi.data.loaders import Split, register_loader
from ddi.data.manifest import validate_or_write
from ddi.data.records_cache import has_cached, read_split, write_split
from ddi.data.splits import hash_split

# All splits this loader emits. `test` is a locally-derived half of validation.
_SPLITS: tuple[Split, ...] = ("train", "val", "test")

# The dataset ID we resolve on the HF hub. If it changes upstream, change it here so the choice is
# visible without spelunking through the loader body.
_HF_DATASET_ID = "google-research-datasets/nq_open"


def _stable_id(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def _normalize(row: dict) -> dict:
    question = row.get("question", "")
    answers = list(row.get("answer", []) or [])
    first_answer = answers[0] if answers else ""

    raw_id = row.get("id") or row.get("example_id")
    record_id = str(raw_id) if raw_id is not None else _stable_id(question)

    meta = {k: v for k, v in row.items() if k not in {"question", "answer", "id", "example_id"}}

    return {
        "id": record_id,
        "info_contexts": [],
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": first_answer},
        ],
        "possible_answers": answers,
        "hop_type": "single",
        "gold_position": None,
        "meta": meta,
    }


def _load_hf_split(hf_split: str) -> list[dict]:
    from datasets import load_dataset  # local import: `datasets` is optional at test time

    ds = load_dataset(_HF_DATASET_ID, split=hf_split)
    return [_normalize(r) for r in ds]


def _all_splits(seed: int) -> dict[Split, list[dict]]:
    # NQ Open ships train + validation; we hold out half of validation as test so both are
    # unseen during training.
    train = _load_hf_split("train")
    val_all = _load_hf_split("validation")

    train.sort(key=lambda r: r["id"])
    val_all.sort(key=lambda r: r["id"])

    rng = random.Random(seed)
    order = list(range(len(val_all)))
    rng.shuffle(order)
    half = len(order) // 2
    val_idx = set(order[:half])
    val = [val_all[i] for i in sorted(val_idx)]
    test = [val_all[i] for i in range(len(val_all)) if i not in val_idx]

    return {"train": train, "val": val, "test": test}


@register_loader("nq")
def load_nq(split: Split, seed: int) -> Iterator[dict]:
    if has_cached("nq", seed, _SPLITS):
        yield from read_split("nq", seed, split)
        return

    by_split = _all_splits(seed)

    hashes: dict[Split, tuple[int, str]] = {
        name: (len(recs), hash_split([r["id"] for r in recs]))
        for name, recs in by_split.items()
    }
    # Split membership is a function of `seed` (val/test are a random half-split of NQ's
    # validation set); key the manifest per seed so seeds 1/2 don't collide with seed 0's
    # frozen split (issue #40).
    validate_or_write(f"nq__seed{seed}", hashes)

    for s in _SPLITS:
        write_split("nq", seed, s, sorted(by_split[s], key=lambda r: r["id"]))

    for rec in sorted(by_split[split], key=lambda r: r["id"]):
        yield rec
