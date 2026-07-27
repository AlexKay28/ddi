"""HotpotQA distractor loader (issue #43).

HotpotQA ships 10 passages per query (2 gold + 8 distractors), so no retrieval step is required
before this loader is useful for DDI eval. Every record is multi-hop by construction — the native
`type` field (`bridge` / `comparison`) becomes the hop label for free (Phase 5.1's hop-split
renderer consumes it directly).

HF splits: `train` (~90k) and `validation` (~7.4k). We hold out half of `validation` as our `test`
so both `val` and `test` are unseen during any training. HF has no public labelled `test` split.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from ddi.data.loaders import Split, register_loader
from ddi.data.manifest import validate_or_write
from ddi.data.splits import hash_split

# The dataset ID we resolve on the HF hub. If it changes upstream, change it here so the choice is
# visible without spelunking through the loader body.
_HF_DATASET_ID = "hotpot_qa"
_HF_CONFIG = "distractor"


def _normalize(row: dict) -> dict:
    context = row.get("context", {}) or {}
    titles = list(context.get("title", []) or [])
    sentences = list(context.get("sentences", []) or [])
    # One doc per title: join that title's sentences into a single paragraph. Titles and sentence
    # lists are index-aligned per the HF schema.
    info_contexts = [" ".join(sentences[i]) for i in range(len(titles))]

    supporting = row.get("supporting_facts", {}) or {}
    supporting_titles = list(supporting.get("title", []) or [])

    question = row.get("question", "")
    answer = row.get("answer", "")

    return {
        "id": str(row["id"]),
        "info_contexts": info_contexts,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
        # HotpotQA is 100% multi-hop by construction: both `bridge` and `comparison` require
        # combining evidence from two documents.
        "hop_type": "multi",
        "gold_position": None,
        "meta": {
            "type": row.get("type"),
            "level": row.get("level"),
            "supporting_titles": supporting_titles,
        },
    }


def _load_hf_split(hf_split: str) -> list[dict]:
    from datasets import load_dataset  # local import: `datasets` is optional at test time

    ds = load_dataset(_HF_DATASET_ID, _HF_CONFIG, split=hf_split)
    return [_normalize(r) for r in ds]


def _all_splits(seed: int) -> dict[Split, list[dict]]:
    # HotpotQA distractor ships train + validation; hold out half of validation as test so both
    # are unseen during any training.
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


@register_loader("hotpot_qa")
def load_hotpot_qa(split: Split, seed: int) -> Iterator[dict]:
    by_split = _all_splits(seed)

    hashes: dict[Split, tuple[int, str]] = {
        name: (len(recs), hash_split([r["id"] for r in recs]))
        for name, recs in by_split.items()
    }
    validate_or_write("hotpot_qa", hashes)

    for rec in sorted(by_split[split], key=lambda r: r["id"]):
        yield rec
