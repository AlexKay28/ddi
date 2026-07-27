"""MuSiQue-ans loader (issue #44, Phase 1.1).

MuSiQue is a 2-to-4-hop QA dataset with 20 paragraphs per query (mostly distractors) and
author-verified compositionality — the strongest test in the benchmark survey (#42) for DDI's
predicted C5 cross-document composition failure. We keep `n_hops` in `meta` so the report layer
can slice quality by depth (2 vs 3 vs 4).

Only the answerable subset is loaded; the `-full` variant adds unanswerable questions, which is a
separate research direction and would confound the composition-failure signal.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from ddi.data.loaders import Split, register_loader
from ddi.data.manifest import validate_or_write
from ddi.data.splits import hash_split

# WHY dgslibisey/MuSiQue: highest apparent monthly downloads among the community mirrors
# (~12.9k/mo vs ~9.7k/mo for bdsaglam/musique at the time of writing), and its row counts
# (train=19,938, validation=2,417) match the canonical MuSiQue-Ans release from
# stonybrooknlp/musique on GitHub, i.e. it is already the answerable-only subset.
# Alternatives considered:
#   - bdsaglam/musique: ships both -ans and -full under `default`/`answerable` configs; would
#     force a config selection we don't otherwise need.
#   - agentlans/musique: could not verify from the sandbox (auth-gated at fetch time).
# If dgslibisey ever goes unmaintained, re-pin here — the record shape is stable across mirrors.
_HF_DATASET_ID = "dgslibisey/MuSiQue"


def _normalize(row: dict) -> dict:
    question = row.get("question", "")
    answer = row.get("answer", "")
    aliases = list(row.get("answer_aliases") or [])
    paragraphs = list(row.get("paragraphs") or [])
    decomposition = list(row.get("question_decomposition") or [])

    info_contexts = [f"{p['title']}: {p['paragraph_text']}" for p in paragraphs]
    supporting_idxs = [p["idx"] for p in paragraphs if p.get("is_supporting")]

    return {
        "id": str(row["id"]),
        "info_contexts": info_contexts,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer, *aliases],
        "hop_type": "multi",
        "gold_position": None,
        "meta": {
            "n_hops": len(decomposition),
            "supporting_idxs": supporting_idxs,
        },
    }


def _load_hf_split(hf_split: str) -> list[dict]:
    from datasets import load_dataset  # local import: `datasets` is optional at test time

    ds = load_dataset(_HF_DATASET_ID, split=hf_split)

    kept: list[dict] = []
    dropped = 0
    for row in ds:
        # `dgslibisey/MuSiQue` is already the -ans subset, but guard anyway: if a future mirror
        # or re-pin sneaks in -full rows, `answerable=False` records get logged and skipped.
        if row.get("answerable") is False:
            dropped += 1
            continue
        kept.append(_normalize(row))

    if dropped:
        print(f"[musique] dropped {dropped} unanswerable rows from split={hf_split!r}")

    return kept


def _all_splits(seed: int) -> dict[Split, list[dict]]:
    # `dgslibisey/MuSiQue` publishes train + validation only (no public test with answers).
    # Map validation -> our val; hold train back for future retrieval-index work; leave test empty
    # so the manifest has a stable three-key shape.
    train = _load_hf_split("train")
    val_all = _load_hf_split("validation")

    train.sort(key=lambda r: r["id"])
    val_all.sort(key=lambda r: r["id"])

    # Deterministic 50/50 split of validation -> val + test so both are unseen at train time.
    # Same pattern as ddi.data.nq — MuSiQue's public test lacks gold answers, so we roll our own.
    rng = random.Random(seed)
    order = list(range(len(val_all)))
    rng.shuffle(order)
    half = len(order) // 2
    val_idx = set(order[:half])
    val = [val_all[i] for i in sorted(val_idx)]
    test = [val_all[i] for i in range(len(val_all)) if i not in val_idx]

    return {"train": train, "val": val, "test": test}


@register_loader("musique")
def load_musique(split: Split, seed: int) -> Iterator[dict]:
    by_split = _all_splits(seed)

    hashes: dict[Split, tuple[int, str]] = {
        name: (len(recs), hash_split([r["id"] for r in recs]))
        for name, recs in by_split.items()
    }
    validate_or_write("musique", hashes)

    for rec in sorted(by_split[split], key=lambda r: r["id"]):
        yield rec
