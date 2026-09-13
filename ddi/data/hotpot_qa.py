"""HotpotQA distractor loader (issue #43).

HotpotQA ships 10 passages per query (2 gold + 8 distractors), so no retrieval step is required
before this loader is useful for DDI eval. Every record is multi-hop by construction — the native
`type` field (`bridge` / `comparison`) becomes the hop label for free (Phase 5.1's hop-split
renderer consumes it directly).

HF splits: `train` (~90k) and `validation` (~7.4k). We hold out half of `validation` as our `test`
so both `val` and `test` are unseen during any training. HF has no public labelled `test` split.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

from ddi.data.loaders import Split, register_loader
from ddi.data.manifest import validate_or_write
from ddi.data.records_cache import has_cached, read_split, write_split
from ddi.data.splits import hash_split

# All splits this loader emits; needed by both the cache-presence check and the
# record-persistence loop below.
_SPLITS: tuple[Split, ...] = ("train", "val", "test")

# The dataset ID we resolve on the HF hub. If it changes upstream, change it here so the choice is
# visible without spelunking through the loader body.
# Namespaced, for the same reason as ddi/data/kilt_nq.py: datasets>=4 rejects a bare
# canonical name ("Repository id must be 'namespace/name'"), so a cold cache cannot fetch
# at all. Only the record cache hides it. The other loaders here were already namespaced.
_HF_DATASET_ID = "hotpotqa/hotpot_qa"
_HF_CONFIG = "distractor"

# Document-load axis (VIBEQUEUE-2 rows 6 and 20). Unset = all 10 passages, the native
# distractor setting and the previous behaviour.
#
# Subsetting keeps every *supporting* paragraph and drops distractors only. Truncating
# blindly would drop gold evidence on some records, and the resulting quality drop would
# look exactly like a document-count effect while actually being an answerability effect
# — which is the confound this axis exists to avoid. Selected paragraphs are emitted in
# their original relative order, so gold position stays as varied as it was and the axis
# does not smuggle in a position effect either.
_ENV_N_PARAGRAPHS = "DDI_HOTPOT_N_PARAGRAPHS"


def n_paragraphs() -> int | None:
    """Paragraphs per record, or None for all of them (the default)."""
    raw = os.environ.get(_ENV_N_PARAGRAPHS)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_N_PARAGRAPHS} must be an integer, got {raw!r}") from exc
    if value < 2:
        # Below 2 the DDI mask family collapses to `full` (ddi/masks.py) and the record
        # measures nothing at all.
        raise ValueError(f"{_ENV_N_PARAGRAPHS} must be >= 2, got {value}")
    return value


def _select_paragraphs(
    info_contexts: list[str], titles: list[str], supporting_titles: list[str]
) -> list[int]:
    """Indices to keep: all supporting paragraphs, then distractors, in original order."""
    limit = n_paragraphs()
    if limit is None or len(info_contexts) <= limit:
        return list(range(len(info_contexts)))

    supporting = set(supporting_titles)
    gold = [i for i, t in enumerate(titles) if t in supporting]
    distractors = [i for i in range(len(info_contexts)) if i not in set(gold)]
    keep = set(gold[:limit]) | set(distractors[: max(0, limit - len(gold))])
    return sorted(keep)


def _normalize(row: dict) -> dict:
    context = row.get("context", {}) or {}
    titles = list(context.get("title", []) or [])
    sentences = list(context.get("sentences", []) or [])
    # One doc per title: join that title's sentences into a single paragraph. Titles and sentence
    # lists are index-aligned per the HF schema.
    info_contexts = [" ".join(sentences[i]) for i in range(len(titles))]

    supporting = row.get("supporting_facts", {}) or {}
    supporting_titles = list(supporting.get("title", []) or [])

    keep = _select_paragraphs(info_contexts, titles, supporting_titles)
    info_contexts = [info_contexts[i] for i in keep]
    kept_titles = [titles[i] for i in keep]

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
            # Actual count delivered, which may be below the configured limit on a thin
            # row — the same contract kilt_nq's `n_passages` follows.
            "n_paragraphs": len(info_contexts),
            "supporting_idxs": [
                i for i, t in enumerate(kept_titles) if t in set(supporting_titles)
            ],
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


def _cache_name() -> str:
    """Cache key that includes the paragraph limit.

    Record IDs are stable across the limit (the manifest hashes IDs only), but the
    *payload* — `info_contexts` length and `meta["n_paragraphs"]` — changes with it.
    Segregating the cache prevents a `DDI_HOTPOT_N_PARAGRAPHS=2` run from silently
    reading the 10-paragraph cache and reporting a document-count result that never
    varied the document count. Mirrors `ddi/data/kilt_nq.py::_cache_name`.
    """
    limit = n_paragraphs()
    return "hotpot_qa" if limit is None else f"hotpot_qa_n{limit}"


@register_loader("hotpot_qa")
def load_hotpot_qa(split: Split, seed: int) -> Iterator[dict]:
    # Fast path: every split's records file exists on disk from a previous fetch.
    # Skip the HF-hub round-trip entirely so offline containers can serve. Note we
    # still gate on *all* splits (not just the requested one) so the manifest hash
    # — computed from the full partition — is stable regardless of load order.
    cache_name = _cache_name()
    if has_cached(cache_name, seed, _SPLITS):
        yield from read_split(cache_name, seed, split)
        return

    by_split = _all_splits(seed)

    hashes: dict[Split, tuple[int, str]] = {
        name: (len(recs), hash_split([r["id"] for r in recs]))
        for name, recs in by_split.items()
    }
    validate_or_write(f"hotpot_qa__seed{seed}", hashes)

    # Persist every split now (not just the requested one) so a subsequent call
    # for a different split still hits the fast path.
    for s in _SPLITS:
        write_split(cache_name, seed, s, sorted(by_split[s], key=lambda r: r["id"]))

    for rec in sorted(by_split[split], key=lambda r: r["id"]):
        yield rec
