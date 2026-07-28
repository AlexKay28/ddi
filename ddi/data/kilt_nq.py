"""KILT-NQ loader (issue #45, Phase 1.1).

Loads the KILT-formatted Natural Questions subset via
``datasets.load_dataset("kilt_tasks", "nq")``. Unlike :mod:`ddi.data.nq` (bare NQ Open,
no passages), this loader attaches the top-N Wikipedia passages KILT ships as *provenance*
so we get a retrieved-passage variant of NQ that is directly comparable to the numbers
PCW (Ratner et al., ACL 2023) reports — the paper's C1 comparability column depends on
this.

Passages are drawn from KILT's precomputed provenance list on each row. This yields a
richer-than-Tier-A stream (up to N passages per query, not just the single gold snippet),
while avoiding the operational cost of running a fresh DPR retrieval in this loader. If the
provenance list turns out too thin at higher N in practice, a private
``_retrieve_top_n(question, n)`` helper can be added here later; a shared retrieval module
will only be worth extracting once a second loader needs the same code path.

Rows with fewer than N provenance entries carrying extractable text yield fewer than N
passages rather than padding with an empty placeholder — `ddi.masks.build_mask` requires
every document to be non-empty, so a zero-length "document" would crash the run.
``meta["n_passages"]`` always reflects the actual count delivered, which may be less than
the configured N.

Configurable N passages
-----------------------
The default is 5 (an upper bound; thin rows may yield fewer). To override without editing
code::

    DDI_KILT_N_PASSAGES=10 python -m ddi.cli ...

or from Python::

    from ddi.data import kilt_nq
    kilt_nq.set_n_passages(10)

Manifest note
-------------
The split manifest hashes record IDs only (see :func:`ddi.data.splits.hash_split`), so
changing ``n_passages`` does *not* invalidate a previously committed manifest — the set of
records in each split is unchanged, only the payload per record grows or shrinks.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterator

from ddi.data.loaders import Split, register_loader
from ddi.data.manifest import validate_or_write
from ddi.data.splits import hash_split

_HF_DATASET_ID = "kilt_tasks"
_HF_CONFIG = "nq"

_DEFAULT_N_PASSAGES = 5
_ENV_N_PASSAGES = "DDI_KILT_N_PASSAGES"

# Module-level override so tests / orchestrators can bump N without an env var; env var
# still wins when set so a run script can override without touching Python state.
_n_passages_override: int | None = None


def set_n_passages(n: int) -> None:
    """Override the module default for the number of passages per record.

    An env var (``DDI_KILT_N_PASSAGES``) takes precedence over this override so shell
    invocations remain the last word.
    """
    if n < 1:
        raise ValueError(f"n_passages must be >= 1, got {n}")
    global _n_passages_override
    _n_passages_override = int(n)


def _current_n_passages() -> int:
    env = os.environ.get(_ENV_N_PASSAGES)
    if env is not None:
        try:
            n = int(env)
        except ValueError as exc:
            raise ValueError(f"{_ENV_N_PASSAGES}={env!r} is not an int") from exc
        if n < 1:
            raise ValueError(f"{_ENV_N_PASSAGES} must be >= 1, got {n}")
        return n
    if _n_passages_override is not None:
        return _n_passages_override
    return _DEFAULT_N_PASSAGES


def _stable_id(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def _passage_text(prov: dict) -> str:
    """Best-effort text extraction from a KILT provenance dict.

    KILT rows carry either an inline ``text`` snippet or a wikipedia reference — we prefer
    the inline text; fall back to a ``"<title>"`` placeholder so downstream code can still
    count passages while a follow-up wires a real wiki lookup.
    """
    text = prov.get("text")
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if t)
    if text:
        return str(text).strip()
    title = prov.get("title") or prov.get("wikipedia_title")
    return f"[{title}]" if title else ""


def _provenance_title(prov: dict) -> str:
    return str(prov.get("title") or prov.get("wikipedia_title") or "")


def _collect_provenance(row: dict) -> list[dict]:
    """Flatten every provenance dict from every output entry, in KILT-yield order.

    KILT stores multiple ``output`` entries (accepted answers) and each carries its own
    provenance list. The concatenation preserves the retrieval-ranking KILT ships for the
    first entry, then appends any extras from alternate answers — with dedup by title so a
    passage doesn't appear twice.
    """
    seen_titles: set[str] = set()
    out: list[dict] = []
    for entry in row.get("output", []) or []:
        for prov in entry.get("provenance", []) or []:
            title = _provenance_title(prov)
            key = title.lower()
            if key and key in seen_titles:
                continue
            if key:
                seen_titles.add(key)
            out.append(prov)
    return out


def _gold_title(row: dict) -> str:
    """First provenance title on the first output entry — KILT's canonical gold pointer."""
    for entry in row.get("output", []) or []:
        for prov in entry.get("provenance", []) or []:
            title = _provenance_title(prov)
            if title:
                return title
    return ""


def _normalize(row: dict, n_passages: int) -> dict:
    question = row.get("input", "") or ""
    outputs = row.get("output", []) or []
    answers = [str(o["answer"]) for o in outputs if isinstance(o, dict) and "answer" in o]
    first_answer = answers[0] if answers else ""

    raw_id = row.get("id")
    record_id = str(raw_id) if raw_id is not None else _stable_id(question)

    # Drop provenance entries with no extractable text before capping at n_passages, so a
    # thin KILT row yields fewer real passages rather than padding with an empty "document"
    # that downstream masking (build_mask) rejects outright (each doc must be non-empty).
    all_prov = [p for p in _collect_provenance(row) if _passage_text(p)]
    top = all_prov[:n_passages]
    passages = [_passage_text(p) for p in top]

    gold_title = _gold_title(row)

    # gold_position is intentionally None for kilt_nq: KILT does not carry a stable
    # gold-vs-distractor annotation. `_gold_title` returns whichever passage happens
    # to be listed first, which is not a rotation-stable identity — it moves when the
    # provenance list is permuted. Downstream analysis that needs a real gold pointer
    # should use ddi_stress (which has controlled gold placement via `gold_positions`)
    # or a dataset that ships explicit gold flags (hotpot_qa's supporting_facts).
    # Closes #55.
    return {
        "id": record_id,
        "info_contexts": passages,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": first_answer},
        ],
        "possible_answers": answers,
        "hop_type": "single",
        "gold_position": None,
        "meta": {
            "retriever": "kilt_provenance",
            "n_passages": len(passages),
            "gold_title": gold_title,
        },
    }


def _load_hf_split(hf_split: str, n_passages: int) -> list[dict]:
    from datasets import load_dataset  # local import: `datasets` is optional at test time

    ds = load_dataset(_HF_DATASET_ID, _HF_CONFIG, split=hf_split)
    return [_normalize(r, n_passages) for r in ds]


def _all_splits(seed: int, n_passages: int) -> dict[Split, list[dict]]:
    # KILT ships train + validation for NQ; hold out half of validation as test so both
    # val and test are unseen during training — mirrors ddi/data/nq.py.
    train = _load_hf_split("train", n_passages)
    val_all = _load_hf_split("validation", n_passages)

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


@register_loader("kilt_nq")
def load_kilt_nq(split: Split, seed: int) -> Iterator[dict]:
    n_passages = _current_n_passages()
    by_split = _all_splits(seed, n_passages)

    hashes: dict[Split, tuple[int, str]] = {
        name: (len(recs), hash_split([r["id"] for r in recs]))
        for name, recs in by_split.items()
    }
    validate_or_write(f"kilt_nq__seed{seed}", hashes)

    for rec in sorted(by_split[split], key=lambda r: r["id"]):
        yield rec
