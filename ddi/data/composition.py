"""Cross-document composition probe (issue #28, Phase 5.3).

A synthetic multi-hop task where the answer PROVABLY requires two specific documents:

    doc_A:  "<X> lives in <CITY>."
    doc_B:  "<CITY> is in <COUNTRY>."
    question: "What country does <X> live in?"
    answer:   <COUNTRY>

The remaining N-2 documents are distractors whose text mentions none of X, CITY, or COUNTRY —
so the answer is only reachable by composing (doc_A, doc_B). Full causal attention lets doc_B
attend to doc_A, so the composition can happen inside the document region; `ddi_qf` blocks that
edge, forcing the composition into the query/answer region. Varying the number of distractors
between doc_A and doc_B (`distance`) and the pair's location in the doc list (`pair_position`)
lets Phase 5.3 characterise the boundary where ddi_qf stops recovering — exactly what issue #28's
acceptance criterion asks for.

Deterministic given `(split, seed)`. All fact pools are module-level constants so the loader
works offline (ground rule 1: no hidden network reads in the data path).
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

_DEFAULT_N_DOCS = 5
_MAX_DISTANCE = 4  # distance in {0, 1, 2, 3, 4}; upper bound is _DEFAULT_N_DOCS - 2 == 3, +1 slack for future N=6

# WHY a fixed pool: the answer strings must be recognisable words the model has seen during
# pretraining, otherwise EM would test tokenizer luck rather than composition. Kept small so the
# fact space is tractable for tests but non-trivial for a real sweep.
_CITY_COUNTRY_PAIRS: list[tuple[str, str]] = [
    ("Paris", "France"),
    ("Lyon", "France"),
    ("Berlin", "Germany"),
    ("Munich", "Germany"),
    ("Madrid", "Spain"),
    ("Barcelona", "Spain"),
    ("Rome", "Italy"),
    ("Milan", "Italy"),
    ("Lisbon", "Portugal"),
    ("Porto", "Portugal"),
    ("Amsterdam", "Netherlands"),
    ("Rotterdam", "Netherlands"),
    ("Brussels", "Belgium"),
    ("Vienna", "Austria"),
    ("Zurich", "Switzerland"),
    ("Geneva", "Switzerland"),
    ("Stockholm", "Sweden"),
    ("Oslo", "Norway"),
    ("Copenhagen", "Denmark"),
    ("Helsinki", "Finland"),
    ("Warsaw", "Poland"),
    ("Krakow", "Poland"),
    ("Prague", "Czechia"),
    ("Budapest", "Hungary"),
    ("Athens", "Greece"),
    ("Dublin", "Ireland"),
    ("London", "England"),
    ("Manchester", "England"),
    ("Edinburgh", "Scotland"),
    ("Cardiff", "Wales"),
    ("Toronto", "Canada"),
    ("Montreal", "Canada"),
    ("Vancouver", "Canada"),
    ("Boston", "America"),
    ("Chicago", "America"),
    ("Seattle", "America"),
    ("Portland", "America"),
    ("Denver", "America"),
    ("Tokyo", "Japan"),
    ("Osaka", "Japan"),
    ("Kyoto", "Japan"),
    ("Seoul", "Korea"),
    ("Beijing", "China"),
    ("Shanghai", "China"),
    ("Mumbai", "India"),
    ("Delhi", "India"),
    ("Sydney", "Australia"),
    ("Melbourne", "Australia"),
    ("Auckland", "NewZealand"),
    ("Cairo", "Egypt"),
]

_NAMES: list[str] = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Olivia", "Peggy",
    "Quinn", "Rupert", "Sybil", "Trent", "Uma", "Victor", "Wendy", "Xander",
    "Yolanda", "Zach", "Alex", "Blake", "Casey", "Drew", "Erin", "Finn",
    "Gale", "Harper", "Iris", "Jamie", "Kai", "Logan", "Morgan", "Nova",
]

# Distractor sentences: unrelated prose that mentions no country/city/name from the fact pool.
# WHY hand-fixed: any auto-generated distractor pool risks incidentally naming a city/country
# and leaking the answer. Enforced offline by the loader's leakage check per record; also
# guarded by a test.
_DISTRACTOR_SENTENCES: list[str] = [
    "The kettle whistled sharply on the old iron stove.",
    "She unpacked the wooden crate with slow, deliberate motions.",
    "A soft wind pushed the yellow curtain against the window frame.",
    "He tuned the ancient radio until static gave way to a violin.",
    "The library smelled faintly of dust, glue, and warm paper.",
    "Rain drummed against the tin roof for most of the evening.",
    "The bicycle chain rattled with every uneven turn of the pedal.",
    "A candle flickered on the table beside a half-eaten pear.",
    "The wooden bench creaked when he finally sat down to rest.",
    "She counted the coins twice before slipping them into her pocket.",
    "The dog dozed on the rug, one ear twitching now and then.",
    "A slow river of mist crept along the low garden wall.",
    "He sharpened the pencil until the tip was almost invisible.",
    "The clock on the mantel ticked louder than seemed reasonable.",
    "She folded the linen napkins into neat little triangles.",
    "The bread was still warm when it came out of the paper bag.",
    "He whistled tunelessly while polishing the brass door handle.",
    "The kite tugged at its string with surprising, insistent force.",
    "A single moth circled the porch light for the better part of an hour.",
    "She hummed to herself while stirring the pot with a wooden spoon.",
    "The hallway echoed with the soft click of shoes on old tile.",
    "He wound the pocket watch slowly and set it beside the lamp.",
    "The garden gate squeaked in a way that made the cat flinch.",
    "She traced a lazy circle on the window with her fingertip.",
    "The morning was quiet except for the hum of a distant tractor.",
    "He filled the watering can and set off toward the tomato bed.",
    "A stack of newspapers slouched in the corner of the porch.",
    "The chess pieces were carved from a mottled dark wood.",
    "She read the letter twice, then tucked it into her sleeve.",
    "The camera shutter clicked in a slow, deliberate rhythm.",
]


def _record_rng(seed: int, record_idx: int) -> random.Random:
    # Same mixing as synthetic.py so the two loaders' seed spaces are independent.
    return random.Random((seed & 0xFFFFFFFF) * 1_000_003 + record_idx + 7)


def _arrange_pair(
    n_docs: int, distance: int, pair_position: str, rng: random.Random
) -> tuple[int, int]:
    """Return (doc_A_idx, doc_B_idx) placing the pair with `distance` distractors between them.

    `pair_position`:
      - "start"  -> doc_A sits as close to index 0 as possible
      - "end"    -> doc_B sits as close to index n_docs-1 as possible
      - "middle" -> pair centred inside the doc list

    Distance is capped to `n_docs - 2` (the maximum room for gaps in an N-doc list).
    """
    max_distance = n_docs - 2
    if distance > max_distance:
        distance = max_distance
    span = distance + 1  # doc_A_idx + span == doc_B_idx
    if pair_position == "start":
        a = 0
    elif pair_position == "end":
        a = n_docs - 1 - span
    elif pair_position == "middle":
        # Centre the pair; ties break toward the earlier slot for determinism.
        a = (n_docs - 1 - span) // 2
    else:
        raise ValueError(f"unknown pair_position {pair_position!r}")
    a = max(0, min(a, n_docs - 1 - span))
    b = a + span
    # rng is threaded for future tie-breaking freedom; unused today keeps determinism explicit.
    _ = rng
    return a, b


def _make_record(record_idx: int, seed: int, n_docs: int) -> dict:
    rng = _record_rng(seed, record_idx)

    distance = rng.randint(0, min(_MAX_DISTANCE, n_docs - 2))
    pair_position = rng.choice(("start", "middle", "end"))
    name = rng.choice(_NAMES)
    city, country = rng.choice(_CITY_COUNTRY_PAIRS)

    doc_a = f"{name} lives in {city}."
    doc_b = f"{city} is in {country}."

    doc_a_idx, doc_b_idx = _arrange_pair(n_docs, distance, pair_position, rng)

    # WHY the leakage guard: if a distractor sentence incidentally names <CITY> or <COUNTRY>
    # (or the entity <name>), the answer becomes reachable from a single distractor and the
    # composition property collapses. Reject such sentences by re-sampling. The distractor
    # pool is small and hand-curated so this is fast in practice.
    forbidden = {name.lower(), city.lower(), country.lower()}
    n_distractors = n_docs - 2
    pool = list(_DISTRACTOR_SENTENCES)
    rng.shuffle(pool)
    distractors: list[str] = []
    for sent in pool:
        low = sent.lower()
        if any(f in low for f in forbidden):
            continue
        distractors.append(sent)
        if len(distractors) == n_distractors:
            break
    if len(distractors) < n_distractors:
        # Falls back to repeating the safe pool; extremely unlikely with 30 sentences and 3 stopwords.
        raise RuntimeError(
            f"composition: could not find {n_distractors} leakage-free distractors "
            f"for record {record_idx}"
        )

    docs: list[str] = [""] * n_docs
    docs[doc_a_idx] = doc_a
    docs[doc_b_idx] = doc_b
    d_iter = iter(distractors)
    for i in range(n_docs):
        if i == doc_a_idx or i == doc_b_idx:
            continue
        docs[i] = next(d_iter)

    question = f"What country does {name} live in?"
    record_id = f"comp_{seed}_{record_idx:06d}"
    return {
        "id": record_id,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": country},
        ],
        "possible_answers": [country],
        "hop_type": "multi",
        "gold_position": None,
        "meta": {
            "distance": distance,
            "pair_position": pair_position,
            "doc_A_idx": doc_a_idx,
            "doc_B_idx": doc_b_idx,
            "entity": name,
            "city": city,
            "country": country,
            "n_docs": n_docs,
        },
    }


def _all_records(seed: int, n_docs: int) -> list[dict]:
    return [_make_record(i, seed, n_docs) for i in range(_N_RECORDS)]


@register_loader("composition")
def _load_composition(split: Split, seed: int) -> Iterator[dict]:
    """Generate records with a fact-composition structure.

    Each record's info_contexts has N=5 docs; the answer requires composing exactly two of them.
    Split sizes: train=60, val=20, test=20. Split partition is deterministic in seed. Every
    record's meta carries `distance`, `pair_position`, `doc_A_idx`, `doc_B_idx`.
    """
    n_docs = _DEFAULT_N_DOCS
    records = _all_records(seed, n_docs)
    all_ids = [r["id"] for r in records]
    by_split = split_ids(all_ids, val_frac=_VAL_FRAC, test_frac=_TEST_FRAC, seed=seed)

    hashes = {name: (len(ids), hash_split(ids)) for name, ids in by_split.items()}
    validate_or_write("composition", hashes)

    wanted = set(by_split[split])
    by_id = {r["id"]: r for r in records}
    for rid in sorted(wanted):
        yield by_id[rid]
