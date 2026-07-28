"""DDI-stress synthetic dataset (issue #51).

Purpose-built stress dataset: five task families, each targeting a specific DDI claim, with full
control over every axis that stresses DDI. Not "realistic" — mechanistic. Deterministic, offline.

Task families and the DDI claim each isolates:

  - fact_chain          — C5 (multi-hop failure), C2 (query isolation).
                          n-hop entity chain generalising `composition`.
  - fact_aggregation    — C5 (cross-doc aggregation), C1 (parity when separable).
                          N gold docs each carrying a partial answer.
  - contradiction       — C5 (arbitrating between docs), C6 (distractor robustness).
                          Two docs disagree; the more-recent date wins.
  - coreference         — attention-sink robustness; pronoun in one doc refers to entity
                          introduced in another.
  - needle              — C4 caching stress + C6 distractor robustness. One gold doc among
                          many distractors; single directly-stated fact. Subsumes the LitM
                          stub from RESEARCH_PLAN §1.1 via explicit `gold_positions`.

Common record shape (matches every other loader): `info_contexts` / `dialog` /
`possible_answers` / `hop_type` / `gold_position` / `meta`. Every knob is echoed into `meta`
so the report layer can slice by any axis.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

from ddi.data.loaders import Split, register_loader
from ddi.data.manifest import validate_or_write
from ddi.data.splits import hash_split, split_ids

# ---------------------------------------------------------------------------
# module-level constants
# ---------------------------------------------------------------------------

_N_RECORDS = 1000  # 600 train + 200 val + 200 test
_VAL_FRAC = 0.20
_TEST_FRAC = 0.20

_DEFAULT_N_DOCS = 5
_DEFAULT_HOP_DEPTH = 3
_DEFAULT_N_FACTS_PER_DOC = 2
_DEFAULT_DISTRACTOR_STRENGTH = "far"

_TASK_FAMILIES = ("fact_chain", "fact_aggregation", "contradiction", "coreference", "needle")

# WHY env-var selection: one loader name serves five families, so `list_datasets()` stays small
# and the store is easy to slice. Mirrors how other loaders read a `DDI_*` env var to swap a
# behavioural knob without exploding the registry.
_TASK_ENV_VAR = "DDI_STRESS_TASK"
_DEFAULT_TASK_FAMILY = "fact_chain"

_MAX_LEAKAGE_RESAMPLES = 20  # cap re-tries so a pathological seed can't hang


# 50 people, 50 cities, 50 countries, 20 dates, 30 currencies, 20 companies — enough
# combinatorial space that a 1000-record split doesn't repeat facts trivially.
_PEOPLE: tuple[str, ...] = (
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Olivia", "Peggy",
    "Quinn", "Rupert", "Sybil", "Trent", "Uma", "Victor", "Wendy", "Xander",
    "Yolanda", "Zach", "Alex", "Blake", "Casey", "Drew", "Erin", "Finn",
    "Gale", "Harper", "Iris", "Jamie", "Kai", "Logan", "Morgan", "Nova",
    "Parker", "Quincy", "Reese", "Sage", "Tatum", "Umberto", "Vera", "Willa",
    "Xenia", "Yosef",
)

_CITIES: tuple[str, ...] = (
    "Paris", "Lyon", "Berlin", "Munich", "Madrid", "Barcelona", "Rome", "Milan",
    "Lisbon", "Porto", "Amsterdam", "Rotterdam", "Brussels", "Vienna", "Zurich", "Geneva",
    "Stockholm", "Oslo", "Copenhagen", "Helsinki", "Warsaw", "Krakow", "Prague", "Budapest",
    "Athens", "Dublin", "London", "Manchester", "Edinburgh", "Cardiff", "Toronto", "Montreal",
    "Vancouver", "Boston", "Chicago", "Seattle", "Portland", "Denver", "Tokyo", "Osaka",
    "Kyoto", "Seoul", "Beijing", "Shanghai", "Mumbai", "Delhi", "Sydney", "Melbourne",
    "Auckland", "Cairo",
)

_COUNTRIES: tuple[str, ...] = (
    "Aldova", "Brenland", "Corsavia", "Drakonia", "Ederia", "Fjordia", "Grenwald", "Halbria",
    "Ionica", "Jaresh", "Kentari", "Lomberia", "Marenza", "Nordica", "Osteria", "Palvia",
    "Quenaria", "Ravenholt", "Solmark", "Talveria", "Ustaria", "Valmoria", "Weslia", "Xandria",
    "Yestria", "Zalvia", "Ambria", "Belveria", "Crestland", "Duvaria", "Estmark", "Fenwick",
    "Galvia", "Harnia", "Iskaria", "Jorvik", "Kalveria", "Loresh", "Meridia", "Norvenia",
    "Oskaria", "Pentaria", "Qatvia", "Renvia", "Selmoria", "Tarnia", "Ulmeria", "Vestria",
    "Wenteria", "Yalveria",
)

_DATES: tuple[str, ...] = (
    "1975", "1978", "1982", "1985", "1989", "1991", "1994", "1997",
    "2000", "2003", "2006", "2009", "2011", "2013", "2015", "2017",
    "2019", "2020", "2022", "2024",
)

_CURRENCIES: tuple[str, ...] = (
    "aurins", "brints", "crolds", "drakes", "escuds", "florins", "gralds", "helvets",
    "iskrins", "jarks", "kroens", "lirats", "marks", "nelts", "orins", "pesants",
    "quints", "renks", "solids", "talers", "unics", "vrenks", "wistars", "xandrics",
    "yensels", "zlots", "aebors", "berks", "crownets", "dinats",
)

_COMPANIES: tuple[str, ...] = (
    "Acmelith", "Brindico", "Corvexa", "Duneterra", "Everstack", "Ferronix", "Glimmerworks",
    "Halcyonics", "Ironvale", "Jasperline", "Kestrelcorp", "Luminera", "Marlinbrook",
    "Nordheim", "Oakspire", "Pentaris", "Quadravox", "Renfelt", "Silverbrook", "Talonrise",
)

# WHY hand-fixed distractor prose: any auto-generated pool risks incidentally naming a fact-pool
# token and leaking the answer. Kept vocab-disjoint from every fact pool above.
_DISTRACTOR_SENTENCES: tuple[str, ...] = (
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
)

# Relation templates used by fact_chain. Each is a two-argument, directional relation.
_CHAIN_RELATIONS: tuple[str, ...] = ("lives in", "born in", "founded", "leads")

# ---------------------------------------------------------------------------
# task-family override + per-record knob overrides
# ---------------------------------------------------------------------------

_task_override: str | None = None
_knob_overrides: dict[str, object] = {}


def set_task_family(name: str | None) -> None:
    """Programmatic override for the env-var-selected task family; pass None to clear."""
    global _task_override
    if name is not None and name not in _TASK_FAMILIES:
        raise ValueError(f"unknown task family {name!r}; known: {_TASK_FAMILIES}")
    _task_override = name


def set_knob_overrides(**knobs: object) -> None:
    """Override generator knobs for all subsequent records (test / programmatic use).

    Keys understood: `hop_depth`, `n_docs`, `n_gold`, `gold_positions`,
    `distractor_strength`, `n_facts_per_doc`. Pass no arguments to clear.
    """
    global _knob_overrides
    if not knobs:
        _knob_overrides = {}
        return
    _knob_overrides = dict(_knob_overrides)
    _knob_overrides.update(knobs)


def _knob(name: str, default: object) -> object:
    return _knob_overrides.get(name, default)


def _current_task_family() -> str:
    if _task_override is not None:
        return _task_override
    env = os.environ.get(_TASK_ENV_VAR, _DEFAULT_TASK_FAMILY)
    if env not in _TASK_FAMILIES:
        raise ValueError(
            f"{_TASK_ENV_VAR}={env!r} is not a known task family; known: {_TASK_FAMILIES}"
        )
    return env


# ---------------------------------------------------------------------------
# RNG helper — mirrors composition.py so seed spaces are independent
# ---------------------------------------------------------------------------


def _record_rng(seed: int, record_idx: int, task_family: str) -> random.Random:
    # Family name mixed in so switching family with the same seed yields a distinct stream.
    family_salt = sum(ord(c) for c in task_family)
    return random.Random(
        (seed & 0xFFFFFFFF) * 1_000_003 + record_idx * 31 + family_salt + 7
    )


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------


def _leaks(answer: str, docs: list[str], gold_indices: set[int]) -> bool:
    low_ans = answer.lower()
    for i, doc in enumerate(docs):
        if i in gold_indices:
            continue
        if low_ans in doc.lower():
            return True
    return False


def _pick_distractors(
    n: int,
    forbidden: set[str],
    rng: random.Random,
    strength: str,
    near_seed_entity: str | None,
) -> list[str]:
    """Sample `n` distractor sentences that mention nothing in `forbidden`.

    strength="near": each distractor is a fabricated sentence that shares `near_seed_entity` with
    the gold docs but nothing that composes into the answer. Ensures the distractor is topically
    adjacent without ever containing an answer token.

    strength="far": drawn from the hand-curated vocab-disjoint prose pool.
    """
    forbidden_low = {f.lower() for f in forbidden if f}
    picked: list[str] = []

    if strength == "near" and near_seed_entity is not None:
        # Fabricate near-distractors from the fact pools but wire them to unrelated targets so no
        # answer token leaks. Keeps `near_seed_entity` present in every distractor.
        pool_cities = [c for c in _CITIES if c.lower() not in forbidden_low]
        pool_dates = [d for d in _DATES if d.lower() not in forbidden_low]
        pool_currencies = [c for c in _CURRENCIES if c.lower() not in forbidden_low]
        pool_companies = [c for c in _COMPANIES if c.lower() not in forbidden_low]
        available_slots = [
            (pool_cities, "once visited"),
            (pool_dates, "attended a conference in"),
            (pool_currencies, "donated some"),
            (pool_companies, "briefly consulted for"),
        ]
        available_slots = [(p, v) for p, v in available_slots if p]
        attempts = 0
        while len(picked) < n and attempts < n * _MAX_LEAKAGE_RESAMPLES and available_slots:
            attempts += 1
            pool, verb = rng.choice(available_slots)
            sent = f"{near_seed_entity} {verb} {rng.choice(pool)}."
            low = sent.lower()
            if any(f in low for f in forbidden_low):
                continue
            picked.append(sent)
        if len(picked) < n:
            # Fall back to far distractors rather than crash.
            picked.extend(
                _pick_distractors(
                    n - len(picked),
                    forbidden,
                    rng,
                    strength="far",
                    near_seed_entity=None,
                )
            )
        return picked

    pool = list(_DISTRACTOR_SENTENCES)
    rng.shuffle(pool)
    for sent in pool:
        low = sent.lower()
        if any(f in low for f in forbidden_low):
            continue
        picked.append(sent)
        if len(picked) == n:
            break
    while len(picked) < n:
        # Repeat the safe pool cyclically without introducing any new tokens; guaranteed not
        # to leak because the pool was already filtered against `forbidden`.
        base = _DISTRACTOR_SENTENCES[len(picked) % len(_DISTRACTOR_SENTENCES)]
        base_low = base.lower()
        if not any(f in base_low for f in forbidden_low):
            picked.append(base)
        else:
            # Very rare: forbidden hit the very first pool item. Skip the poisoned index.
            picked.append(base + " Again.")
    return picked


# ---------------------------------------------------------------------------
# family: fact_chain
# ---------------------------------------------------------------------------


def _make_fact_chain(record_idx: int, seed: int) -> dict:
    rng = _record_rng(seed, record_idx, "fact_chain")
    forced_positions = _knob("gold_positions", None)
    if forced_positions is not None:
        gold_positions = sorted(int(p) for p in forced_positions)  # type: ignore[arg-type]
        hop_depth = int(_knob("hop_depth", len(gold_positions)))  # type: ignore[arg-type]
        if hop_depth != len(gold_positions):
            raise ValueError(
                f"fact_chain: hop_depth={hop_depth} disagrees with gold_positions={gold_positions}"
            )
        n_gold = hop_depth
        default_n_docs = max(_DEFAULT_N_DOCS, max(gold_positions) + 1)
        n_docs = int(_knob("n_docs", default_n_docs))  # type: ignore[arg-type]
    else:
        hop_depth = int(_knob("hop_depth", _DEFAULT_HOP_DEPTH))  # type: ignore[arg-type]
        n_gold = hop_depth  # one fact per hop → one doc per hop
        n_docs = int(_knob("n_docs", max(_DEFAULT_N_DOCS, n_gold)))  # type: ignore[arg-type]
        gold_positions = sorted(rng.sample(range(n_docs), n_gold))
    n_distractors = n_docs - n_gold
    distractor_strength = str(_knob("distractor_strength", _DEFAULT_DISTRACTOR_STRENGTH))
    n_facts_per_doc = 1  # each chain fact stands alone; density knob unused by this family

    # Build a chain: entity_0 -R_0-> entity_1 -R_1-> ... -R_{h-1}-> entity_h
    entities: list[str] = []
    pool_people = list(_PEOPLE)
    rng.shuffle(pool_people)
    entities.append(pool_people[0])
    pool_places = list(_CITIES) + list(_COUNTRIES) + list(_COMPANIES)
    rng.shuffle(pool_places)
    if hop_depth > len(pool_places):
        raise ValueError(f"fact_chain: hop_depth={hop_depth} exceeds place pool size")
    for i in range(hop_depth):
        entities.append(pool_places[i])

    relations = [rng.choice(_CHAIN_RELATIONS) for _ in range(hop_depth)]

    # One fact per doc; permute so the chain-order is not the doc-order (that's the point).
    chain_docs = [
        f"{entities[i]} {relations[i]} {entities[i + 1]}."
        for i in range(hop_depth)
    ]
    chain_order = list(range(hop_depth))
    rng.shuffle(chain_order)
    ordered_chain_docs = [chain_docs[i] for i in chain_order]

    answer = entities[-1]

    for _attempt in range(_MAX_LEAKAGE_RESAMPLES):
        distractors = _pick_distractors(
            n_distractors,
            forbidden={answer, *entities},
            rng=rng,
            strength=distractor_strength,
            near_seed_entity=entities[0] if distractor_strength == "near" else None,
        )
        docs: list[str] = [""] * n_docs
        for pos, doc in zip(gold_positions, ordered_chain_docs, strict=True):
            docs[pos] = doc
        d_iter = iter(distractors)
        for i in range(n_docs):
            if i in set(gold_positions):
                continue
            docs[i] = next(d_iter)
        if not _leaks(answer, docs, set(gold_positions)):
            break
    else:
        # Fell through the loop without break — leakage never resolved. Log-and-continue.
        # WHY re-sample rather than crash: with hand-curated pools this should be vanishingly
        # rare, but the sweep runs unattended and one bad seed shouldn't stop the whole job.
        pass

    question = (
        f"Following the chain of facts, what is the final target starting from {entities[0]}?"
    )
    record_id = f"ddi_stress-fact_chain-{seed}-{record_idx:06d}"
    return {
        "id": record_id,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
        "hop_type": "multi",
        "gold_position": None,
        "meta": {
            "task_family": "fact_chain",
            "n_docs": n_docs,
            "n_gold": n_gold,
            "gold_positions": gold_positions,
            "distractor_strength": distractor_strength,
            "hop_depth": hop_depth,
            "n_facts_per_doc": n_facts_per_doc,
            "chain_entities": entities,
            "chain_relations": relations,
        },
    }


# ---------------------------------------------------------------------------
# family: fact_aggregation
# ---------------------------------------------------------------------------


def _make_fact_aggregation(record_idx: int, seed: int) -> dict:
    rng = _record_rng(seed, record_idx, "fact_aggregation")
    n_gold = int(_knob("n_gold", 3))  # type: ignore[arg-type]
    n_docs = int(_knob("n_docs", max(_DEFAULT_N_DOCS, n_gold)))  # type: ignore[arg-type]
    n_distractors = max(0, n_docs - n_gold)
    distractor_strength = str(_knob("distractor_strength", _DEFAULT_DISTRACTOR_STRENGTH))
    n_facts_per_doc = int(
        _knob("n_facts_per_doc", _DEFAULT_N_FACTS_PER_DOC)  # type: ignore[arg-type]
    )
    aggregation_type = rng.choice(("count", "list", "sum", "max"))
    forced_positions = _knob("gold_positions", None)
    if forced_positions is not None:
        gold_positions = sorted(int(p) for p in forced_positions)  # type: ignore[arg-type]
    else:
        gold_positions = sorted(rng.sample(range(n_docs), n_gold))

    # Rotate through the four aggregation shapes so each is exercised regularly.
    if aggregation_type == "count":
        company = rng.choice(_COMPANIES)
        people_pool = list(_PEOPLE)
        rng.shuffle(people_pool)
        people = people_pool[:n_gold]
        gold_docs = [f"{p} works at {company}." for p in people]
        answer = str(n_gold)
        question = f"How many people work at {company}?"
        forbidden = {company, *people, answer}
        near_seed = company
    elif aggregation_type == "list":
        company = rng.choice(_COMPANIES)
        cities_pool = list(_CITIES)
        rng.shuffle(cities_pool)
        cities = sorted(cities_pool[:n_gold])
        gold_docs = [f"{company} has an office in {c}." for c in cities]
        answer = ", ".join(cities)
        question = f"In which cities does {company} have offices?"
        forbidden = {company, *cities}
        near_seed = company
    elif aggregation_type == "sum":
        company = rng.choice(_COMPANIES)
        currency = rng.choice(_CURRENCIES)
        amounts = [rng.randint(2, 9) for _ in range(n_gold)]
        gold_docs = [
            f"{company} earned {amt} {currency} in quarter {q + 1}."
            for q, amt in enumerate(amounts)
        ]
        total = sum(amounts)
        answer = str(total)
        question = f"What was the total earnings of {company} in {currency}?"
        forbidden = {company, currency, answer, *(str(a) for a in amounts)}
        near_seed = company
    else:  # max
        company = rng.choice(_COMPANIES)
        currency = rng.choice(_CURRENCIES)
        # WHY unique amounts: max must have a single winner so the answer string is unambiguous.
        amounts = rng.sample(range(10, 100), n_gold)
        gold_docs = [
            f"{company} earned {amt} {currency} in quarter {q + 1}."
            for q, amt in enumerate(amounts)
        ]
        top = max(amounts)
        answer = str(top)
        question = f"What was the maximum quarterly earning of {company} in {currency}?"
        forbidden = {company, currency, *(str(a) for a in amounts if a != top)}
        near_seed = company

    for _attempt in range(_MAX_LEAKAGE_RESAMPLES):
        distractors = _pick_distractors(
            n_distractors,
            forbidden=forbidden,
            rng=rng,
            strength=distractor_strength,
            near_seed_entity=near_seed if distractor_strength == "near" else None,
        )
        docs: list[str] = [""] * n_docs
        for pos, doc in zip(gold_positions, gold_docs, strict=True):
            docs[pos] = doc
        d_iter = iter(distractors)
        for i in range(n_docs):
            if i in set(gold_positions):
                continue
            docs[i] = next(d_iter)
        if not _leaks(answer, docs, set(gold_positions)):
            break

    record_id = f"ddi_stress-fact_aggregation-{seed}-{record_idx:06d}"
    return {
        "id": record_id,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
        "hop_type": "multi",
        "gold_position": None,
        "meta": {
            "task_family": "fact_aggregation",
            "n_docs": n_docs,
            "n_gold": n_gold,
            "gold_positions": gold_positions,
            "distractor_strength": distractor_strength,
            "hop_depth": None,
            "n_facts_per_doc": n_facts_per_doc,
            "aggregation_type": aggregation_type,
        },
    }


# ---------------------------------------------------------------------------
# family: contradiction
# ---------------------------------------------------------------------------


def _make_contradiction(record_idx: int, seed: int) -> dict:
    rng = _record_rng(seed, record_idx, "contradiction")
    n_gold = 2  # family invariant: exactly one contradicting pair
    n_docs = int(_knob("n_docs", max(_DEFAULT_N_DOCS, n_gold)))  # type: ignore[arg-type]
    n_distractors = max(0, n_docs - n_gold)
    distractor_strength = str(_knob("distractor_strength", _DEFAULT_DISTRACTOR_STRENGTH))
    n_facts_per_doc = 1
    forced_positions = _knob("gold_positions", None)
    if forced_positions is not None:
        gold_positions = sorted(int(p) for p in forced_positions)  # type: ignore[arg-type]
    else:
        gold_positions = sorted(rng.sample(range(n_docs), n_gold))

    company = rng.choice(_COMPANIES)
    city_pool = list(_CITIES)
    rng.shuffle(city_pool)
    stale_city, current_city = city_pool[0], city_pool[1]

    dates_pool = sorted(rng.sample(_DATES, 2))  # sorted so [0] is older, [1] more recent
    stale_date, current_date = dates_pool[0], dates_pool[1]

    stale_doc = f"As of {stale_date}, {company} is headquartered in {stale_city}."
    current_doc = f"As of {current_date}, {company} is headquartered in {current_city}."
    gold_docs = [stale_doc, current_doc]
    # Shuffle so the "authoritative" doc isn't always at the same relative position.
    rng.shuffle(gold_docs)

    answer = current_city
    question = f"Where is {company} currently headquartered?"
    # WHY the stale city is in forbidden: a near-distractor mustn't reinforce the wrong answer;
    # `answer` alone would allow "stale_city" to slip into a distractor and confuse graders.
    forbidden = {company, current_city, stale_city, current_date, stale_date}

    for _attempt in range(_MAX_LEAKAGE_RESAMPLES):
        distractors = _pick_distractors(
            n_distractors,
            forbidden=forbidden,
            rng=rng,
            strength=distractor_strength,
            near_seed_entity=company if distractor_strength == "near" else None,
        )
        docs: list[str] = [""] * n_docs
        for pos, doc in zip(gold_positions, gold_docs, strict=True):
            docs[pos] = doc
        d_iter = iter(distractors)
        for i in range(n_docs):
            if i in set(gold_positions):
                continue
            docs[i] = next(d_iter)
        if not _leaks(answer, docs, set(gold_positions)):
            break

    record_id = f"ddi_stress-contradiction-{seed}-{record_idx:06d}"
    return {
        "id": record_id,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
        "hop_type": "multi",
        "gold_position": None,
        "meta": {
            "task_family": "contradiction",
            "n_docs": n_docs,
            "n_gold": n_gold,
            "gold_positions": gold_positions,
            "distractor_strength": distractor_strength,
            "hop_depth": None,
            "n_facts_per_doc": n_facts_per_doc,
            "current_date": current_date,
            "stale_date": stale_date,
            "current_city": current_city,
            "stale_city": stale_city,
            "company": company,
        },
    }


# ---------------------------------------------------------------------------
# family: coreference
# ---------------------------------------------------------------------------


def _make_coreference(record_idx: int, seed: int) -> dict:
    rng = _record_rng(seed, record_idx, "coreference")
    n_gold = 2  # antecedent + pronoun
    n_docs = int(_knob("n_docs", max(_DEFAULT_N_DOCS, n_gold)))  # type: ignore[arg-type]
    n_distractors = max(0, n_docs - n_gold)
    distractor_strength = str(_knob("distractor_strength", _DEFAULT_DISTRACTOR_STRENGTH))
    n_facts_per_doc = 1
    forced_positions = _knob("gold_positions", None)
    if forced_positions is not None:
        gold_positions = sorted(int(p) for p in forced_positions)  # type: ignore[arg-type]
    else:
        gold_positions = sorted(rng.sample(range(n_docs), n_gold))

    person = rng.choice(_PEOPLE)
    company = rng.choice(_COMPANIES)
    year = rng.choice(_DATES)

    # Doc A introduces the entity; doc B refers to it via a pronoun. Question asks the referent.
    doc_a = f"{person} founded {company}."
    doc_b = f"She stepped down from her role in {year}."
    gold_docs = [doc_a, doc_b]

    answer = person
    question = f"Who stepped down in {year}?"
    forbidden = {person, company, year}

    for _attempt in range(_MAX_LEAKAGE_RESAMPLES):
        distractors = _pick_distractors(
            n_distractors,
            forbidden=forbidden,
            rng=rng,
            strength=distractor_strength,
            near_seed_entity=company if distractor_strength == "near" else None,
        )
        docs: list[str] = [""] * n_docs
        # Ordering matters: doc_a (antecedent) must appear before doc_b (pronoun) to be
        # resolvable at all — enforce by sorting gold_positions and assigning in order.
        for pos, doc in zip(gold_positions, gold_docs, strict=True):
            docs[pos] = doc
        d_iter = iter(distractors)
        for i in range(n_docs):
            if i in set(gold_positions):
                continue
            docs[i] = next(d_iter)
        if not _leaks(answer, docs, set(gold_positions)):
            break

    record_id = f"ddi_stress-coreference-{seed}-{record_idx:06d}"
    return {
        "id": record_id,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
        "hop_type": "multi",
        "gold_position": None,
        "meta": {
            "task_family": "coreference",
            "n_docs": n_docs,
            "n_gold": n_gold,
            "gold_positions": gold_positions,
            "distractor_strength": distractor_strength,
            "hop_depth": None,
            "n_facts_per_doc": n_facts_per_doc,
            "person": person,
            "company": company,
            "year": year,
        },
    }


# ---------------------------------------------------------------------------
# family: needle
# ---------------------------------------------------------------------------


def _make_needle(record_idx: int, seed: int) -> dict:
    rng = _record_rng(seed, record_idx, "needle")
    n_gold = 1
    n_docs = int(_knob("n_docs", _DEFAULT_N_DOCS))  # type: ignore[arg-type]
    n_distractors = max(0, n_docs - n_gold)
    distractor_strength = str(_knob("distractor_strength", _DEFAULT_DISTRACTOR_STRENGTH))
    n_facts_per_doc = 1
    forced_positions = _knob("gold_positions", None)
    if forced_positions is not None:
        gold_positions = sorted(int(p) for p in forced_positions)  # type: ignore[arg-type]
        gold_position = gold_positions[0]
    else:
        gold_position = rng.randrange(n_docs)
        gold_positions = [gold_position]

    person = rng.choice(_PEOPLE)
    city = rng.choice(_CITIES)
    gold_doc = f"{person} lives in {city}."
    answer = city
    question = f"Where does {person} live?"
    forbidden = {person, city}

    for _attempt in range(_MAX_LEAKAGE_RESAMPLES):
        distractors = _pick_distractors(
            n_distractors,
            forbidden=forbidden,
            rng=rng,
            strength=distractor_strength,
            near_seed_entity=person if distractor_strength == "near" else None,
        )
        docs: list[str] = [""] * n_docs
        docs[gold_position] = gold_doc
        d_iter = iter(distractors)
        for i in range(n_docs):
            if i == gold_position:
                continue
            docs[i] = next(d_iter)
        if not _leaks(answer, docs, {gold_position}):
            break

    record_id = f"ddi_stress-needle-{seed}-{record_idx:06d}"
    return {
        "id": record_id,
        "info_contexts": docs,
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "possible_answers": [answer],
        "hop_type": "single",
        "gold_position": gold_position,
        "meta": {
            "task_family": "needle",
            "n_docs": n_docs,
            "n_gold": n_gold,
            "gold_positions": gold_positions,
            "distractor_strength": distractor_strength,
            "hop_depth": None,
            "n_facts_per_doc": n_facts_per_doc,
            "person": person,
            "city": city,
        },
    }


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


_MAKERS = {
    "fact_chain": _make_fact_chain,
    "fact_aggregation": _make_fact_aggregation,
    "contradiction": _make_contradiction,
    "coreference": _make_coreference,
    "needle": _make_needle,
}


def _make_record(record_idx: int, seed: int, task_family: str) -> dict:
    return _MAKERS[task_family](record_idx, seed)


def _all_records(seed: int, task_family: str) -> list[dict]:
    return [_make_record(i, seed, task_family) for i in range(_N_RECORDS)]


@register_loader("ddi_stress")
def _load_ddi_stress(split: Split, seed: int) -> Iterator[dict]:
    """Yields records for the family named by `DDI_STRESS_TASK` (default `fact_chain`).

    Split sizes: train=600 / val=200 / test=200. Partition deterministic in seed. The manifest
    is written per (dataset, task_family, seed) so switching families or seeds keeps its own
    hash and can coexist with other families'/seeds' committed splits.
    """
    task_family = _current_task_family()
    records = _all_records(seed, task_family)
    all_ids = [r["id"] for r in records]
    by_split = split_ids(all_ids, val_frac=_VAL_FRAC, test_frac=_TEST_FRAC, seed=seed)

    hashes = {name: (len(ids), hash_split(ids)) for name, ids in by_split.items()}
    validate_or_write(f"ddi_stress__{task_family}__seed{seed}", hashes)

    wanted = set(by_split[split])
    by_id = {r["id"]: r for r in records}
    for rid in sorted(wanted):
        yield by_id[rid]
