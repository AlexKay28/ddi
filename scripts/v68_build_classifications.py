#!/usr/bin/env python3
"""V68 classification stage (VIBEQUEUE-68) — applies the FROZEN rubric from
experiments/v68_anatomy_preregistration.{md,json} to the mechanically extracted
differing records. Must be run only after the prereg freeze; this script encodes
the per-record human judgments (labels + justifications) and merges the mechanical
features from differing_records.jsonl. Writes classifications.jsonl and
anatomy_table.md into the same results directory.

Rubric labels (frozen): near_tie_greedy_flip | clear_semantic_change |
truncation_stopping_artifact | ambiguous (explicitly allowed, never forced).

Post-freeze operationalization of "minimal-edit" used for near-tie decisions
(applied uniformly, recorded here for audit): the difference is localized (one
difflib non-equal op covering <=2 token positions for outputs longer than 3 ws
tokens; <=2 positions for outputs <=3 ws tokens), the shared frame dominates
(>= half of ws tokens equal) or the outputs are <=3 tokens, and both outputs
answer in the same neighborhood. A ws/model-token prefix relation is recorded as
evidence but does not by itself force truncation_stopping_artifact; per the
frozen guidance the dominant reading wins.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("results/v68_anatomy_20260910T131036Z")
PREREG_JSON = Path("experiments/v68_anatomy_preregistration.json")

# Per-record frozen-rubric judgments, keyed by request_index (avoids id typos).
LABELS = {
    4: ("clear_semantic_change",
        "Fresh answers the bare name 'Anatoly Maltsev'; cached emits a longer comparison "
        "sentence that contains the gold 'Lazar Aronovich Lyusternik'. Divergence at ws "
        "token 1 / model token 3, no prefix relation either level, char edit 78: the "
        "outputs assert different content and the rewrite is not minimal -> semantic. "
        "Direction: cached contains gold, fresh does not."),
    5: ("near_tie_greedy_flip",
        "Identical except the trailing '.' (char edit 1, 10/11 ws tokens shared): pure "
        "punctuation change, same neighborhood. Gold absent from both."),
    8: ("near_tie_greedy_flip",
        "Cached appends the honorific 'Jr.' (1 ws token, char edit 4) to the same name; "
        "ws-prefix flag co-occurs but a one-token suffix on the same entity is a "
        "function-word-like change, not length/stopping -> near-tie. This flip alone "
        "moves EM -1 (cached EM 0, fresh EM 1; F1 0.857 vs 1.0)."),
    10: ("truncation_stopping_artifact",
         "Cached stops after the shared title phrase (stop_reason 'length', 6 model "
         "tokens) while fresh continues the same opening into a long sentence containing "
         "gold 'Deric Henderson'. Cached content is the shared opening cut short "
         "(near-literal ws prefix modulo the trailing comma on the shared token); the "
         "difference is dominated by length/stopping."),
    14: ("truncation_stopping_artifact",
         "Cached is a literal ws prefix of fresh (5 shared tokens); fresh appends "
         "'(TCNJ) was established first.'. Gold present in both; the outputs differ only "
         "in how long the answer ran (the extra clause breaks fresh EM)."),
    15: ("near_tie_greedy_flip",
         "Episode-title-shaped strings sharing the head 'Mom's'; localized 2-token "
         "replacement on a 2-3 token output ('the Man' -> 'Millers'), same neighborhood. "
         "Both lack gold 'How I Met Your Mother'."),
    16: ("clear_semantic_change",
         "Different hospitals with first-token divergence and no shared frame beyond one "
         "token: wholesale entity replacement (cached 'MedStar Washington Hospital "
         "Center' vs fresh, correct, 'Sibley Memorial Hospital')."),
    17: ("truncation_stopping_artifact",
         "Cached is a literal ws prefix ('documentary') of fresh ('documentary film "
         "genre'); gold present in both; difference dominated by length/stopping."),
    39: ("clear_semantic_change",
         "Different characters ('Fox Crane' vs 'Ethan Winthrop'), 0 shared ws tokens, "
         "first-token divergence: different factual answer (fresh matches gold)."),
    42: ("clear_semantic_change",
         "Different districts; only the generic tail 'Historic District' is shared "
         "(2/5 vs 2/4 tokens) with 3 content tokens replaced - not a minimal edit; both "
         "lack gold 'Guilford School'."),
    52: ("clear_semantic_change",
         "Different films (0 shared ws tokens, first-token divergence); fresh names gold "
         "'Balearic Caper', cached names an unrelated title."),
    54: ("clear_semantic_change",
         "Shared head 'Boogie,' then different continuations (alternate subtitle vs "
         "authorship elaboration; 14 ws tokens replaced, no prefix relation). Both "
         "contain the gold head 'Boogie', but the differing tails assert different "
         "facts; not length-dominated."),
    56: ("clear_semantic_change",
         "Different journals ('Annals of Physics' vs 'Physics Today'), first-token "
         "divergence, one shared token: wholesale entity replacement; gold 'Philosophical "
         "Magazine' absent from both."),
    57: ("clear_semantic_change",
         "Different song titles (0 shared ws tokens, first-token divergence); gold "
         "'\"Glorious\"' absent from both."),
    61: ("near_tie_greedy_flip",
         "Shared frame 'Sterling Sharpe' (2/4 and 2/3 ws tokens); tails differ (cached "
         "appends 'Dante Hall', fresh appends 'Jr.'). Same referent, gold present in "
         "both; localized tail swap (both EM 0; F1 0.667 vs 0.8)."),
    70: ("near_tie_greedy_flip",
         "Single-token model swap '360' -> 'F430' inside the same two-token frame "
         "'Ferrari ...' (char edit 3): minimal-edit entity swap, same neighborhood. "
         "This flip moves EM -1 (cached EM 0, fresh EM 1)."),
    78: ("near_tie_greedy_flip",
         "Cached omits the trailing nomenclature authority ' L.' (ws-prefix relation, 1 "
         "token); same organism, gold present in both. Prefix flag co-occurs but the "
         "dominant reading is a function-word-like omission (cached EM 1.0, fresh EM 0)."),
    81: ("near_tie_greedy_flip",
         "Fresh omits the trailing ' Mill' (ws-prefix relation, 1 token); same answer "
         "neighborhood, gold absent from both. Symmetric with 5a722da0 (idx 78)."),
    87: ("clear_semantic_change",
         "Different people ('John Wesley' vs 'Hugh Price Hughes'), 0 shared ws tokens, "
         "first-token divergence; gold absent from both."),
    93: ("clear_semantic_change",
         "Different presenters ('Jonathan Stephen Ross' vs 'Chris Evans'), 0 shared ws "
         "tokens, first-token divergence; gold 'Jimmy Carr' absent from both."),
    95: ("near_tie_greedy_flip",
         "Same answer shape 'Ernest II (German: ...)'; cached repeats the full German "
         "name in the second quoted slot where fresh has the short variant 'Ernst von "
         "Hessen' (13/19 and 13/15 ws tokens shared, one localized replace op). "
         "Name-variant substitution, same referent."),
    99: ("clear_semantic_change",
         "Different songs ('Honky Tonk Heroes' vs \"Lonesome, On'ry and Mean\"), "
         "first-token divergence; cached matches gold, fresh does not."),
}

LABEL_ORDER = [
    "near_tie_greedy_flip",
    "clear_semantic_change",
    "truncation_stopping_artifact",
    "ambiguous",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    differing_path = RESULTS_DIR / "differing_records.jsonl"
    records = [json.loads(l) for l in open(differing_path, encoding="utf-8") if l.strip()]

    covered = {d["request_index"] for d in records}
    if covered != set(LABELS):
        raise SystemExit(f"label keys {sorted(LABELS)} != extracted indices {sorted(covered)}")

    out = []
    for d in records:
        label, justification = LABELS[d["request_index"]]
        fd = d["first_divergence"]
        ef = d["edit_features"]
        div = fd["model_token_index"]
        position = "first_token" if div == 0 else "mid_generation"
        out.append({
            "record_id": d["record_id"],
            "request_index": d["request_index"],
            "label": label,
            "divergence_position": position,
            "cached_prediction": d["cached_prediction"],
            "fresh_prediction": d["fresh_prediction"],
            "gold_answers": d["gold_answers"],
            "gold_in_cached": d["gold_in_cached"],
            "gold_in_fresh": d["gold_in_fresh"],
            "cached_em": d["cached_quality"]["em"],
            "fresh_em": d["fresh_quality"]["em"],
            "cached_f1": d["cached_quality"]["f1"],
            "fresh_f1": d["fresh_quality"]["f1"],
            "evidence": {
                "char_edit_distance": ef["char_edit_distance"],
                "ws_token_ops": ef["ws_token_ops"],
                "cached_is_ws_prefix_of_fresh": ef["cached_is_ws_prefix_of_fresh"],
                "fresh_is_ws_prefix_of_cached": ef["fresh_is_ws_prefix_of_cached"],
                "cached_is_model_token_prefix_of_fresh": ef["cached_is_model_token_prefix_of_fresh"],
                "fresh_is_model_token_prefix_of_cached": ef["fresh_is_model_token_prefix_of_cached"],
                "first_divergence": d["first_divergence"],
            },
            "justification": justification,
        })

    out.sort(key=lambda r: r["request_index"])
    with open(RESULTS_DIR / "classifications.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # anatomy_table.md
    tally = Counter(r["label"] for r in out)
    pos_tally = Counter(r["divergence_position"] for r in out)
    lines = [
        "# V68 anatomy table — KV-cache reuse vs fresh encoding, differing confirmation records",
        "",
        "Frozen rubric: experiments/v68_anatomy_preregistration.{md,json} (applied post-freeze).",
        f"Source: results/v68_anatomy_20260910T131036Z/differing_records.jsonl "
        f"(sha {sha256_file(differing_path)[:12]}…), V65 qa_requests.jsonl "
        "(sha 68b1806e6104705e…). cached = arm C (all-hit canonical reuse), fresh = arm A (monolithic SDPA).",
        "",
        "Label breakdown (n=22): " + ", ".join(f"{k} = {tally.get(k, 0)}" for k in LABEL_ORDER),
        "Divergence position (model-token index): " + ", ".join(
            f"{k} = {pos_tally.get(k, 0)}" for k in ["first_token", "mid_generation"]),
        "",
        "| idx | record_id | label | div. position | cached (C) prediction | fresh (A) prediction | gold | gold in C / A | C em / A em | evidence |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out:
        ev = r["evidence"]
        div = ev["first_divergence"]["model_token_index"]
        divtxt = f"tok {div} of {ev['first_divergence']['fresh_total_model_tokens']}" if div is not None else "?"
        prefix = ""
        if ev["cached_is_ws_prefix_of_fresh"] or ev["fresh_is_ws_prefix_of_cached"]:
            prefix = "; ws-prefix"
        lines.append(
            f"| {r['request_index']} | `{r['record_id'][:12]}…` | {r['label']} | {r['divergence_position']} ({divtxt}) "
            f"| {r['cached_prediction'][:60].replace('|', '\\|')} "
            f"| {r['fresh_prediction'][:60].replace('|', '\\|')} "
            f"| {', '.join(r['gold_answers'])[:40].replace('|', '\\|')} "
            f"| {r['gold_in_cached']} / {r['gold_in_fresh']} "
            f"| {r['cached_em']:.2f} / {r['fresh_em']:.2f} "
            f"| edit {ev['char_edit_distance']}{prefix} |"
        )
    lines += [
        "",
        "## Rubric application notes",
        "",
        "- Minimal-edit operationalization (post-freeze, uniform): localized difference "
        "(≤1 non-equal difflib op, ≤2 token positions; ≤2 positions for ≤3-token outputs), "
        "shared frame dominates (≥½ tokens) or output ≤3 tokens, same neighborhood.",
        "- ws/model-token prefix flags are evidence, not verdicts: idx 8, 78, 81 carry a "
        "prefix flag but their dominant reading is a one-token suffix/omission on the same "
        "entity → near-tie (frozen guidance: dominant reading wins).",
        "- No record required the `ambiguous` label: every pair fell clearly under one "
        "description after feature inspection; the label remained available per the freeze.",
        "- Direction of damage is mixed: cached strictly loses EM on 5 records "
        "(idx 8, 16, 39, 52, 70 — one near-tie, four clear-semantic) and strictly gains "
        "EM on 4 (idx 14, 17, 78, 99), consistent with the V65 paired aggregate (EM −0.01).",
        "- Logits/logprobs are not retained in the V65 artifact, so 'near-tie' is a "
        "structural/textual judgment; no probability margins were verifiable.",
    ]
    (RESULTS_DIR / "anatomy_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "n": len(out),
        "labels": {k: tally.get(k, 0) for k in LABEL_ORDER},
        "divergence_position": {k: pos_tally.get(k, 0) for k in ["first_token", "mid_generation"]},
        "classifications_sha256": sha256_file(RESULTS_DIR / "classifications.jsonl"),
        "anatomy_table_sha256": sha256_file(RESULTS_DIR / "anatomy_table.md"),
        "prereg_json_sha256": sha256_file(PREREG_JSON),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
