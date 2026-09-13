#!/usr/bin/env python3
"""V68 anatomy of KV-cache reuse output differences (VIBEQUEUE-68), mechanical stage.

Reads the retained V65 combined-validation per-record QA artifact
(results/v65_combined_20260910T102918Z/qa_requests.jsonl) and extracts every
confirmation record whose cached (arm C, all-hit canonical reuse) prediction
differs from the fresh (arm A, monolithic SDPA) prediction.

This script is MECHANICAL ONLY: it computes divergence positions, edit
features and gold-containment flags. It does NOT classify records; the
classification rubric is frozen in experiments/v68_anatomy_preregistration.{md,json}
and applied after the freeze.

Divergence position is computed twice:
  primary   - over the retained model token ids (arm["natural_generated_ids"]);
              assumption: generation tokens counted from 0 = first sampled
              token after the prompt (Qwen2.5 tokenizer ids, ends at <|im_end|>);
              cross-checked against the retained per-record first_token_equal field.
  fallback  - whitespace tokenization of the prediction strings (stated in prereg).

CPU only; stdlib only.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path


def normalize_ws(text: str) -> str:
    """Casefold and collapse/strip whitespace."""
    return " ".join(text.split()).casefold()


def whitespace_tokens(text: str) -> list[str]:
    return text.split()


def first_divergence_index(a: list, b: list):
    """Index of the first position where the sequences differ; None if identical.

    Works for model token ids and whitespace tokens alike.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def is_token_prefix(candidate: list, other: list) -> bool:
    """True if `candidate` is a token-level prefix of `other` (or equal)."""
    if len(candidate) > len(other):
        return False
    return all(x == y for x, y in zip(candidate, other))


def levenshtein(a: str, b: str) -> int:
    """Character-level edit distance (predictions are short; O(len*len) is fine)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def token_ops(a: list, b: list) -> dict:
    """difflib opcodes on token lists -> counts of equal/replace/insert/delete tokens."""
    counts = {"equal": 0, "replace": 0, "insert": 0, "delete": 0}
    n_ops = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            counts["equal"] += i2 - i1
        elif tag == "replace":
            counts["replace"] += max(i2 - i1, j2 - j1)
            n_ops += 1
        elif tag == "insert":
            counts["insert"] += j2 - j1
            n_ops += 1
        elif tag == "delete":
            counts["delete"] += i2 - i1
            n_ops += 1
    counts["n_nonequal_ops"] = n_ops
    return counts


def gold_in_prediction(prediction: str, gold_answers: list[str]) -> bool:
    """Normalized case/whitespace string containment of any gold answer in the prediction."""
    pred = normalize_ws(prediction)
    return any(normalize_ws(g) in pred for g in gold_answers if g)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(qa_requests: Path, cached_arm: str, fresh_arm: str) -> list[dict]:
    records = []
    with open(qa_requests, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    differing = []
    for r in records:
        cached = r[cached_arm]
        fresh = r[fresh_arm]
        cached_pred = cached["prediction"]
        fresh_pred = fresh["prediction"]
        if cached_pred == fresh_pred:
            continue
        if r.get("prediction_equal") is not False:
            raise ValueError(
                f"record {r.get('record_id')}: prediction strings differ but "
                "retained prediction_equal is not False; artifact inconsistency"
            )

        c_ids = cached["natural_generated_ids"]
        f_ids = fresh["natural_generated_ids"]
        c_ws = whitespace_tokens(cached_pred)
        f_ws = whitespace_tokens(fresh_pred)
        div_model = first_divergence_index(c_ids, f_ids)
        div_ws = first_divergence_index(c_ws, f_ws)

        # consistency check against the retained first-token flag
        retained_ft_equal = r.get("first_token_equal")
        computed_ft_equal = (c_ids[0] == f_ids[0]) if c_ids and f_ids else None
        if retained_ft_equal is not None and retained_ft_equal != computed_ft_equal:
            raise ValueError(
                f"record {r.get('record_id')}: retained first_token_equal="
                f"{retained_ft_equal} disagrees with computed {computed_ft_equal}"
            )

        golds = r.get("gold_answers", [])
        differing.append(
            {
                "record_id": r["record_id"],
                "request_id": r.get("request_id"),
                "request_index": r.get("request_index"),
                "cached_arm": cached_arm,
                "fresh_arm": fresh_arm,
                "cached_prediction": cached_pred,
                "fresh_prediction": fresh_pred,
                "gold_answers": golds,
                "cached_prediction_chars": len(cached_pred),
                "fresh_prediction_chars": len(fresh_pred),
                "cached_tokens_ws": len(c_ws),
                "fresh_tokens_ws": len(f_ws),
                "first_divergence": {
                    "model_token_index": div_model,
                    "cached_total_model_tokens": len(c_ids),
                    "fresh_total_model_tokens": len(f_ids),
                    "whitespace_token_index": div_ws,
                    "cached_total_ws_tokens": len(c_ws),
                    "fresh_total_ws_tokens": len(f_ws),
                    "first_token_equal_retained": retained_ft_equal,
                },
                "gold_in_cached": gold_in_prediction(cached_pred, golds),
                "gold_in_fresh": gold_in_prediction(fresh_pred, golds),
                "cached_quality": cached.get("quality"),
                "fresh_quality": fresh.get("quality"),
                "cached_stop_reason": cached.get("stop_reason"),
                "fresh_stop_reason": fresh.get("stop_reason"),
                "edit_features": {
                    "char_edit_distance": levenshtein(cached_pred, fresh_pred),
                    "ws_token_ops": token_ops(c_ws, f_ws),
                    "cached_is_ws_prefix_of_fresh": is_token_prefix(c_ws, f_ws),
                    "fresh_is_ws_prefix_of_cached": is_token_prefix(f_ws, c_ws),
                    "cached_is_model_token_prefix_of_fresh": is_token_prefix(c_ids, f_ids),
                    "fresh_is_model_token_prefix_of_cached": is_token_prefix(f_ids, c_ids),
                },
            }
        )
    differing.sort(key=lambda d: d["request_index"])
    return differing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qa-requests",
        type=Path,
        default=Path("results/v65_combined_20260910T102918Z/qa_requests.jsonl"),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cached-arm", default="C")
    p.add_argument("--fresh-arm", default="A")
    args = p.parse_args(argv)

    records = [json.loads(l) for l in open(args.qa_requests, encoding="utf-8") if l.strip()]
    differing = extract(args.qa_requests, args.cached_arm, args.fresh_arm)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "differing_records.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for d in differing:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    summary = {
        "total_confirmation_records": len(records),
        "differing_records": len(differing),
        "first_token_divergent": sum(
            1 for d in differing if d["first_divergence"]["model_token_index"] == 0
        ),
        "mid_generation_divergent": sum(
            1 for d in differing if d["first_divergence"]["model_token_index"] is not None
            and d["first_divergence"]["model_token_index"] > 0
        ),
        "gold_in_cached": sum(1 for d in differing if d["gold_in_cached"]),
        "gold_in_fresh": sum(1 for d in differing if d["gold_in_fresh"]),
        "input_sha256": sha256_file(args.qa_requests),
        "output_sha256": sha256_file(out_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
