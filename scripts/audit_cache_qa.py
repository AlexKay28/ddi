"""Independently audit V58 matched document-KV QA artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ddi.metrics import answer_presence, exact_match, token_f1
from scripts.validate_cache_qa import file_sha256
from scripts.validate_cache_workload import compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.attempt / "manifest.json").read_text())
    rows = [json.loads(line) for line in (args.attempt / "requests.jsonl").read_text().splitlines()]
    expected_ids = manifest["config"]["record_ids"]
    failures = []
    if [row["record_id"] for row in rows] != expected_ids:
        failures.append("request IDs do not match the frozen manifest order")

    recomputed = []
    for row in rows:
        answers = row["gold_answers"]
        for arm in ("fresh", "cached"):
            prediction = row[arm]["prediction"]
            observed = row[arm]["quality"]
            expected = {
                "em": exact_match(prediction, answers),
                "f1": token_f1(prediction, answers),
                "presence": answer_presence(prediction, answers),
            }
            if observed != expected:
                failures.append(f"{row['record_id']}: {arm} quality mismatch")
        if any(row["cold_hit_state"]) or not all(row["all_hit_state"]):
            failures.append(f"{row['record_id']}: invalid cold/all-hit state")
        if len(row["timing"]["fresh_ms"]) < 5 or len(row["timing"]["cached_ms"]) < 5:
            failures.append(f"{row['record_id']}: insufficient timing repeats")

        logits_path = args.attempt / row["logits_file"]
        if file_sha256(logits_path) != row["logits_sha256"]:
            failures.append(f"{row['record_id']}: logit payload digest mismatch")
        payload = torch.load(logits_path, map_location="cpu", weights_only=True)
        observed_comparison = row["comparison"]
        expected_comparison = compare(payload["fresh"], payload["cached_forced"])
        if observed_comparison != expected_comparison:
            failures.append(f"{row['record_id']}: logit comparison mismatch")
        recomputed.append(expected_comparison)

        cache_dir = args.attempt / "cache" / str(row["record_id"])
        actual_files = sorted(cache_dir.glob("*.pt"))
        if [path.name for path in actual_files] != sorted(row["cache_files"]):
            failures.append(f"{row['record_id']}: cache file set mismatch")
        for path in actual_files:
            if file_sha256(path) != row["cache_files"][path.name]:
                failures.append(f"{row['record_id']}: cache digest mismatch for {path.name}")

    audit = {
        "pass": not failures,
        "failures": failures,
        "record_count": len(rows),
        "finite": all(item["finite"] for item in recomputed),
        "numerical_failures": sum(not item["numerical_pass"] for item in recomputed),
        "max_abs": max(item["max_abs"] for item in recomputed),
    }
    (args.attempt / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
