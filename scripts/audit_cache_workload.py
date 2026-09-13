"""Verify completed V53/V57 cache artifacts and recompute retained correctness fields."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _token_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    timing = [json.loads(line) for line in (root / "timing.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in
                  (root / "validation.jsonl").read_text().splitlines()]
    config = manifest["config"]
    seeds = config["seeds"]
    request_count = config["warmups"] + config["measured"]
    assert len(timing) == len(seeds) * len(["ddi_cache", "ddi_recompute", "full_causal"]) \
        * request_count
    assert len(validation) in {len(seeds) * (4 + request_count), len(seeds) * (5 + request_count)}
    identity_manifest = dict(manifest)
    stored_identity = identity_manifest.pop("cache_identity")
    assert _digest_json(identity_manifest) == stored_identity
    for name, expected in manifest["source_hashes"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((root / "working.patch").read_bytes()).hexdigest() \
        == manifest["diff_sha256"]
    hit_rates, warmup_hits, natural_lengths = [], [], []
    timing_behavior_disagreements = []
    valid_token_digests = set()
    for seed_index, seed in enumerate(seeds):
        inputs = json.loads((root / f"inputs_seed{seed}.json").read_text())
        assert list(map(len, inputs["docs"])) == [config["L"]] * config["top"]
        assert len(inputs["query"]) == config["Q"]
        valid_token_digests.update(_token_digest(ids) for ids in inputs["docs"])
        rows = [r for r in timing if r["seed"] == seed]
        modes = ["ddi_cache", "ddi_recompute", "full_causal"]
        order = modes[seed_index:] + modes[:seed_index]
        assert [r["mode"] for r in rows] == [
            mode for mode in order for _ in range(request_count)
        ]
        assert all(row["order"] == order for row in rows)
        for mode in modes:
            selected = [r for r in rows if r["mode"] == mode]
            assert len(selected) == request_count
            measured = selected[config["warmups"]:]
            qps = config["measured"] * 1000 / sum(row["elapsed_ms"] for row in measured)
            summary_index = seeds.index(seed)
            assert math.isclose(qps, summary["qps"][mode][summary_index], rel_tol=1e-12)
            seen = set()
            for i, row in enumerate(selected):
                assert row["indices"] == inputs["stream"][i]
                assert row["request"] == i and row["warmup"] == (i < config["warmups"])
                assert row["doc_lengths"] == [config["L"]] * config["N"]
                assert row["query_len"] == config["Q"]
                assert row["generated_count"] == len(row["generated_ids"]) == config["A"]
                assert row["decode_forwards"] == config["A"] - 1 and row["elapsed_ms"] > 0
                if mode == "ddi_cache":
                    assert row["hit_state"] == [index in seen for index in row["indices"]]
                    seen.update(row["indices"])
                else:
                    assert row["hit_state"] == []
                if mode != "full_causal":
                    ref = next(r for r in validation if r["seed"] == seed
                               and r["case"] == f"stream_{i}")
                    expected = ref["cache" if mode == "ddi_cache" else "fresh"]
                    if row["generated_ids"] != expected["generated_ids"]:
                        timing_behavior_disagreements.append([seed, mode, i])
        cached = [r for r in rows if r["mode"] == "ddi_cache"]
        hit_rates.append(
            sum(sum(r["hit_state"]) for r in cached[config["warmups"]:])
            / (config["measured"] * config["N"])
        )
        warmup_hits.append(sum(sum(r["hit_state"]) for r in cached[:config["warmups"]]))
        for name, hits in [
            ("cold", 0),
            ("warm", config["N"]),
            ("permuted", config["N"]),
            ("partial", config["N"] - 1),
        ]:
            row = next(r for r in validation if r["seed"] == seed and r["case"] == name)
            assert sum(row["cache"]["hit_state"]) == hits
    for row in validation:
        for mode in ["fresh", "cache"]:
            natural_lengths.append(row[f"natural_{mode}"]["generated_count"])
        logits_path = root / f"logits_{row['seed']}_{row['case']}.pt"
        payload = torch.load(logits_path, map_location="cpu", weights_only=False)
        fresh = payload["fresh"].float()
        forced = payload["cache_forced"].float()
        delta = (fresh - forced).abs()
        comparison = row["comparison"]
        assert comparison["finite"] == bool(
            torch.isfinite(fresh).all() and torch.isfinite(forced).all()
        )
        assert math.isclose(comparison["max_abs"], float(delta.max()), rel_tol=0, abs_tol=1e-12)
        assert math.isclose(comparison["mean_abs"], float(delta.mean()), rel_tol=0, abs_tol=1e-12)
        assert comparison["numerical_pass"] == bool(
            torch.allclose(fresh, forced, atol=config["atol"], rtol=config["rtol"])
        )
        assert comparison["argmax_agreement"] == (
            fresh.argmax(-1) == forced.argmax(-1)
        ).tolist()
        if "expected_hit_state" in row:
            assert row["hit_state_pass"] == (row["cache"]["hit_state"] == row["expected_hit_state"])
    for cache_path in root.glob("*_cache_*/*.pt"):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        assert meta["key"] == cache_path.stem
        assert meta["cache_identity"] == stored_identity
        assert meta["token_ids_sha256"] in valid_token_digests
        assert meta["schema_version"] == 2
    failures = [{"seed": r["seed"], "case": r["case"],
                 "greedy_equal": r["greedy_equal"], "natural_equal": r["natural_equal"],
                 "first_argmax_mismatch": next((i for i, ok in
                     enumerate(r["comparison"]["argmax_agreement"]) if not ok), None)}
                for r in validation if not r["greedy_equal"] or not r["natural_equal"]]
    assert summary["greedy_failures"] == sum(not r["greedy_equal"] for r in validation)
    assert summary["natural_failures"] == sum(not r["natural_equal"] for r in validation)
    assert summary["numerical_failures"] == sum(not r["comparison"]["numerical_pass"]
                                                for r in validation)
    expected_artifacts = ["manifest.json", "summary.json", "timing.jsonl", "validation.jsonl"]
    expected_artifacts.extend(f"inputs_seed{seed}.json" for seed in seeds)
    return dict(artifact_integrity_pass=True, correctness_pass=summary["correctness_pass"],
                hit_rates=hit_rates, warmup_hits=warmup_hits,
                 artifact_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in expected_artifacts},
                postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                base_sha=manifest["base_sha"], source_sha256=manifest["source_sha256"],
                diff_sha256=manifest["diff_sha256"], cache_identity=manifest["cache_identity"],
                natural_min_tokens=min(natural_lengths), natural_max_tokens=max(natural_lengths),
                 natural_early_stops=sum(n < config["A"] for n in natural_lengths),
                timing_vs_validation_disagreements=timing_behavior_disagreements,
                behavior_failures=failures, summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    (args.output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
