"""Independently audit a V57 matched-QA run and render its bounded trade-off plot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from ddi.metrics import answer_presence, exact_match, token_f1


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def paired_interval(values: list[float], *, seed: int = 57) -> dict:
    if not values:
        raise ValueError("paired values are required")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000)
    for i in range(len(draws)):
        draws[i] = array[rng.integers(0, len(array), len(array))].mean()
    return {
        "per_request": values,
        "mean": float(array.mean()),
        "ci95_paired_bootstrap": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def _svg(rows: list[dict]) -> str:
    width, height = 900, 420
    colors = {"full": "#315c8c", "ddi_qf": "#c44e2b"}
    points = []
    for row in rows:
        x = 70 + 330 * row["actual_sparsity"]
        y = 350 - 280 * row["quality"]["f1"]
        points.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" '
            f'fill="{colors.get(row["variant"], "#555")}" fill-opacity="0.45"/>'
        )
    by_request = {}
    for row in rows:
        by_request.setdefault(row["request_id"], {})[row["variant"]] = row
    tradeoff_points = []
    for arms in by_request.values():
        if "full" not in arms or "ddi_qf" not in arms:
            continue
        speedup = arms["full"]["fixed"]["median_request_ms"] / arms["ddi_qf"]["fixed"][
            "median_request_ms"
        ]
        quality_delta = arms["full"]["quality"]["f1"] - arms["ddi_qf"]["quality"]["f1"]
        x = 520 + 280 * min(max((speedup - 0.9) / 0.2, 0), 1)
        y = 350 - 280 * min(max(quality_delta, 0), 1)
        tradeoff_points.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#6a3d9a" fill-opacity="0.45"/>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fbf8f1"/>
<g font-family="sans-serif" fill="#222">
<text x="70" y="28" font-size="18">Quality vs actual causal-pair sparsity</text>
<text x="500" y="28" font-size="18">F1 loss vs measured request ratio</text>
<path d="M70 350H400M70 70V350M500 350H830M500 70V350" stroke="#333" fill="none"/>
<text x="190" y="390" font-size="13">actual sparsity</text>
<text x="600" y="390" font-size="13">full / DDI-QF latency</text>
<text x="16" y="210" font-size="13" transform="rotate(-90 16 210)">F1</text>
<text x="450" y="235" font-size="13" transform="rotate(-90 450 235)">full - DDI-QF F1</text>
{''.join(points)}
{''.join(tradeoff_points)}
<circle cx="90" cy="405" r="4" fill="#315c8c"/><text x="100" y="410" font-size="12">full</text>
<circle cx="150" cy="405" r="4" fill="#c44e2b"/><text x="160" y="410" font-size="12">DDI-QF</text>
</g></svg>"""


def audit(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    config = manifest["config"]
    expected_rows = len(config["record_ids"]) * len(config["variants"])
    assert len(rows) == expected_rows
    assert digest((root / "working.patch").read_bytes()) == manifest["diff_sha256"]
    for name, expected in manifest["source_hashes"].items():
        assert digest((root / "source" / name).read_bytes()) == expected
    grouped = {}
    for row in rows:
        grouped.setdefault(row["request_id"], {})[row["variant"]] = row
        ids_hash = digest(np.asarray(row["input_ids"], dtype=np.int64).tobytes())
        assert ids_hash == row["input_ids_sha256"]
        fixed = row["fixed"]
        assert len(fixed["raw_prefill_ms"]) == len(fixed["raw_request_ms"]) == config["repeats"]
        assert fixed["generated_count"] == config["fixed_new_tokens"]
        assert fixed["decode_forwards"] == config["fixed_new_tokens"] - 1
        assert math.isclose(fixed["median_prefill_ms"], statistics.median(fixed["raw_prefill_ms"]))
        assert math.isclose(fixed["median_request_ms"], statistics.median(fixed["raw_request_ms"]))
        prediction = row["natural"]["prediction"]
        gold = row["gold_answers"]
        expected_quality = {
            "em": exact_match(prediction, gold),
            "f1": token_f1(prediction, gold),
            "presence": answer_presence(prediction, gold),
        }
        assert row["quality"] == expected_quality
    required = set(config["variants"])
    assert all(set(arms) == required for arms in grouped.values())
    contrasts = {}
    if required.issuperset({"full", "ddi_qf"}):
        for metric in ("em", "f1", "presence"):
            contrasts[f"full_minus_ddi_qf_{metric}"] = paired_interval(
                [arms["full"]["quality"][metric] - arms["ddi_qf"]["quality"][metric]
                 for arms in grouped.values()]
            )
        for scope, field in (("prefill", "median_prefill_ms"), ("request", "median_request_ms")):
            interval = paired_interval(
                [arms["full"]["fixed"][field] / arms["ddi_qf"]["fixed"][field]
                 for arms in grouped.values()]
            )
            interval["material_speedup_pass"] = interval["ci95_paired_bootstrap"][0] > 1.05
            contrasts[f"ddi_qf_{scope}_speedup"] = interval
    return {
        "artifact_integrity_pass": True,
        "request_count": len(grouped),
        "row_count": len(rows),
        "contrasts": contrasts,
        "original_summary_sha256": digest((root / "summary.json").read_bytes()),
        "requests_sha256": digest((root / "requests.jsonl").read_bytes()),
        "postprocessor_sha256": digest(Path(__file__).read_bytes()),
        "source_sha256": manifest["source_sha256"],
        "diff_sha256": manifest["diff_sha256"],
        "admissible_paper_evidence": summary["admissible_paper_evidence"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    (args.output / "post_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = [json.loads(line) for line in (args.output / "requests.jsonl").read_text().splitlines()]
    (args.output / "tradeoffs.svg").write_text(_svg(rows))
    (args.output / "post_audit_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
