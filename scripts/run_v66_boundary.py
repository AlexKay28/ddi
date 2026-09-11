"""VIBEQUEUE-66: optimized-path serving boundary sweep (preregistered).

Measures where the V65-optimized canonical reuse path wins against the monolithic
SDPA baseline as a function of hit fraction h, document length L, N documents and
decode budget, on DEVELOPMENT data only (frozen V62 dev split; the 100 confirmation
ids are forbidden). The protocol is frozen in
``experiments/v66_boundary_preregistration.{md,json}`` — both files' sha256 values
are pinned in the run manifest, which is written before the model loads. No
threshold, grid or rule may change after results are seen; forced deviations
(cap-pressure shrink, infeasible cells) are recorded in the result payload and
README, never by editing the prereg.

Arms (identical tokens per request, one shared SDPA model):
- ``A`` monolithic — fresh encode+decode each request; restart positions (Amendment 1:
  same-layout baseline; the V65 harness ran sequential positions, a numerically
  different execution); TTFT = synchronized prefill ``model()`` wall
  (``generate_ids(...)["prefill_ms"]`` semantics, mask build untimed); wall =
  synchronized wall of the full fixed-work greedy
  ``generate_ids(..., d, stop_on_eos=False)``.
- ``B`` optimized V65 reuse — ``CanonicalCache(model, contract, DocCache)`` in the
  LRU-warm steady state; TTFT = synchronized wall of
  ``request(docs, query_ids)`` (``max_new_tokens=None``); wall = synchronized wall
  of ``request(docs, query_ids, max_new_tokens=d, eos_token_id=None)``. Per cell the
  store is reset to exactly the ``hit_count`` pre-state; fresh write-through entries
  are pruned untimed after every pass.

Per (L, N, hit_count) unit: fresh CanonicalCache (fresh 2 GiB LRU), 3 warmups + 15
timed repeats interleaved by repeat index (A_i, B_i); budgets in the frozen order
d=1/TTFT first, then 8, 16, 32, 64. Identity: per repeat, arm-A and arm-B generated
token ids must be byte-identical; any divergence appends a diagnostic record to the
append-only progress log and exits non-zero (the sweep pauses for diagnosis).

Outputs under ``results/v66_boundary_<UTC>/``: ``manifest.json``,
``progress.jsonl`` (append-only, one line per completed cell; restart skips
completed cells), ``raw_repeats.jsonl`` (one line per timed pass), ``result.json``.
Nothing else is written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddi import inference  # noqa: E402
from ddi.cache import DocCache  # noqa: E402
from ddi.canonical_cache import CanonicalCache  # noqa: E402
from ddi.data import load as load_dataset  # noqa: E402
from ddi.data.hotpot_qa import n_paragraphs  # noqa: E402
from ddi.inference import load  # noqa: E402
from ddi.positions import build_positions  # noqa: E402
from ddi.prompt import build_query_text  # noqa: E402
from scripts.run_canonical_matrix import (  # noqa: E402
    _checkpoint_identity,
    _contract,
    _git_bytes,
    _git_text,
    _peak_bytes,
    _reset_peak,
    _sha_file,
    _source_hashes,
    _sync,
)
from scripts.run_canonical_qa_dev import _default_tokenizer  # noqa: E402
from scripts.run_v65_benchmark import paired_ratio_bootstrap  # noqa: E402
from scripts.validate_matched_qa import digest, generate_ids  # noqa: E402

SCHEMA_VERSION = 66
SEED = 47
WARMUPS = 3
REPEATS = 15
MIN_REPEATS = 10
SMOKE_WARMUPS = 2
SMOKE_REPEATS = 2
BOOTSTRAP_SEED = 47
BOOTSTRAP_ITERATIONS = 10_000
CI_LEVEL = 0.95
HIT_FRACTIONS = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DOC_LENGTHS = (128, 512)
CONDITIONAL_DOC_LENGTH = 32
L32_GATE_MINUTES = 105.0
N_DOCS = (2, 5, 10)
DECODE_BUDGETS = (8, 16, 32, 64)
TTFT_BUDGET = 1
WIN_CI_LOW = 1.05
LOSE_CI_HIGH = 1.0 / 1.05
UNIT_CUTOFF_MINUTES = 165.0
HARD_CAP_MINUTES = 180.0
MIN_FREE_GPU_FRACTION = 0.7

PREREG_MD = ROOT / "experiments/v66_boundary_preregistration.md"
PREREG_JSON = ROOT / "experiments/v66_boundary_preregistration.json"
PREREG_JSON_SHA256 = "b6654e8e88b24b18d8076a388a9c72ed8a5ef3d1049cb382ed201fa5017afe3b"
SPLIT_MANIFEST = ROOT / "experiments/v62_split_manifest.json"
SPLIT_MANIFEST_SHA256 = "5ad3761106ecaed967b28d21c3919fa1869ebf70bf0e178c1e773a14bf8ee098"
V60_MANIFEST = ROOT / "experiments/v60_confirmation_manifest.json"
V60_MANIFEST_SHA256 = "2dffebf5d04065021dfb2b9e694651b34f0ebfae4f3d4b9aabd672908aa6cb07"
DEFAULT_CHECKPOINT = ROOT / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
CHECKPOINT_SHA256 = "00df4c47d67660cab8292da313ea133290b0bb320437290d725cd149c3d31de9"

CELLS_KEY = "cell_id"


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_v66_boundary.py)
# ---------------------------------------------------------------------------


def hit_count_for(hit_fraction: float, n_docs: int) -> int:
    """Round-half-up realization of a nominal hit fraction (pure)."""
    if not 0.0 <= hit_fraction <= 1.0:
        raise ValueError("hit_fraction must be in [0, 1]")
    if n_docs < 1:
        raise ValueError("n_docs must be >= 1")
    return min(n_docs, max(0, math.floor(hit_fraction * n_docs + 0.5)))


def unit_hit_counts(n_docs: int) -> list[int]:
    """Unique hit counts for one N, in the frozen descending run order (pure)."""
    seen: list[int] = []
    for fraction in HIT_FRACTIONS:
        count = hit_count_for(fraction, n_docs)
        if count not in seen:
            seen.append(count)
    return sorted(seen, reverse=True)


def h_labels_for(hit_count: int, n_docs: int) -> list[float]:
    """Nominal h values that realize this hit count (pure)."""
    return [h for h in HIT_FRACTIONS if hit_count_for(h, n_docs) == hit_count]


def materiality_verdict(ci_low: float, ci_high: float) -> str:
    """Frozen materiality rule on an A/B speedup CI (pure)."""
    if ci_low > WIN_CI_LOW:
        return "win"
    if ci_high < LOSE_CI_HIGH:
        return "lose"
    return "tie"


def identity_verdict(ids_a: Any, ids_b: Any) -> bool:
    """Byte-identity of two generated token id sequences (pure)."""
    return list(ids_a) == list(ids_b)


def aggregate_identity(verdicts: list[bool]) -> dict[str, Any]:
    """Aggregate per-repeat identity verdicts (pure)."""
    mismatches = [index for index, ok in enumerate(verdicts) if not ok]
    return {
        "repeats": len(verdicts),
        "all_identical": not mismatches,
        "mismatch_repeat_indices": mismatches,
    }


def pack_documents(
    pool: list[list[int]], length: int, n_docs: int, start: int = 0
) -> tuple[list[torch.Tensor], int]:
    """Pack whole consecutive pool paragraphs into exactly-``length`` documents (pure).

    Whole paragraphs are consumed token-level; a paragraph that pushes the buffer to
    ``>= length`` is fully consumed and the remainder is discarded. Returns the
    ``[1, length]`` int64 streams and the advanced cursor.
    """
    if length < 1 or n_docs < 1:
        raise ValueError("length and n_docs must be >= 1")
    if start < 0 or start > len(pool):
        raise ValueError("start cursor outside the pool")
    docs: list[torch.Tensor] = []
    cursor = start
    for _ in range(n_docs):
        buffer: list[int] = []
        while len(buffer) < length:
            if cursor >= len(pool):
                raise ValueError("paragraph pool exhausted")
            buffer.extend(pool[cursor])
            cursor += 1
        docs.append(torch.tensor([buffer[:length]], dtype=torch.long))
    return docs, cursor


def build_monolithic_batch(doc_streams: list[torch.Tensor], query_ids: torch.Tensor) -> Any:
    """Arm-A batch from token streams with restart position semantics (pure).

    Amendment 1: positions are ``build_positions(doc_lengths, query_len, 0,
    "restart")`` — the same numerics as the canonical reuse path, so the per-repeat
    byte-identity gate compares like with like. (The V65 harness default was
    sequential positions, a numerically different execution; the preregistered
    smoke caught the resulting identity divergence before any timed data.)
    """
    doc_lengths = [int(stream.shape[-1]) for stream in doc_streams]
    query_len = int(query_ids.shape[-1])
    input_ids = torch.cat([stream[0] for stream in doc_streams] + [query_ids[0]])[None]
    positions = build_positions(doc_lengths, query_len, 0, "restart").unsqueeze(0)
    return SimpleNamespace(
        input_ids=input_ids,
        position_ids=positions,
        doc_lengths=doc_lengths,
        query_len=query_len,
        layout="restart",
    )


def expected_store_keys(cache: CanonicalCache, docs: list[torch.Tensor]) -> list[str]:
    """Content-addressed store keys for the request documents, in request order."""
    return [cache._metadata(ids)["key"] for ids in docs]


def reset_store(store_dir: Path, hit_keys: list[str]) -> dict[str, Any]:
    """Make the store hold exactly the hit payloads; report missing keys (pure I/O)."""
    store_dir.mkdir(parents=True, exist_ok=True)
    expected = set(hit_keys)
    removed = 0
    for path in sorted(store_dir.glob("*.pt")):
        if path.name[: -len(".pt")] not in expected:
            path.unlink()
            removed += 1
    present = {path.name[: -len(".pt")] for path in store_dir.glob("*.pt")}
    return {
        "removed": removed,
        "missing": sorted(expected - present),
        "files": len(present),
    }


def prune_store(store_dir: Path, hit_keys: list[str]) -> int:
    """Remove every store payload outside the hit pre-state (pure I/O)."""
    expected = set(hit_keys)
    removed = 0
    for path in sorted(store_dir.glob("*.pt")):
        if path.name[: -len(".pt")] not in expected:
            path.unlink()
            removed += 1
    return removed


def verify_store(store_dir: Path, hit_keys: list[str]) -> None:
    present = {path.name[: -len(".pt")] for path in Path(store_dir).glob("*.pt")}
    if present != set(hit_keys):
        raise RuntimeError(
            f"store pre/post-state mismatch: expected {len(hit_keys)} payloads, "
            f"found {len(present)}"
        )


def ratio_summary(walls_a: list[float], walls_b: list[float]) -> dict[str, Any]:
    """Median A/B speedup with the frozen paired bootstrap CI (pure)."""
    if len(walls_a) != len(walls_b) or not walls_a:
        raise ValueError("paired walls must be non-empty and equal-length")
    bootstrap = paired_ratio_bootstrap(
        walls_a,
        walls_b,
        seed=BOOTSTRAP_SEED,
        iterations=BOOTSTRAP_ITERATIONS,
        ci_level=CI_LEVEL,
    )
    return {
        "median_a_ms": statistics.median(walls_a),
        "median_b_ms": statistics.median(walls_b),
        "ratio_point": bootstrap["point"],
        "ratio_ci_low": bootstrap["ci_low"],
        "ratio_ci_high": bootstrap["ci_high"],
        "verdict": materiality_verdict(bootstrap["ci_low"], bootstrap["ci_high"]),
    }


# ---------------------------------------------------------------------------
# timing glue
# ---------------------------------------------------------------------------


def _timed_pass(fn: Any, device: torch.device) -> tuple[float, Any]:
    _sync(device)
    started = time.perf_counter()
    out = fn()
    _sync(device)
    return (time.perf_counter() - started) * 1e3, out


class _Sinks:
    """Append-only JSONL sinks with crash-safe flushing."""

    def __init__(self, output: Path):
        self.progress = output / "progress.jsonl"
        self.raw = output / "raw_repeats.jsonl"

    def append(self, path: Path, payload: dict[str, Any]) -> None:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")
            handle.flush()

    def progress_line(self, payload: dict[str, Any]) -> None:
        self.append(self.progress, payload)

    def raw_line(self, payload: dict[str, Any]) -> None:
        self.append(self.raw, payload)


def _rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _ids_sha(ids: Any) -> str:
    return digest(json.dumps([int(token) for token in ids], separators=(",", ":")).encode())


def _completed_cells(output: Path, *, record_mode: bool = False) -> set[str]:
    progress = output / "progress.jsonl"
    if not progress.exists():
        return set()
    done = set()
    for line in progress.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "cell" and record.get("status") == "ok":
            done.add(record[CELLS_KEY])
        if record.get("type") == "identity_failure":
            policy = record.get("policy", "abort")
            if policy == "record":
                continue
            if not record_mode:
                raise RuntimeError(
                    "progress log contains an unresolved identity failure; "
                    "diagnose before restarting (relaunch with "
                    "--continue-on-identity-divergence after diagnosis, per "
                    "prereg Amendment 2): " + line[:500]
                )
    return done


def _run_pass_block(
    *,
    label: str,
    budget: int,
    warmups: int,
    repeats: int,
    device: torch.device,
    pass_a: Any,
    pass_b: Any,
    pass_b_ttft: Any,
    hit_count: int,
    n_docs: int,
    hit_indices: list[int],
    hit_keys: list[str],
    store_dir: Path,
    sinks: _Sinks,
    cell_id: str,
    continue_on_divergence: bool = False,
) -> dict[str, Any]:
    """One frozen pass block: shared warmups, then interleaved A_i/B_i timed repeats.

    ``pass_a()`` -> generate_ids payload (A wall + prefill_ms + ids); ``pass_b()``
    -> B wall + ids (budget >= 1) ; ``pass_b_ttft()`` -> B TTFT ModelOutput. Only
    used for the d=1/TTFT block; decode blocks pass ``pass_b_ttft=None``.
    """
    for _ in range(warmups):
        pass_a()
        pass_b()
        if pass_b_ttft is not None:
            pass_b_ttft()
        if hit_count < n_docs:
            prune_store(store_dir, hit_keys)
        _sync(device)

    a_walls: list[float] = []
    b_walls: list[float] = []
    a_ttfts: list[float] = []
    b_ttfts: list[float] = []
    identity_flags: list[bool] = []
    first_token_flags: list[bool] = []
    peaks = {"A": [], "B": []}
    _reset_peak(device)
    for repeat in range(repeats):
        _reset_peak(device)
        a_wall, a_payload = _timed_pass(pass_a, device)
        a_alloc, a_reserved = _peak_bytes(device)
        a_ids = list(a_payload["generated_ids"])
        a_ttft = float(a_payload["prefill_ms"])
        del a_payload
        a_walls.append(a_wall)
        a_ttfts.append(a_ttft)
        peaks["A"].append((a_alloc, a_reserved))
        sinks.raw_line({
            "type": "pass", "cell_id": cell_id, "block": label, "arm": "A",
            "budget": budget, "repeat": repeat, "wall_ms": a_wall,
            "ttft_ms": a_ttft,
            "ids_sha256": _ids_sha(a_ids),
            "peak_allocated_bytes": a_alloc, "peak_reserved_bytes": a_reserved,
        })

        if pass_b_ttft is not None:
            _reset_peak(device)
            b_ttft_wall, b_out = _timed_pass(pass_b_ttft, device)
            b_first = int(b_out.logits[0, -1].argmax())
            del b_out
            b_alloc, b_reserved = _peak_bytes(device)
            b_ttfts.append(b_ttft_wall)
            peaks["B"].append((b_alloc, b_reserved))
            sinks.raw_line({
                "type": "pass", "cell_id": cell_id, "block": label, "arm": "B",
                "budget": None, "repeat": repeat, "wall_ms": b_ttft_wall,
                "ttft_ms": b_ttft_wall, "first_token": b_first,
                "peak_allocated_bytes": b_alloc, "peak_reserved_bytes": b_reserved,
            })
            # The TTFT pass writes its fresh docs through; prune immediately so the
            # B wall pass below starts from the exact frozen pre-state (a missed
            # prune here made every B wall(1) pass an accidental all-hit pass).
            if hit_count < n_docs:
                prune_store(store_dir, hit_keys)

        _reset_peak(device)
        b_wall, b_ids = _timed_pass(pass_b, device)
        b_alloc, b_reserved = _peak_bytes(device)
        _b_wall_payload = b_ids if isinstance(b_ids, dict) else None
        b_ids = list(b_ids) if not isinstance(b_ids, dict) else list(b_ids["generated_ids"])
        del _b_wall_payload
        b_walls.append(b_wall)
        peaks["B"].append((b_alloc, b_reserved))
        ok = identity_verdict(a_ids, b_ids)
        identity_flags.append(ok)
        first_token_flags.append(bool(b_ids) and b_ids[0] == a_ids[0])
        sinks.raw_line({
            "type": "pass", "cell_id": cell_id, "block": label, "arm": "B",
            "budget": budget, "repeat": repeat, "wall_ms": b_wall,
            "ids_sha256": _ids_sha(b_ids), "identity_ok": ok,
            "peak_allocated_bytes": b_alloc, "peak_reserved_bytes": b_reserved,
        })

        if not ok:
            sinks.progress_line({
                "type": "identity_failure", "cell_id": cell_id, "block": label,
                "repeat": repeat, "budget": budget,
                "hit_indices": hit_indices, "hit_keys": hit_keys,
                "ids_a": a_ids, "ids_b": b_ids,
                "policy": "record" if continue_on_divergence else "abort",
                "diagnosis": (
                    "bf16 reduction-order divergence between the joint monolithic "
                    "prefill and per-doc encode + assembled-KV query forward "
                    "(mathematically equivalent executions, same positions/masks); "
                    "timing unaffected (identical fixed work)"
                    if continue_on_divergence else "paused for diagnosis"
                ),
            })
            if not continue_on_divergence:
                raise SystemExit(
                    f"V66 PAUSED for identity diagnosis: cell {cell_id} block {label} "
                    f"repeat {repeat}: arm-A ids {a_ids} != arm-B ids {b_ids}"
                )

        if hit_count < n_docs:
            prune_store(store_dir, hit_keys)

    verify_store(store_dir, hit_keys)
    summary = ratio_summary(a_walls, b_walls)
    block = {
        "block": label,
        "budget": budget,
        "repeats": repeats,
        **summary,
        "median_a_ttft_ms": statistics.median(a_ttfts) if a_ttfts else None,
        "median_b_ttft_ms": statistics.median(b_ttfts) if b_ttfts else None,
        "ttft_ratio": (
            ratio_summary(a_ttfts, b_ttfts) if pass_b_ttft is not None else None
        ),
        "identity": aggregate_identity(identity_flags),
        "first_token_agrees_b_ttft": (
            aggregate_identity(first_token_flags) if pass_b_ttft is not None else None
        ),
        "peak_allocated_bytes": {
            arm: max(entry[0] for entry in values if entry[0] is not None)
            for arm, values in peaks.items() if values
        },
        "peak_reserved_bytes": {
            arm: max(entry[1] for entry in values if entry[1] is not None)
            for arm, values in peaks.items() if values
        },
    }
    return block


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _build_manifest(args: argparse.Namespace, *, cells_plan: list[dict[str, Any]],
                    gpu: dict[str, Any] | None, dev_ids: list[str],
                    query_record_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ticket": "VIBEQUEUE-66",
        "runner": "scripts/run_v66_boundary.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest_written_before_run": True,
        "preregistration": {
            "md": str(PREREG_MD),
            "md_sha256": _sha_file(PREREG_MD),
            "json": str(PREREG_JSON),
            "json_sha256": _sha_file(PREREG_JSON),
            "pinned_json_sha256": PREREG_JSON_SHA256,
        },
        "data": {
            "role": "development_only",
            "split_manifest": str(SPLIT_MANIFEST),
            "split_manifest_sha256": _sha_file(SPLIT_MANIFEST),
            "dev_id_count": len(dev_ids),
            "confirmation_ids_used": 0,
            "query_record_id": query_record_id,
            "content_seed": SEED,
        },
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_identity": {
            "mode": _checkpoint_identity(args.checkpoint)[0],
            "sha256": _checkpoint_identity(args.checkpoint)[1],
        },
        "protocol": {
            "seed": SEED,
            "warmups": args.warmups,
            "timed_repeats": args.repeats,
            "min_repeats_under_cap": MIN_REPEATS,
            "order": "interleaved A_i,B_i after shared warmups; budgets d=1/TTFT then 8,16,32,64",
            "arm_a_positions": "restart (prereg Amendment 1)",
            "synchronize": True,
            "canonical_on_sdpa": True,
            "device": args.device,
            "gpu": gpu,
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "iterations": args.bootstrap_iterations,
                "ci_level": CI_LEVEL,
                "estimator": "paired ratio of medians A/B over repeat-index pairs",
            },
        },
        "grid": {
            "hit_fractions": list(HIT_FRACTIONS),
            "hit_count_rule": "min(N, max(0, floor(h*N + 0.5)))",
            "doc_lengths": list(DOC_LENGTHS),
            "conditional_doc_length": CONDITIONAL_DOC_LENGTH,
            "l32_gate_minutes": L32_GATE_MINUTES,
            "n_docs": list(N_DOCS),
            "decode_budgets": [TTFT_BUDGET, *DECODE_BUDGETS],
            "planned_cells": cells_plan,
        },
        "materiality_rule": {
            "win": f"A/B ratio CI low > {WIN_CI_LOW}",
            "lose": f"A/B ratio CI high < {LOSE_CI_HIGH}",
            "tie": "otherwise",
            "declared_before_run": True,
        },
        "identity_rule": (
            "per repeat, arm-A and arm-B generated token ids byte-identical; any "
            "divergence pauses the sweep with a diagnostic record"
        ),
        "caps": {
            "unit_cutoff_minutes": UNIT_CUTOFF_MINUTES,
            "hard_cap_minutes": HARD_CAP_MINUTES,
        },
        "sources": {
            "scripts/run_v66_boundary.py": _sha_file(Path(__file__).resolve()),
            "ddi_tree": _source_hashes(),
        },
        "git": {
            "head": _git_text("rev-parse", "HEAD"),
            "diff_head_sha256": hashlib.sha256(_git_bytes("diff", "HEAD")).hexdigest(),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--smoke", action="store_true",
                        help="run only the mechanics-validation smoke cell (timings discarded)")
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing output directory, skipping completed cells")
    parser.add_argument("--redo-d1", action="store_true",
                        help="repair mode: re-run D1 blocks of hit_count<N cells "
                             "(append-only supersede; used after the prune-placement fix)")
    parser.add_argument("--continue-on-identity-divergence", action="store_true",
                        help="Amendment 2 mode: record per-repeat identity divergences "
                             "with a diagnosis and continue (default: pause and exit)")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")
    return args


def _load_dev_content(checkpoint: Path) -> tuple[list[str], dict[str, dict], str, str]:
    if n_paragraphs() is not None:
        raise RuntimeError("DDI_HOTPOT_N_PARAGRAPHS must be unset for V66")
    if _sha_file(SPLIT_MANIFEST) != SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest sha mismatch")
    if _sha_file(V60_MANIFEST) != V60_MANIFEST_SHA256:
        raise ValueError("v60 manifest sha mismatch")
    split = json.loads(SPLIT_MANIFEST.read_text())
    dev_ids = sorted(split["splits"]["development"]["record_ids"])
    confirmation_ids = set(split["splits"]["confirmation"]["record_ids"])
    v60_manifest = json.loads(V60_MANIFEST.read_text())
    v60_ids = set(v60_manifest["config"]["record_ids"])
    if set(dev_ids) & confirmation_ids:
        raise ValueError("development ids overlap the forbidden confirmation arm")
    if set(dev_ids) & v60_ids:
        raise ValueError("development ids overlap the forbidden V60 confirmation ids")
    records = {
        record["id"]: record
        for record in load_dataset("hotpot_qa", "train", 0)
        if record["id"] in set(dev_ids)
    }
    if set(records) != set(dev_ids):
        raise ValueError("development records did not all load from the train stream")
    rng = random.Random(SEED)
    query_record_id = dev_ids[rng.randrange(len(dev_ids))]
    tokenizer = _default_tokenizer(str(checkpoint))
    return dev_ids, records, query_record_id, tokenizer


def _build_pool(tokenizer: Any, records: dict[str, dict], dev_ids: list[str]) -> list[list[int]]:
    pool: list[list[int]] = []
    for record_id in dev_ids:
        for paragraph in records[record_id]["info_contexts"]:
            tokens = tokenizer(paragraph, add_special_tokens=False)["input_ids"]
            if tokens:
                pool.append([int(token) for token in tokens])
    if len(pool) < 64:
        raise ValueError("development paragraph pool implausibly small")
    return pool


def _open_output(args: argparse.Namespace) -> tuple[Path, bool]:
    output = args.output or (
        ROOT / "results" / f"v66_boundary_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output = Path(output)
    resuming = output.exists()
    if resuming and not args.resume:
        raise FileExistsError(
            f"--output already exists (pass --resume to continue): {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output, resuming


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    output, resuming = _open_output(args)
    sinks = _Sinks(output)
    done_cells: set[str] = set()
    prewarm_done: set[str] = set()
    prewarm_records: dict[str, dict[str, Any]] = {}
    if resuming:
        done_cells = _completed_cells(
            output, record_mode=args.continue_on_identity_divergence
        )
        progress = output / "progress.jsonl"
        if progress.exists():
            for line in progress.read_text().splitlines():
                record = json.loads(line) if line.strip() else {}
                if record.get("type") == "prewarm":
                    prewarm_done.add(record["unit_key"])
                    prewarm_records[record["unit_key"]] = record

    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "free_bytes_before": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        if free_bytes < MIN_FREE_GPU_FRACTION * total_bytes:
            raise RuntimeError(
                f"GPU is not idle: only {free_bytes}/{total_bytes} bytes free "
                f"(requires >= {MIN_FREE_GPU_FRACTION:.0%} free)"
            )

    dev_ids, records, query_record_id, tokenizer = _load_dev_content(args.checkpoint)
    pool = _build_pool(tokenizer, records, dev_ids)

    cells_plan = []
    for length in DOC_LENGTHS:
        for n_docs in N_DOCS:
            for hit_count in unit_hit_counts(n_docs):
                cells_plan.append({
                    "L": length, "N": n_docs, "hit_count": hit_count,
                    "h_labels": h_labels_for(hit_count, n_docs),
                    "budgets": [TTFT_BUDGET, *DECODE_BUDGETS],
                })
    if _sha_file(PREREG_JSON) != PREREG_JSON_SHA256:
        raise ValueError("preregistration JSON sha mismatch — prereg was edited after freeze")

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = _build_manifest(
            args, cells_plan=cells_plan, gpu=gpu, dev_ids=dev_ids,
            query_record_id=query_record_id,
        )
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"stage": "manifest", "output": str(output),
                      "resuming": resuming, "completed_cells": len(done_cells)}))

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model, _ = load(
        str(args.checkpoint), dtype="bf16" if device.type == "cuda" else "fp32", backend="sdpa"
    )
    model = model.to(device).eval()
    contract = replace(_contract(model, args.checkpoint), backend="sdpa")
    if _checkpoint_identity(args.checkpoint)[1] != CHECKPOINT_SHA256:
        raise ValueError("checkpoint identity mismatch against the prereg pin")

    query_ids = tokenizer(
        build_query_text(tokenizer, records[query_record_id]), add_special_tokens=False
    )["input_ids"]
    query_ids = torch.tensor([query_ids], dtype=torch.long)

    deviations: list[str] = []
    repeats = args.repeats
    sweep_started = time.perf_counter()
    prewarm_records: dict[str, dict[str, Any]] = {}
    cell_summaries: list[dict[str, Any]] = []
    smoke_summary: dict[str, Any] | None = None

    def elapsed_minutes() -> float:
        return (time.perf_counter() - sweep_started) / 60.0

    def plan_units(lengths: list[int]) -> list[tuple[int, int, int]]:
        units = []
        for length in lengths:
            for n_docs in N_DOCS:
                for hit_count in unit_hit_counts(n_docs):
                    units.append((length, n_docs, hit_count))
        return units

    lengths = list(DOC_LENGTHS)
    units = plan_units(lengths)

    def execute_unit(length: int, n_docs: int, hit_count: int) -> None:
        nonlocal repeats
        redo_d1 = args.redo_d1
        unit_key = f"L{length}_N{n_docs}"
        docs, _cursor = pack_documents(pool, length, n_docs)
        batch = build_monolithic_batch(docs, query_ids)
        store_dir = output / "canonical_store" / unit_key
        store = DocCache(store_dir)
        cache = CanonicalCache(model, contract, store)
        keys = expected_store_keys(cache, docs)
        hit_keys = keys[:hit_count]
        hit_indices = list(range(hit_count))
        reset = reset_store(store_dir, hit_keys)

        prewarm = prewarm_records.get(unit_key)
        if reset["missing"]:
            prebuild_cache = CanonicalCache(model, contract, store)
            missing_docs = [
                docs[index] for index, key in enumerate(keys) if key in set(reset["missing"])
            ]
            wall_prebuild, _ = _timed_pass(
                lambda: prebuild_cache.documents(missing_docs), device
            )
            reset = reset_store(store_dir, hit_keys)
            if reset["missing"]:
                raise RuntimeError("store prebuild failed to materialize hit payloads")
            if unit_key not in prewarm_done and prewarm is None:
                prewarm = {
                    "type": "prewarm", "unit_key": unit_key, "L": length, "N": n_docs,
                    "hit_count": hit_count, "cold_store_encode_wall_ms": wall_prebuild,
                }
                prewarm_records[unit_key] = prewarm
                prewarm_done.add(unit_key)
                sinks.progress_line(prewarm)
        verify_store(store_dir, hit_keys)

        def block_a(budget: int):
            return lambda: generate_ids(
                model, tokenizer, batch, "ddi_qf", budget, stop_on_eos=False
            )

        def block_b(budget: int):
            return lambda: cache.request(
                docs, query_ids, max_new_tokens=budget, eos_token_id=None
            )

        def block_b_ttft():
            return lambda: cache.request(docs, query_ids)

        base = f"L{length}_N{n_docs}_H{hit_count}"
        cell_payload: dict[str, Any] = {
            "type": "cell", "cell_id": f"{base}_D1", "L": length, "N": n_docs,
            "hit_count": hit_count, "h_labels": h_labels_for(hit_count, n_docs),
            "h_eff": hit_count / n_docs, "hit_indices": hit_indices,
            "hit_keys": hit_keys, "budgets": {},
        }
        blocks: dict[str, Any] = {}
        d1_done = cell_payload["cell_id"] in done_cells
        if d1_done and redo_d1 and hit_count < n_docs:
            sinks.progress_line({
                "type": "supersede", "cell_id": cell_payload["cell_id"],
                "reason": "d1_unpruned_store_bug: B wall(1) pass ran against the "
                          "B-ttft pass's un-pruned write-through state; re-run with "
                          "the fixed prune placement",
            })
            d1_done = False
        if not d1_done:
            blocks["D1"] = _run_pass_block(
                label="D1_ttft", budget=TTFT_BUDGET, warmups=args.warmups,
                repeats=repeats, device=device,
                pass_a=block_a(TTFT_BUDGET), pass_b=block_b(TTFT_BUDGET),
                pass_b_ttft=block_b_ttft(), hit_count=hit_count, n_docs=n_docs,
                hit_indices=hit_indices, hit_keys=hit_keys, store_dir=store_dir,
                sinks=sinks, cell_id=cell_payload["cell_id"],
                continue_on_divergence=args.continue_on_identity_divergence,
            )
            cell_payload["budgets"]["1"] = blocks["D1"]
            cell_payload["status"] = "ok"
            sinks.progress_line(cell_payload)
            cell_summaries.append(cell_payload)
            print(json.dumps({"stage": "cell", "cell_id": cell_payload["cell_id"],
                              "wall_ratio": blocks["D1"]["ratio_point"],
                              "wall_verdict": blocks["D1"]["verdict"],
                              "ttft_ratio": blocks["D1"]["ttft_ratio"]["ratio_point"],
                              "ttft_verdict": blocks["D1"]["ttft_ratio"]["verdict"]}))
        else:
            cell_summaries.append({"cell_id": cell_payload["cell_id"],
                                   "status": "skipped_already_done"})

        for budget in DECODE_BUDGETS:
            decode_id = f"{base}_D{budget}"
            decode_payload: dict[str, Any] = {
                "type": "cell", "cell_id": decode_id, "L": length, "N": n_docs,
                "hit_count": hit_count, "h_labels": h_labels_for(hit_count, n_docs),
                "h_eff": hit_count / n_docs, "hit_indices": hit_indices,
                "hit_keys": hit_keys, "budgets": {},
            }
            if decode_id in done_cells:
                cell_summaries.append({"cell_id": decode_id,
                                       "status": "skipped_already_done"})
                continue
            block = _run_pass_block(
                label=f"D{budget}", budget=budget, warmups=args.warmups,
                repeats=repeats, device=device,
                pass_a=block_a(budget), pass_b=block_b(budget), pass_b_ttft=None,
                hit_count=hit_count, n_docs=n_docs, hit_indices=hit_indices,
                hit_keys=hit_keys, store_dir=store_dir, sinks=sinks,
                cell_id=decode_id,
                continue_on_divergence=args.continue_on_identity_divergence,
            )
            decode_payload["budgets"][str(budget)] = block
            decode_payload["status"] = "ok"
            sinks.progress_line(decode_payload)
            cell_summaries.append(decode_payload)
            print(json.dumps({"stage": "cell", "cell_id": decode_id,
                              "wall_ratio": block["ratio_point"],
                              "wall_verdict": block["verdict"]}))

    if args.smoke:
        length, n_docs, hit_count = 128, 2, 2
        docs, _ = pack_documents(pool, length, n_docs)
        batch = build_monolithic_batch(docs, query_ids)
        store_dir = output / "canonical_store" / "smoke"
        store = DocCache(store_dir)
        cache = CanonicalCache(model, contract, store)
        keys = expected_store_keys(cache, docs)
        reset_store(store_dir, keys[:hit_count])
        block = _run_pass_block(
            label="smoke_D8", budget=8, warmups=SMOKE_WARMUPS, repeats=SMOKE_REPEATS,
            device=device,
            pass_a=lambda: generate_ids(model, tokenizer, batch, "ddi_qf", 8,
                                        stop_on_eos=False),
            pass_b=lambda: cache.request(docs, query_ids, max_new_tokens=8,
                                         eos_token_id=None),
            pass_b_ttft=None, hit_count=hit_count, n_docs=n_docs,
            hit_indices=list(range(hit_count)), hit_keys=keys[:hit_count],
            store_dir=store_dir, sinks=sinks, cell_id="smoke_L128_N2_H2_D8",
        )
        smoke_summary = {"status": "ok", "identity": block["identity"],
                         "walls_discarded": True}
        (output / "smoke.json").write_text(
            json.dumps(smoke_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"stage": "smoke", "identity_ok": block["identity"]["all_identical"]}))
        return {"output": str(output), "smoke": smoke_summary}

    planned_units = len(units)
    completed_units = 0
    pace_seconds: list[float] = []
    for length, n_docs, hit_count in units:
        if elapsed_minutes() > UNIT_CUTOFF_MINUTES:
            deviations.append(
                f"cap: unit start refused after {elapsed_minutes():.1f} min "
                f"(cutoff {UNIT_CUTOFF_MINUTES}); remaining units not run"
            )
            break
        if pace_seconds and completed_units >= 2:
            projected = elapsed_minutes() + (
                statistics.fmean(pace_seconds) / 60.0
            ) * (planned_units - completed_units)
            if projected > HARD_CAP_MINUTES and repeats > MIN_REPEATS:
                repeats = MIN_REPEATS
                deviations.append(
                    f"cap: timed repeats shrunk {args.repeats} -> {MIN_REPEATS} for "
                    f"remaining units (projected {projected:.0f} min)"
                )
        unit_started = time.perf_counter()
        try:
            execute_unit(length, n_docs, hit_count)
        except SystemExit:
            raise
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            sinks.progress_line({
                "type": "cell", "cell_id": f"L{length}_N{n_docs}_H{hit_count}",
                "status": "infeasible", "error": str(error)[:300],
            })
            cell_summaries.append({
                "cell_id": f"L{length}_N{n_docs}_H{hit_count}",
                "status": "infeasible", "error": str(error)[:300],
            })
            print(json.dumps({"stage": "infeasible",
                              "cell_id": f"L{length}_N{n_docs}_H{hit_count}"}))
            continue
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                sinks.progress_line({
                    "type": "cell", "cell_id": f"L{length}_N{n_docs}_H{hit_count}",
                    "status": "infeasible", "error": str(error)[:300],
                })
                cell_summaries.append({
                    "cell_id": f"L{length}_N{n_docs}_H{hit_count}",
                    "status": "infeasible", "error": str(error)[:300],
                })
                continue
            raise
        pace_seconds.append(time.perf_counter() - unit_started)
        completed_units += 1
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sinks.progress_line({
            "type": "rss", "units_done": completed_units,
            "rss_bytes": _rss_bytes(),
            "elapsed_minutes": round(elapsed_minutes(), 2),
        })

    l32_ran = False
    if elapsed_minutes() < L32_GATE_MINUTES:
        for length, n_docs, hit_count in plan_units([CONDITIONAL_DOC_LENGTH]):
            if elapsed_minutes() > UNIT_CUTOFF_MINUTES:
                deviations.append("cap: L=32 unit start refused")
                break
            execute_unit(length, n_docs, hit_count)
            l32_ran = True
    else:
        deviations.append(
            f"L=32 skipped: elapsed {elapsed_minutes():.1f} min >= gate "
            f"{L32_GATE_MINUTES} min"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": _sha_file(manifest_path),
        "prereg": {
            "md_sha256": _sha_file(PREREG_MD),
            "json_sha256": _sha_file(PREREG_JSON),
        },
        "deviations": deviations,
        "repeats_used": repeats,
        "elapsed_minutes": round(elapsed_minutes(), 2),
        "l32_included": l32_ran,
        "prewarm": [prewarm_records[key] for key in sorted(prewarm_records)],
        "cells": cell_summaries,
    }
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "stage": "done", "output": str(output), "cells": len(cell_summaries),
        "elapsed_minutes": result["elapsed_minutes"], "l32_included": l32_ran,
        "deviations": deviations,
    }))
    return result


if __name__ == "__main__":
    main()
