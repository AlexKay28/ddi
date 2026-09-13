"""VIBEQUEUE-71: trained-contract serving boundary sweep (preregistered).

Measures whether the V70 trained k=5 link-token reuse path keeps the V66
warm-path serving win, on the frozen V66 development grid. The protocol is
frozen in ``experiments/v71_preregistration.{md,json}``; the JSON SHA is
pinned below and the manifest is written before any model loads. Development
data only; confirmation IDs are forbidden.

Arms per (L, N, hit_count) cell, budgets d=1/TTFT then 8,16,32,64:
- ``A``  monolithic on the untrained restart checkpoint (V66 semantics,
  ``generate_ids``; restart positions).
- ``BP`` trained fresh (V70 ``checkpoint_k5``): per-pass independent canonical
  encode in memory, k=5 fixed-marker layout prefill, fixed-work greedy decode.
- ``CP`` trained cached: entries from a per-unit ``CanonicalCache``+``DocCache``
  store; mixed cells write through misses; identical executed callable to BP.
- ``DP`` trained full-causal no-link prefill; TTFT only, D1 block, diagnostic.

Primary gate: A/CP paired ratio of medians under the frozen V66 materiality
rule. Secondary gate: CP vs BP generated ids byte-identical per repeat
(divergence pauses the sweep). All other ratios are descriptive.

Outputs under ``results/v71_trained_serving_<UTC>/``: ``manifest.json``,
``progress.jsonl``, ``raw_repeats.jsonl``, ``result.json`` (and ``smoke.json``
in smoke mode). Nothing else is written.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddi.cache import DocCache  # noqa: E402
from ddi.canonical_cache import CanonicalCache  # noqa: E402
from ddi.inference import load  # noqa: E402
from ddi.link_token_contract import DocumentPayload  # noqa: E402
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
from scripts.run_v66_boundary import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    CONDITIONAL_DOC_LENGTH,
    DECODE_BUDGETS,
    DOC_LENGTHS,
    HIT_FRACTIONS,
    MIN_FREE_GPU_FRACTION,
    MIN_REPEATS,
    N_DOCS,
    REPEATS,
    SEED,
    SMOKE_REPEATS,
    SMOKE_WARMUPS,
    TTFT_BUDGET,
    WARMUPS,
    WIN_CI_LOW,
    _build_pool,
    _completed_cells,
    _ids_sha,
    _load_dev_content,
    _rss_bytes,
    _Sinks,
    _timed_pass,
    aggregate_identity,
    build_monolithic_batch,
    expected_store_keys,
    h_labels_for,
    identity_verdict,
    pack_documents,
    prune_store,
    ratio_summary,
    reset_store,
    unit_hit_counts,
    verify_store,
)
from scripts.run_v66_boundary import (  # noqa: E402
    CHECKPOINT_SHA256 as UNTRAINED_CHECKPOINT_SHA256,
)
from scripts.run_v66_boundary import (  # noqa: E402
    DEFAULT_CHECKPOINT as UNTRAINED_CHECKPOINT,
)
from scripts.v70_harness import (  # noqa: E402
    EncodedRequest,
    assert_original_inputs,
    generate_independent_ids,
    independent_forward_from_entries,
    install_fixed_link_tokens,
)
from scripts.v70_run import _checkpoint_hashes  # noqa: E402
from scripts.v71_serving_paths import (  # noqa: E402
    PrecomputedRequest,
    dp_ttft_fast,
    generate_fast,
    ttft_fast,
)
from scripts.validate_matched_qa import generate_ids  # noqa: E402

SCHEMA_VERSION = 71

PREREG_MD = ROOT / "experiments/v71_preregistration.md"
PREREG_JSON = ROOT / "experiments/v71_preregistration.json"
PREREG_JSON_SHA256 = "85f0d7b9a0bb3b8ab7a4a99bf87cbc1f52a03b48b02d5a945fae1d25998da07d"

TRAINED_CHECKPOINT = ROOT / "results/v70_arms_valid_seed47/checkpoint_k5"
TRAINED_CHECKPOINT_AGGREGATE_SHA256 = (
    "04fe4dcb3c70f0ef75f0cdfbf1f99164a90ab54df0833cb4b14cc56bd723ba87"
)

# Caps scaled from V66 for the added arms (prereg: Caps and stop rules).
UNIT_CUTOFF_MINUTES = 270.0
HARD_CAP_MINUTES = 300.0
L32_GATE_MINUTES = 170.0

K_LINKS = 5


# ---------------------------------------------------------------------------
# trained-contract request glue
# ---------------------------------------------------------------------------


class _MissCounter:
    """Counts write-through encodes fired by the cached CanonicalCache."""

    def __init__(self, cache: CanonicalCache):
        self.cache = cache
        self.count = 0
        self._wrap()

    def _wrap(self) -> None:
        original = self.cache.encode_document

        def counted(ids: torch.Tensor) -> Any:
            self.count += 1
            return original(ids)

        self.cache.encode_document = counted  # type: ignore[method-assign]

    def reset(self) -> int:
        value = self.count
        self.count = 0
        return value


def _payloads(docs: list[torch.Tensor]) -> tuple[DocumentPayload, ...]:
    return tuple(DocumentPayload(token_ids=tuple(int(t) for t in stream[0].tolist()))
                 for stream in docs)


def _trained_request(request_docs: tuple[DocumentPayload, ...],
                     query_token_ids: tuple[int, ...]) -> EncodedRequest:
    return EncodedRequest(documents=request_docs, query_token_ids=query_token_ids)


@torch.no_grad()
def _trained_ttft(model: Any, contract: Any, request: EncodedRequest,
                  entries: list[Any]) -> tuple[int, Any]:
    out = independent_forward_from_entries(model, contract, request, entries, k=K_LINKS)
    return int(out.logits[0, -1].argmax()), out


# ---------------------------------------------------------------------------
# pass block
# ---------------------------------------------------------------------------


def _run_pass_block(
    *,
    label: str,
    budget: int,
    warmups: int,
    repeats: int,
    device: torch.device,
    pass_a: Any,
    pass_cp: Any,
    pass_cp_ttft: Any | None,
    pass_bp: Any,
    pass_bp_ttft: Any | None,
    pass_dp_ttft: Any | None,
    miss_counter: _MissCounter,
    hit_count: int,
    n_docs: int,
    hit_keys: list[str],
    store_dir: Path,
    sinks: _Sinks,
    cell_id: str,
    continue_on_divergence: bool = False,
) -> dict[str, Any]:
    """One frozen pass block: shared warmups, then interleaved timed repeats.

    Frozen repeat order, D1 block: A, CP-TTFT, [prune], CP-wall, [prune],
    BP-TTFT, BP-wall, DP-TTFT. Decode blocks: A, CP-wall, [prune], BP-wall.
    Identity gate: CP vs BP generated ids byte-identical per repeat.
    """
    for _ in range(warmups):
        pass_a()
        if pass_cp_ttft is not None:
            pass_cp_ttft()
            if hit_count < n_docs:
                prune_store(store_dir, hit_keys)
            miss_counter.reset()
        pass_cp()
        if hit_count < n_docs:
            prune_store(store_dir, hit_keys)
        miss_counter.reset()
        if pass_bp_ttft is not None:
            pass_bp_ttft()
        pass_bp()
        if pass_dp_ttft is not None:
            pass_dp_ttft()
        _sync(device)

    a_walls: list[float] = []
    a_ttfts: list[float] = []
    cp_walls: list[float] = []
    cp_ttfts: list[float] = []
    bp_walls: list[float] = []
    bp_ttfts: list[float] = []
    dp_ttfts: list[float] = []
    miss_counts: list[int] = []
    identity_flags: list[bool] = []
    first_token_flags: list[bool] = []
    peaks: dict[str, list[tuple[int | None, int | None]]] = {
        "A": [], "CP": [], "BP": [], "DP": []
    }

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
            "ttft_ms": a_ttft, "ids_sha256": _ids_sha(a_ids),
            "peak_allocated_bytes": a_alloc, "peak_reserved_bytes": a_reserved,
        })

        if pass_cp_ttft is not None:
            _reset_peak(device)
            cp_ttft_wall, cp_first = _timed_pass(pass_cp_ttft, device)
            cp_alloc, cp_reserved = _peak_bytes(device)
            cp_ttfts.append(cp_ttft_wall)
            peaks["CP"].append((cp_alloc, cp_reserved))
            miss_counts.append(miss_counter.reset())
            if hit_count < n_docs:
                prune_store(store_dir, hit_keys)
            sinks.raw_line({
                "type": "pass", "cell_id": cell_id, "block": label, "arm": "CP",
                "budget": None, "repeat": repeat, "wall_ms": cp_ttft_wall,
                "ttft_ms": cp_ttft_wall, "first_token": cp_first,
                "miss_encodes": miss_counts[-1],
                "peak_allocated_bytes": cp_alloc, "peak_reserved_bytes": cp_reserved,
            })

        _reset_peak(device)
        cp_wall, cp_ids = _timed_pass(pass_cp, device)
        cp_alloc, cp_reserved = _peak_bytes(device)
        cp_walls.append(cp_wall)
        peaks["CP"].append((cp_alloc, cp_reserved))
        miss_counts.append(miss_counter.reset())
        if hit_count < n_docs:
            prune_store(store_dir, hit_keys)
        sinks.raw_line({
            "type": "pass", "cell_id": cell_id, "block": label, "arm": "CP",
            "budget": budget, "repeat": repeat, "wall_ms": cp_wall,
            "ids_sha256": _ids_sha(cp_ids), "miss_encodes": miss_counts[-1],
            "peak_allocated_bytes": cp_alloc, "peak_reserved_bytes": cp_reserved,
        })

        if pass_bp_ttft is not None:
            _reset_peak(device)
            bp_ttft_wall, bp_first = _timed_pass(pass_bp_ttft, device)
            bp_alloc, bp_reserved = _peak_bytes(device)
            bp_ttfts.append(bp_ttft_wall)
            peaks["BP"].append((bp_alloc, bp_reserved))
            sinks.raw_line({
                "type": "pass", "cell_id": cell_id, "block": label, "arm": "BP",
                "budget": None, "repeat": repeat, "wall_ms": bp_ttft_wall,
                "ttft_ms": bp_ttft_wall, "first_token": bp_first,
                "peak_allocated_bytes": bp_alloc, "peak_reserved_bytes": bp_reserved,
            })

        _reset_peak(device)
        bp_wall, bp_ids = _timed_pass(pass_bp, device)
        bp_alloc, bp_reserved = _peak_bytes(device)
        bp_walls.append(bp_wall)
        peaks["BP"].append((bp_alloc, bp_reserved))
        ok = identity_verdict(cp_ids, bp_ids)
        identity_flags.append(ok)
        first_token_flags.append(bool(cp_ids) and bool(bp_ids)
                                 and cp_ids[0] == bp_ids[0])
        sinks.raw_line({
            "type": "pass", "cell_id": cell_id, "block": label, "arm": "BP",
            "budget": budget, "repeat": repeat, "wall_ms": bp_wall,
            "ids_sha256": _ids_sha(bp_ids), "identity_ok": ok,
            "peak_allocated_bytes": bp_alloc, "peak_reserved_bytes": bp_reserved,
        })

        if not ok:
            sinks.progress_line({
                "type": "identity_failure", "cell_id": cell_id, "block": label,
                "repeat": repeat, "budget": budget, "hit_keys": hit_keys,
                "ids_cp": cp_ids, "ids_bp": bp_ids,
                "policy": "record" if continue_on_divergence else "abort",
                "diagnosis": (
                    "CP (store-provenance entries) and BP (fresh-encode entries) "
                    "diverged; under the V70 contract these must be byte-identical "
                    "— this indicates a contract break under serving assembly, "
                    "not a bf16 reduction-order flip"
                    if continue_on_divergence else "paused for diagnosis"
                ),
            })
            if not continue_on_divergence:
                raise SystemExit(
                    f"V71 PAUSED for identity diagnosis: cell {cell_id} block {label} "
                    f"repeat {repeat}: CP ids {cp_ids} != BP ids {bp_ids}"
                )

        if pass_dp_ttft is not None:
            _reset_peak(device)
            dp_ttft_wall, _unused = _timed_pass(pass_dp_ttft, device)
            del _unused
            dp_alloc, dp_reserved = _peak_bytes(device)
            dp_ttfts.append(dp_ttft_wall)
            peaks["DP"].append((dp_alloc, dp_reserved))
            sinks.raw_line({
                "type": "pass", "cell_id": cell_id, "block": label, "arm": "DP",
                "budget": None, "repeat": repeat, "wall_ms": dp_ttft_wall,
                "ttft_ms": dp_ttft_wall,
                "peak_allocated_bytes": dp_alloc, "peak_reserved_bytes": dp_reserved,
            })

    verify_store(store_dir, hit_keys)
    block: dict[str, Any] = {
        "block": label,
        "budget": budget,
        "repeats": repeats,
        **ratio_summary(a_walls, cp_walls),
        "median_a_ttft_ms": statistics.median(a_ttfts) if a_ttfts else None,
        "median_cp_ttft_ms": statistics.median(cp_ttfts) if cp_ttfts else None,
        "median_bp_wall_ms": statistics.median(bp_walls) if bp_walls else None,
        "ttft_ratio": ratio_summary(a_ttfts, cp_ttfts) if pass_cp_ttft is not None else None,
        "bp_wall_ratio_descriptive": ratio_summary(a_walls, bp_walls),
        "bp_ttft_ratio_descriptive": (
            ratio_summary(a_ttfts, bp_ttfts) if pass_bp_ttft is not None else None
        ),
        "dp_ttft_ratio_descriptive": (
            ratio_summary(a_ttfts, dp_ttfts) if pass_dp_ttft is not None else None
        ),
        "median_dp_ttft_ms": statistics.median(dp_ttfts) if dp_ttfts else None,
        "identity_cp_bp": aggregate_identity(identity_flags),
        "first_token_agrees_cp_bp": aggregate_identity(first_token_flags)
        if pass_cp_ttft is not None else None,
        "miss_encodes_per_pass": miss_counts,
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
    trained_hashes = _checkpoint_hashes(TRAINED_CHECKPOINT)
    return {
        "schema_version": SCHEMA_VERSION,
        "ticket": "VIBEQUEUE-71",
        "runner": "scripts/run_v71_trained_serving.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest_written_before_run": True,
        "preregistration": {
            "md": str(PREREG_MD),
            "md_sha256": _sha_file(PREREG_MD),
            "json": str(PREREG_JSON),
            "json_sha256": _sha_file(PREREG_JSON),
            "pinned_json_sha256": PREREG_JSON_SHA256,
        },
        "checkpoints": {
            "A": {
                "path": str(Path(args.untrained_checkpoint).resolve()),
                "identity": _checkpoint_identity(Path(args.untrained_checkpoint))[0],
                "sha256": _checkpoint_identity(Path(args.untrained_checkpoint))[1],
                "pinned_sha256": UNTRAINED_CHECKPOINT_SHA256,
            },
            "BCP": {
                "path": str(TRAINED_CHECKPOINT.resolve()),
                "aggregate_sha256": trained_hashes["sha256"],
                "pinned_aggregate_sha256": TRAINED_CHECKPOINT_AGGREGATE_SHA256,
                "files": len(trained_hashes["files"]),
            },
        },
        "data": {
            "role": "development_only",
            "dev_id_count": len(dev_ids),
            "confirmation_ids_used": 0,
            "query_record_id": query_record_id,
            "content_seed": SEED,
        },"protocol": {
            "seed": SEED,
            "warmups": args.warmups,
            "timed_repeats": args.repeats,
            "min_repeats_under_cap": MIN_REPEATS,
            "order": "interleaved A,CP,CP-ttft(D1),BP,BP-ttft(D1),DP-ttft(D1); "
                     "budgets d=1/TTFT then 8,16,32,64",
            "identity_gate": "CP vs BP byte-identical per repeat; divergence pauses",
            "synchronize": True,
            "device": args.device,
            "gpu": gpu,
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "iterations": args.bootstrap_iterations,
                "ci_level": CI_LEVEL,
                "estimator": "paired ratio of medians A/CP over repeat-index pairs",
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
            "win": f"A/CP ratio CI low > {WIN_CI_LOW}",
            "lose": "A/CP ratio CI high < 1/1.05",
            "tie": "otherwise",
            "declared_before_run": True,
        },
        "caps": {
            "unit_cutoff_minutes": UNIT_CUTOFF_MINUTES,
            "hard_cap_minutes": HARD_CAP_MINUTES,
        },
        "sources": {
            "scripts/run_v71_trained_serving.py": _sha_file(Path(__file__).resolve()),
            "ddi_tree": _source_hashes(),
        },
        "git": {
            "head": _git_text("rev-parse", "HEAD"),
            "diff_head_sha256": __import__("hashlib").sha256(
                _git_bytes("diff", "HEAD")
            ).hexdigest(),
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
    parser.add_argument("--untrained-checkpoint", type=Path, default=UNTRAINED_CHECKPOINT)
    parser.add_argument("--trained-checkpoint", type=Path, default=TRAINED_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--smoke", action="store_true",
                        help="run only the mechanics-validation smoke cell (timings discarded)")
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing output directory, skipping completed cells")
    parser.add_argument("--continue-on-identity-divergence", action="store_true",
                        help="record per-repeat CP/BP identity divergences and continue "
                             "(default: pause and exit)")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")
    return args


def _open_output(args: argparse.Namespace) -> tuple[Path, bool]:
    output = args.output or (
        ROOT / "results" / f"v71_trained_serving_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output = Path(output)
    resuming = output.exists()
    if resuming and not args.resume:
        raise FileExistsError(
            f"--output already exists (pass --resume to continue): {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output, resuming


def _load_trained(trained_checkpoint: Path, device: torch.device):
    hashes = _checkpoint_hashes(trained_checkpoint)
    if hashes["sha256"] != TRAINED_CHECKPOINT_AGGREGATE_SHA256:
        raise ValueError("trained checkpoint aggregate sha mismatch against the prereg pin")
    dtype = "bf16" if device.type == "cuda" else "fp32"
    model, tokenizer = load(str(trained_checkpoint), dtype=dtype, backend="sdpa")
    model = model.to(device).eval()
    token_state = install_fixed_link_tokens(model, tokenizer)
    contract = replace(_contract(model, trained_checkpoint), backend="sdpa")
    return model, tokenizer, token_state, contract, hashes["sha256"]


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

    dev_ids, records, query_record_id, tokenizer_a = _load_dev_content(
        args.untrained_checkpoint
    )
    pool = _build_pool(tokenizer_a, records, dev_ids)

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

    dtype = "bf16" if device.type == "cuda" else "fp32"
    model_a, _ = load(str(args.untrained_checkpoint), dtype=dtype, backend="sdpa")
    model_a = model_a.to(device).eval()
    if _checkpoint_identity(Path(args.untrained_checkpoint))[1] != UNTRAINED_CHECKPOINT_SHA256:
        raise ValueError("untrained checkpoint identity mismatch against the V66 pin")

    model_t, _tokenizer_t, token_state, contract_t, trained_sha = _load_trained(
        Path(args.trained_checkpoint), device
    )

    query_ids = tokenizer_a(
        build_query_text(tokenizer_a, records[query_record_id]), add_special_tokens=False
    )["input_ids"]
    query_ids = torch.tensor([query_ids], dtype=torch.long)
    query_token_ids = tuple(int(t) for t in query_ids[0].tolist())

    deviations: list[str] = []
    repeats = args.repeats
    sweep_started = time.perf_counter()
    cell_summaries: list[dict[str, Any]] = []

    def elapsed_minutes() -> float:
        return (time.perf_counter() - sweep_started) / 60.0

    def plan_units(lengths: list[int]) -> list[tuple[int, int, int]]:
        units = []
        for length in lengths:
            for n_docs in N_DOCS:
                for hit_count in unit_hit_counts(n_docs):
                    units.append((length, n_docs, hit_count))
        return units

    def execute_unit(length: int, n_docs: int, hit_count: int) -> None:
        unit_key = f"L{length}_N{n_docs}"
        docs, _cursor = pack_documents(pool, length, n_docs)
        batch = build_monolithic_batch(docs, query_ids)
        payloads = _payloads(docs)
        request = _trained_request(payloads, query_token_ids)
        assert_original_inputs(token_state, request)
        pre = PrecomputedRequest(
            model_t, token_state.contract, request,
            k=K_LINKS, max_answer_tokens=max(DECODE_BUDGETS),
        )

        store_dir = output / "trained_store" / unit_key
        store = DocCache(store_dir)
        cache = CanonicalCache(model_t, contract_t, store)
        miss_counter = _MissCounter(cache)
        fresh = CanonicalCache(model_t, contract_t)
        keys = expected_store_keys(cache, docs)
        hit_keys = keys[:hit_count]
        if hit_count == n_docs:
            def fail_encode(_: torch.Tensor) -> Any:
                raise AssertionError("all-hit cell attempted a write-through encode")
            cache.encode_document = fail_encode  # type: ignore[method-assign]

        reset = reset_store(store_dir, hit_keys)
        if reset["missing"]:
            prebuild = CanonicalCache(model_t, contract_t, store)
            missing_docs = [
                docs[index] for index, key in enumerate(keys)
                if key in set(reset["missing"])
            ]
            wall_prebuild, _ = _timed_pass(
                lambda: prebuild.documents(missing_docs), device
            )
            reset = reset_store(store_dir, hit_keys)
            if reset["missing"]:
                raise RuntimeError("store prebuild failed to materialize hit payloads")
            verify_store(store_dir, hit_keys)
            if unit_key not in prewarm_done and unit_key not in prewarm_records:
                record = {
                    "type": "prewarm", "unit_key": unit_key, "L": length, "N": n_docs,
                    "cold_store_encode_wall_ms": wall_prebuild,
                }
                prewarm_records[unit_key] = record
                prewarm_done.add(unit_key)
                sinks.progress_line(record)

        def pass_a(budget: int):
            return lambda: generate_ids(
                model_a, tokenizer_a, batch, "ddi_qf", budget, stop_on_eos=False
            )

        def pass_cp(budget: int):
            def run():
                entries = cache.documents(docs)
                return generate_fast(
                    model_t, request, entries, pre,
                    max_new_tokens=budget, eos_token_id=None,
                )
            return run

        def pass_cp_ttft():
            def run():
                entries = cache.documents(docs)
                return ttft_fast(model_t, request, entries, pre)
            return run

        def pass_bp(budget: int):
            def run():
                entries = fresh.documents(docs)
                return generate_fast(
                    model_t, request, entries, pre,
                    max_new_tokens=budget, eos_token_id=None,
                )
            return run

        def pass_bp_ttft():
            def run():
                entries = fresh.documents(docs)
                return ttft_fast(model_t, request, entries, pre)
            return run

        def pass_dp_ttft():
            return lambda: dp_ttft_fast(model_t, pre)

        base = f"L{length}_N{n_docs}_H{hit_count}"

        def run_block(cell_id: str, label: str, budget: int, with_ttft: bool) -> dict[str, Any]:
            return _run_pass_block(
                label=label, budget=budget, warmups=args.warmups, repeats=repeats,
                device=device,
                pass_a=pass_a(budget),
                pass_cp=pass_cp(budget),
                pass_cp_ttft=pass_cp_ttft() if with_ttft else None,
                pass_bp=pass_bp(budget),
                pass_bp_ttft=pass_bp_ttft() if with_ttft else None,
                pass_dp_ttft=pass_dp_ttft() if with_ttft else None,
                miss_counter=miss_counter, hit_count=hit_count, n_docs=n_docs,
                hit_keys=hit_keys, store_dir=store_dir, sinks=sinks,
                cell_id=cell_id,
                continue_on_divergence=args.continue_on_identity_divergence,
            )

        def emit(cell_payload: dict[str, Any], block: dict[str, Any]) -> None:
            cell_payload["status"] = "ok"
            sinks.progress_line(cell_payload)
            cell_summaries.append(cell_payload)
            print(json.dumps({
                "stage": "cell", "cell_id": cell_payload["cell_id"],
                "wall_ratio": block["ratio_point"],
                "wall_verdict": block["verdict"],
                "identity": block["identity_cp_bp"]["all_identical"],
            }))

        d1_id = f"{base}_D1"
        if d1_id not in done_cells:
            d1_payload: dict[str, Any] = {
                "type": "cell", "cell_id": d1_id, "L": length, "N": n_docs,
                "hit_count": hit_count, "h_labels": h_labels_for(hit_count, n_docs),
                "h_eff": hit_count / n_docs, "hit_keys": hit_keys, "budgets": {},
            }
            block = run_block(d1_id, "D1_ttft", TTFT_BUDGET, with_ttft=True)
            d1_payload["budgets"]["1"] = block
            emit(d1_payload, block)
        else:
            cell_summaries.append({"cell_id": d1_id, "status": "skipped_already_done"})

        for budget in DECODE_BUDGETS:
            decode_id = f"{base}_D{budget}"
            if decode_id in done_cells:
                cell_summaries.append({"cell_id": decode_id,
                                       "status": "skipped_already_done"})
                continue
            decode_payload: dict[str, Any] = {
                "type": "cell", "cell_id": decode_id, "L": length, "N": n_docs,
                "hit_count": hit_count, "h_labels": h_labels_for(hit_count, n_docs),
                "h_eff": hit_count / n_docs, "hit_keys": hit_keys, "budgets": {},
            }
            block = run_block(decode_id, f"D{budget}", budget, with_ttft=False)
            decode_payload["budgets"][str(budget)] = block
            emit(decode_payload, block)

    if args.smoke:
        length, n_docs, hit_count = 128, 2, 2
        docs, _ = pack_documents(pool, length, n_docs)
        batch = build_monolithic_batch(docs, query_ids)
        payloads = _payloads(docs)
        request = _trained_request(payloads, query_token_ids)
        pre = PrecomputedRequest(
            model_t, token_state.contract, request,
            k=K_LINKS, max_answer_tokens=max(DECODE_BUDGETS),
        )
        store_dir = output / "trained_store" / "smoke"
        store = DocCache(store_dir)
        cache = CanonicalCache(model_t, contract_t, store)
        miss_counter = _MissCounter(cache)
        fresh = CanonicalCache(model_t, contract_t)
        keys = expected_store_keys(cache, docs)
        reset_store(store_dir, keys[:hit_count])
        block = _run_pass_block(
            label="smoke_D8", budget=8, warmups=SMOKE_WARMUPS, repeats=SMOKE_REPEATS,
            device=device,
            pass_a=lambda: generate_ids(
                model_a, tokenizer_a, batch, "ddi_qf", 8, stop_on_eos=False),
            pass_cp=lambda: generate_fast(
                model_t, request, cache.documents(docs), pre,
                max_new_tokens=8, eos_token_id=None),
            pass_cp_ttft=None,
            pass_bp=lambda: generate_fast(
                model_t, request, fresh.documents(docs), pre,
                max_new_tokens=8, eos_token_id=None),
            pass_bp_ttft=None, pass_dp_ttft=None,
            miss_counter=miss_counter, hit_count=hit_count, n_docs=n_docs,
            hit_keys=keys[:hit_count], store_dir=store_dir, sinks=sinks,
            cell_id="smoke_L128_N2_H2_D8",
        )
        smoke_cp_entries = cache.documents(docs)
        slow_ids = generate_independent_ids(
            model_t, token_state.contract, request, smoke_cp_entries, k=K_LINKS,
            max_new_tokens=8, eos_token_id=None,
        )
        fast_ids = generate_fast(
            model_t, request, smoke_cp_entries, pre, max_new_tokens=8,
            eos_token_id=None,
        )
        slow_out = independent_forward_from_entries(
            model_t, token_state.contract, request, smoke_cp_entries, k=K_LINKS
        )
        slow_first = int(slow_out.logits[0, -1].argmax())
        cross_check = {
            "ids_match": fast_ids == slow_ids,
            "ttft_match": ttft_fast(model_t, request, smoke_cp_entries, pre) == slow_first,
        }
        smoke_summary = {
            "status": "ok",
            "identity": block["identity_cp_bp"],
            "fast_vs_v70_harness_cross_check": cross_check,
            "walls_discarded": True,
        }
        (output / "smoke.json").write_text(
            json.dumps(smoke_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"stage": "smoke",
                          "identity_ok": block["identity_cp_bp"]["all_identical"],
                          "cross_check_ok": all(cross_check.values())}))
        if not all(cross_check.values()):
            raise SystemExit("V71 smoke cross-check failed: fast path diverged "
                             "from the V70 harness path")
        return {"output": str(output), "smoke": smoke_summary}

    units = plan_units(list(DOC_LENGTHS))
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
            cell_id = f"L{length}_N{n_docs}_H{hit_count}"
            sinks.progress_line({
                "type": "cell", "cell_id": cell_id,
                "status": "infeasible", "error": str(error)[:300],
            })
            cell_summaries.append({
                "cell_id": cell_id, "status": "infeasible", "error": str(error)[:300],
            })
            print(json.dumps({"stage": "infeasible", "cell_id": cell_id}))
            continue
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                cell_id = f"L{length}_N{n_docs}_H{hit_count}"
                sinks.progress_line({
                    "type": "cell", "cell_id": cell_id,
                    "status": "infeasible", "error": str(error)[:300],
                })
                cell_summaries.append({
                    "cell_id": cell_id, "status": "infeasible",
                    "error": str(error)[:300],
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
        "checkpoints": {
            "A_sha256": UNTRAINED_CHECKPOINT_SHA256,
            "BCP_aggregate_sha256": trained_sha,
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
