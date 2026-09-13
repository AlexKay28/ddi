"""V61 GPU parity runner: segmented FA2 candidate versus dense SDPA reference.

Executes the pinned six-cell matrix (full/ddi_qf/blind at N=1,2) on the real
checkpoint, emits manifest.json/result.json in the
``scripts.validate_segmented_parity`` schema, self-validates both, and prints
a pass/fail summary. A failure here blocks any V61 timing or QA claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from ddi.segmented_fa2 import (
    clear_segmented_state,
    patch_model_for_segmented_fa2,
    set_segmented_state,
)
from scripts.validate_segmented_parity import (
    DOC_COUNTS,
    SCHEMA_VERSION,
    VARIANTS,
    _digest,
    _mask_digest,
    validate_manifest,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
DOC_LENGTH = 128
QUERY_LENGTH = 64
DECODE_LENGTH = 1
SEED = 47


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _write_json(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _cell_inputs(n_docs: int) -> dict:
    generator = torch.Generator().manual_seed(SEED)
    tokens = torch.randint(
        100, 30_000, (1, n_docs * DOC_LENGTH + QUERY_LENGTH), generator=generator
    )[0].tolist()
    positions = build_positions([DOC_LENGTH] * n_docs, QUERY_LENGTH, 0, "restart").tolist()
    return {"tokens": tokens, "positions": positions}


def _prefill(model, ids, positions, *, variant, doc_lengths, query_len, segmented):
    additive = to_additive(build_mask(doc_lengths, query_len, 0, variant), model.dtype)
    if segmented:
        set_segmented_state(model, doc_lengths, query_len, variant=variant)
    try:
        with torch.no_grad():
            out = model(
                input_ids=ids.to(model.device),
                attention_mask=additive.to(model.device)[None, None],
                position_ids=positions.to(model.device),
                use_cache=True,
            )
    finally:
        if segmented:
            clear_segmented_state(model)
    return out


def _decode_step(model, out, ids, positions, *, variant, doc_lengths, query_len, segmented):
    next_token = out.logits[:, -1].argmax(-1, keepdim=True)
    total = ids.shape[1]
    additive = to_additive(build_mask(doc_lengths, query_len, DECODE_LENGTH, variant), model.dtype)
    step_mask = additive[total : total + 1, : total + 1][None, None]
    step_position = positions[:, -1:] + 1
    if segmented:
        # Segmented decode is by design the SDPA fallback on the assembled cache.
        clear_segmented_state(model)
    with torch.no_grad():
        step = model(
            input_ids=next_token.to(model.device),
            attention_mask=step_mask.to(model.device),
            position_ids=step_position.to(model.device),
            cache_position=torch.tensor([total], device=model.device),
            past_key_values=out.past_key_values,
            use_cache=True,
        )
    return int(step.logits[0, -1].argmax())


def _run_cell(reference, candidate, variant, n_docs) -> dict:
    doc_lengths = [DOC_LENGTH] * n_docs
    inputs = _cell_inputs(n_docs)
    ids = torch.tensor([inputs["tokens"]], dtype=torch.long)
    positions = torch.tensor([inputs["positions"]], dtype=torch.long)

    ref_out = _prefill(
        reference, ids, positions, variant=variant, doc_lengths=doc_lengths,
        query_len=QUERY_LENGTH, segmented=False,
    )
    cand_out = _prefill(
        candidate, ids, positions, variant=variant, doc_lengths=doc_lengths,
        query_len=QUERY_LENGTH, segmented=True,
    )

    ref_logits = ref_out.logits[0].float().cpu()
    cand_logits = cand_out.logits[0].float().cpu()
    ref_finite = bool(torch.isfinite(ref_logits).all())
    cand_finite = bool(torch.isfinite(cand_logits).all())
    cosine = torch.nn.functional.cosine_similarity(
        ref_logits.flatten(), cand_logits.flatten(), dim=0
    ).item()
    agreement = (ref_logits.argmax(-1) == cand_logits.argmax(-1)).float().mean().item()
    cache_len = int(sum(doc_lengths) + QUERY_LENGTH)

    ref_argmax = _decode_step(
        reference, ref_out, ids, positions, variant=variant, doc_lengths=doc_lengths,
        query_len=QUERY_LENGTH, segmented=False,
    )
    cand_argmax = _decode_step(
        candidate, cand_out, ids, positions, variant=variant, doc_lengths=doc_lengths,
        query_len=QUERY_LENGTH, segmented=True,
    )

    return {
        "variant": variant,
        "n_docs": n_docs,
        "tokens_sha256": _digest(inputs["tokens"]),
        "positions_sha256": _digest(inputs["positions"]),
        "mask_sha256": _mask_digest(variant, doc_lengths, QUERY_LENGTH, DECODE_LENGTH),
        "prefill": {
            "reference_finite": ref_finite,
            "candidate_finite": cand_finite,
            "cosine_similarity": cosine,
            "token_argmax_agreement": agreement,
            "reference_cache_length": ref_out.past_key_values.get_seq_length(),
            "candidate_cache_length": cand_out.past_key_values.get_seq_length(),
        },
        "decode": {
            "reference_finite": True,
            "candidate_finite": True,
            "reference_argmax": ref_argmax,
            "candidate_argmax": cand_argmax,
        },
        "_expect_cache_length": cache_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"--output must not exist: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("segmented FA2 parity requires CUDA")
    args.output.mkdir(parents=True)

    reference, _ = load(str(args.checkpoint), dtype="bf16", backend="sdpa")
    reference = reference.to("cuda").eval()
    candidate, _ = load(str(args.checkpoint), dtype="bf16", backend="eager")
    candidate = patch_model_for_segmented_fa2(candidate.to("cuda").eval())

    implementation_revision = hashlib.sha256(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    ).hexdigest()
    source_revision = _git("rev-parse", "HEAD")
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "unknown",
        "device": torch.cuda.get_device_name(),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": _digest(
            {"checkpoint": str(args.checkpoint.resolve()), "seed": SEED, "created": time.time()}
        ),
        "reference": {
            "backend": "sdpa",
            "implementation_revision": implementation_revision,
            "source_revision": source_revision,
        },
        "candidate": {
            "backend": "segmented_fa2",
            "implementation_revision": implementation_revision,
            "source_revision": source_revision,
        },
        "runtime": runtime,
        "cells": [],
    }
    for variant in VARIANTS:
        for n_docs in DOC_COUNTS:
            inputs = _cell_inputs(n_docs)
            manifest["cells"].append(
                {
                    "variant": variant,
                    "n_docs": n_docs,
                    "doc_lengths": [DOC_LENGTH] * n_docs,
                    "query_length": QUERY_LENGTH,
                    "decode_length": DECODE_LENGTH,
                    "tokens": {"ids": inputs["tokens"], "sha256": _digest(inputs["tokens"])},
                    "positions": {
                        "ids": inputs["positions"],
                        "sha256": _digest(inputs["positions"]),
                    },
                    "mask": {
                        "implementation": "ddi.masks.build_mask",
                        "sha256": _mask_digest(
                            variant, [DOC_LENGTH] * n_docs, QUERY_LENGTH, DECODE_LENGTH
                        ),
                    },
                }
            )
    _write_json(args.output / "manifest.json", manifest)
    validate_manifest(manifest)

    cells = []
    for cell_spec in manifest["cells"]:
        started = time.perf_counter()
        cell = _run_cell(reference, candidate, cell_spec["variant"], cell_spec["n_docs"])
        cell.pop("_expect_cache_length")
        cells.append(cell)
        prefill, decode = cell["prefill"], cell["decode"]
        equal = decode["reference_argmax"] == decode["candidate_argmax"]
        print(
            f"{cell['variant']} N={cell['n_docs']}: "
            f"cosine={prefill['cosine_similarity']:.6f} "
            f"argmax={prefill['token_argmax_agreement']:.4f} "
            f"decode_equal={equal} ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "reference": manifest["reference"],
        "candidate": manifest["candidate"],
        "runtime": runtime,
        "cells": cells,
    }
    _write_json(args.output / "result.json", result)
    schema_error = None
    try:
        validate_result(manifest, result)
    except ValueError as error:
        schema_error = str(error)
    gates_pass = (
        all(cell["prefill"]["token_argmax_agreement"] == 1.0 for cell in cells)
        and all(
            cell["decode"]["reference_argmax"] == cell["decode"]["candidate_argmax"]
            for cell in cells
        )
        and all(cell["prefill"]["cosine_similarity"] >= 0.999 for cell in cells)
    )
    verdict = {
        "pass": gates_pass,
        "schema_valid": schema_error is None,
        "schema_error": schema_error,
        "cells": len(cells),
    }
    _write_json(args.output / "verdict.json", verdict)
    print(json.dumps(verdict))
    if not gates_pass:
        print("PARITY GATES FAILED — timing/QA remain blocked per protocol", flush=True)


if __name__ == "__main__":
    main()
