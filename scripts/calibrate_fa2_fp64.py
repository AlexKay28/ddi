"""V61 FP64-reference calibration study — measure cross-kernel numerical error margins.

Motivation: the V61 pinned parity gate (``scripts/validate_segmented_parity.py``) requires
prefill ``token_argmax_agreement == 1.0`` and equal decode argmax. That gate fails on
benign bf16/FA2 rounding, so this study measures the actual error scale of the kernels
involved and derives a PROPOSAL for a predeclared task-level RMS gate to replace it.

Method (one run, bounded):
1. Load the real restart checkpoint (bf16, backend ``sdpa``) and run ONE fixed
   N=2, L=128, Q=64 ddi_qf prefill. A forward pre-hook on layer 0 ``self_attn``
   captures the module inputs; q/k/v projections and RoPE-applied q/k are then
   recomputed with the layer's own weights and ``apply_rotary_pos_emb``.
2. On the captured states compute the layer-0 attention output four ways:
   fp64 dense (reference), fp32+SDPA, bf16+SDPA, bf16+FA2 (the exact production
   ``ddi.segmented_fa2._segmented_prefill`` varlen path with cu_seqlens).
3. Report per-method max/RMS relative error vs the fp64 reference (whole tensor
   and worst head), plus pairwise argmax disagreement at the attention-output
   level (argmax over the head_dim axis per (token, head)).
4. Write ``summary.json`` + ``CALIBRATION.md`` + captured states/outputs into
   ``results/v61_fp64_calibration_<UTC timestamp>/``. The gate in the summary is
   explicitly a PROPOSAL pending owner ratification.

Run: ``.venv/bin/python scripts/calibrate_fa2_fp64.py`` (needs CUDA + flash-attn).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

# Fixed study shape: two documents, one query block — the smallest N>1 ddi_qf case the
# V61 matrix pins (validate_segmented_parity DOC_COUNTS includes N=2).
DOC_LENGTHS = [128, 128]
QUERY_LEN = 64
ANSWER_LEN = 0
LAYOUT = "restart"
VARIANT = "ddi_qf"
METHODS = ("fp64_dense", "fp32_sdpa", "bf16_sdpa", "bf16_fa2_segmented")
REFERENCE_METHOD = "fp64_dense"
# Methods a task-level gate would police: the reference backend and the candidate kernel.
GATED_METHODS = ("bf16_sdpa", "bf16_fa2_segmented")
SAFETY_FACTOR = 10.0
SCHEMA = "v61_fp64_calibration_v1"

# Realistic fixed input: hotpot-style multi-hop facts (two documents, one question).
# Encoded with add_special_tokens=False and head-truncated / pad-padded to the fixed
# shape; the exact ids and their digest are recorded in the artifacts.
DOC_TEXTS = (
    "The Auroria Deepsky Telescope, completed in 1987 on Mount Cerulean, was the first "
    "observatory designed specifically for spectroscopy of rapidly rotating pulsars. Its "
    "principal investigator, Dr. Helena Voss, later led the Meridian Sky Survey, which "
    "catalogued over four thousand emission-line stars. The survey's southern station was "
    "built near Lake Argento, a reservoir created in 1954 to cool the Kestrel Valley "
    "smelter, which processed nickel ore shipped from the Port of Dunmore.",
    "Nickel from the Kestrel Valley smelter was refined using the Ostergaard process, "
    "patented in 1931 by engineer Anders Ostergaard of Malmo. The process consumed large "
    "amounts of pyrite, which was imported from the Valdarno mines in Tuscany. The mines "
    "closed in 1979, two years after the publication of Elena Ferraro's monograph on "
    "medieval Tuscan mining law, a work that earned her the Aldobrandi Prize and a "
    "professorship at the University of Siena.",
)
QUERY_TEXT = (
    "Question: Which observatory's principal investigator led a sky survey whose southern "
    "station used water from a reservoir built for a smelter that refined metal using a "
    "process named after its Swedish inventor? Answer:"
)


# ---- pure math helpers (CPU-tested in tests/test_calibrate_fa2_fp64.py) -----------------


def expand_kv(
    k: torch.Tensor, v: torch.Tensor, num_q_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native-GQA K/V [1, Hkv, T, D] -> query-head count via repeat_interleave (repo convention)."""
    groups = num_q_heads // k.shape[1]
    if groups == 1:
        return k, v
    return k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Masked softmax attention computed by explicit matmuls in ``dtype``.

    q [1, Hq, T, D]; k/v [1, Hkv, T, D] (GQA-expanded internally); allow bool [T, T]
    (True = attend). Every row must allow at least one key, else softmax is NaN —
    ddi_qf guarantees each row attends its own diagonal, but the guard is explicit.
    """
    if allow.shape != (q.shape[2], k.shape[2]):
        raise ValueError(f"allow shape {tuple(allow.shape)} does not match q/k token counts")
    if not bool(allow.any(dim=1).all()):
        raise ValueError("allow matrix has a row with no allowed key; softmax would be NaN")
    qq = q.to(dtype)
    kk, vv = expand_kv(k.to(dtype), v.to(dtype), qq.shape[1])
    scores = torch.matmul(qq, kk.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~allow.to(qq.device), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), vv)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """SDPA path with the repo's additive-mask convention (``ddi.masks.to_additive``)."""
    from ddi.masks import to_additive

    additive = to_additive(allow.to(q.device), dtype)
    return F.scaled_dot_product_attention(
        q.to(dtype),
        *expand_kv(k.to(dtype), v.to(dtype), q.shape[1]),
        attn_mask=additive[None, None],
        scale=scale,
    )


def rms_relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    """RMS(out - ref) / RMS(ref), both accumulated in float64."""
    diff = out.double() - ref.double()
    denom = ref.double().pow(2).mean().sqrt()
    if float(denom) == 0.0:
        raise ValueError("reference is all zeros; relative error undefined")
    return float(diff.pow(2).mean().sqrt() / denom)


def max_abs_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.double() - ref.double()).abs().max())


def worst_head_rms_relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Max over heads of RMS_h(out - ref) / RMS_h(ref); heads = dim 1 of [1, H, T, D]."""
    diff = out.double() - ref.double()
    per_head = diff.pow(2).mean(dim=(0, 2, 3)).sqrt()
    ref_head = ref.double().pow(2).mean(dim=(0, 2, 3)).sqrt()
    return float((per_head / ref_head).max())


def argmax_disagreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of (token, head) rows whose argmax over the feature dim differs."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a.argmax(dim=-1) != b.argmax(dim=-1)).double().mean())


def pairwise_argmax_disagreement(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    """All unordered method pairs -> attention-output argmax disagreement rate."""
    table: dict[str, float] = {}
    names = list(outputs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            table[f"{a}_vs_{b}"] = argmax_disagreement(outputs[a], outputs[b])
    return table


def error_table(
    outputs: dict[str, torch.Tensor], reference: torch.Tensor, *, skip: str = REFERENCE_METHOD
) -> dict[str, dict[str, float]]:
    """Per-method error metrics vs the reference; the reference itself is skipped."""
    return {
        name: {
            "rms_rel": rms_relative_error(out, reference),
            "max_abs": max_abs_error(out, reference),
            "worst_head_rms_rel": worst_head_rms_relative_error(out, reference),
        }
        for name, out in outputs.items()
        if name != skip
    }


def pad_or_truncate(ids: list[int], length: int, pad_id: int) -> list[int]:
    """Head-truncate to ``length`` or tail-pad with ``pad_id``; deterministic."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if len(ids) > length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def round_up_to_grid(value: float) -> float:
    """Round up to the 1-2-5 grid: 0.003 -> 0.01, 0.03 -> 0.05, 0.6 -> 1.0, 12 -> 20."""
    if value <= 0:
        raise ValueError(f"value must be > 0, got {value}")
    decade = 1.0
    while decade > value:
        decade /= 10.0
    while decade * 10.0 <= value:
        decade *= 10.0
    for mantissa in (1.0, 2.0, 5.0, 10.0):
        candidate = mantissa * decade
        if value <= candidate:
            return candidate
    raise AssertionError("unreachable")


def derive_gate(worst_measured_rms_rel: float, factor: float = SAFETY_FACTOR) -> dict[str, Any]:
    """Derive the PROPOSED gate value from the measured worst candidate-vs-fp64 RMS error.

    Gate = factor * worst measured, rounded UP onto the 1-2-5 grid so the published
    number is a round figure a reviewer can restate without ambiguity.
    """
    if worst_measured_rms_rel <= 0:
        raise ValueError("worst_measured_rms_rel must be > 0 for a meaningful gate")
    raw = factor * worst_measured_rms_rel
    return {
        "metric": "attention-output RMS relative error vs fp64 dense reference",
        "proposed_max": round_up_to_grid(raw),
        "derivation": (
            f"{factor:g}x the worst measured candidate-vs-fp64 rms_rel "
            f"({worst_measured_rms_rel:.3e}), rounded up on the 1-2-5 grid"
        ),
        "safety_factor": factor,
        "status": "PROPOSAL — pending owner ratification; not a ratified gate",
    }


def render_calibration_md(summary: dict[str, Any]) -> str:
    """Render the short CALIBRATION.md from the summary dict (kept in sync by construction)."""
    gate = summary["gate_proposal"]
    lines = [
        "# V61 FP64-reference calibration (layer 0, one fixed input)",
        "",
        "## Context",
        "",
        "The V61 pinned parity gate requires prefill `token_argmax_agreement == 1.0` and equal",
        "decode argmax (`scripts/validate_segmented_parity.py`). That 100%-argmax criterion fails",
        "on benign cross-kernel rounding. This study measures the actual numerical error margins",
        "so a task-level criterion can replace it.",
        "",
        "## Method",
        "",
        f"Checkpoint `{summary['checkpoint']['path']}` (bf16, backend sdpa), layer 0 self_attn.",
        f"One fixed ddi_qf prefill: docs {summary['input']['doc_lengths']} tokens each, query",
        f"{summary['input']['query_len']} tokens, `{LAYOUT}` positions, "
        f"`ddi.masks.build_mask` + `{VARIANT}`. Q/K/V captured post-RoPE from the real forward;",
        "attention recomputed four ways and compared against an fp64 dense reference",
        "(explicit masked-softmax matmuls in float64). FA2 uses the production",
        "`ddi.segmented_fa2._segmented_prefill` varlen path with cu_seqlens.",
        "",
        "## Measured error vs fp64 reference",
        "",
        "| method | RMS rel | max abs | worst-head RMS rel |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in summary["errors_vs_fp64_dense"].items():
        lines.append(
            f"| {name} | {row['rms_rel']:.3e} | {row['max_abs']:.3e} "
            f"| {row['worst_head_rms_rel']:.3e} |"
        )
    lines += [
        "",
        "## Pairwise argmax disagreement (attention-output level, argmax over head_dim)",
        "",
        "| pair | disagreement |",
        "| --- | --- |",
    ]
    for pair, rate in summary["pairwise_argmax_disagreement"].items():
        lines.append(f"| {pair} | {rate:.4f} |")
    lines += [
        "",
        "## PROPOSED task-level gate (NOT ratified)",
        "",
        f"> **PROPOSAL**: attention-output RMS relative error vs the fp64 reference "
        f"**<= {gate['proposed_max']:g}**",
        f"> ({gate['derivation']}).",
        "",
        "This is a calibration-derived suggestion for the owner to ratify, amend or reject. It",
        "is not a ratified gate and must not be wired into validation until someone owns it.",
        "",
        "## Limitations (read before citing)",
        "",
        "- Layer 0 only, one fixed input, T=320 tokens, one checkpoint. Other layers/inputs",
        "  may differ; deeper layers see larger-magnitude values than embeddings.",
        "- Argmax disagreement here is over the attention-output feature dim, NOT over the",
        "  vocabulary; it does not directly predict token-level divergence.",
        "- fp64 dense is the reference, not ground truth for the full model; it removes kernel",
        "  numerics, not the bf16 projection/rounding already baked into the captured states.",
        "- Captured states were produced by the SDPA forward with the ddi_qf additive mask;",
        "  layer-0 attention inputs are mask-independent, but later layers are not captured.",
    ]
    return "\n".join(lines) + "\n"


# ---- provenance helpers -----------------------------------------------------------------


def _sha256_tensor(t: torch.Tensor) -> str:
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _runtime_info() -> dict[str, str]:
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda or "none",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import flash_attn

        info["flash_attn"] = flash_attn.__version__
    except ImportError:
        info["flash_attn"] = "not installed"
    return info


# ---- input construction and state capture (GPU) ------------------------------------------


def build_input_ids(
    tokenizer: Any, doc_texts: list[str], query_text: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fixed [1, T] token stream: docs (head-truncated/padded to L) then query (to Q)."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    doc_ids = [
        pad_or_truncate(tokenizer.encode(text, add_special_tokens=False), length, pad_id)
        for text, length in zip(doc_texts, DOC_LENGTHS, strict=True)
    ]
    query_ids = pad_or_truncate(
        tokenizer.encode(query_text, add_special_tokens=False), QUERY_LEN, pad_id
    )
    stream = [token for ids in doc_ids for token in ids] + query_ids
    meta = {
        "doc_texts": list(doc_texts),
        "query_text": query_text,
        "pad_token_id": int(pad_id),
        "ids_sha256": _sha256_ids(stream),
    }
    return torch.tensor([stream], dtype=torch.long), meta


def capture_layer0_states(
    model: Any, input_ids: torch.Tensor, positions: torch.Tensor, additive: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run one ddi_qf prefill; recompute layer-0 q/k (post-RoPE) and v from the captured inputs.

    Returns q_rot, k_rot [1, H, T, D], v [1, Hkv, T, D] in the model dtype, plus metadata.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    device = next(model.parameters()).device
    captured: dict[str, tuple] = {}

    def _pre_hook(module, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    attn = model.model.layers[0].self_attn
    handle = attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids.to(device),
                attention_mask=additive.to(device)[None, None],
                position_ids=positions.to(device)[None],
                use_cache=False,
            )
    finally:
        handle.remove()
    if "args" not in captured:
        raise RuntimeError("layer-0 self_attn pre-hook never fired; forward path changed?")

    hidden = captured["args"][0] if captured["args"] else captured["kwargs"]["hidden_states"]
    position_embeddings = captured["kwargs"].get("position_embeddings")
    if position_embeddings is None:
        position_embeddings = captured["args"][1]
    cos, sin = position_embeddings

    hidden_shape = (1, hidden.shape[1], -1, attn.head_dim)
    q = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    k = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    v = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    meta = {"softmax_scale": float(attn.scaling)}
    return q.detach(), k.detach(), v.detach(), meta


def fa2_segmented(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """The exact production V61 candidate path (block-diagonal docs + query-over-all)."""
    from ddi.segmented_fa2 import _segmented_prefill

    state = {"doc_lengths": list(DOC_LENGTHS), "query_len": QUERY_LEN, "variant": VARIANT}
    return _segmented_prefill(q, k, v, state, scale).to(q.dtype)


# ---- main --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for this study (CPU cannot produce the fp64/FA2 cells)")

    from ddi.masks import build_mask, to_additive
    from ddi.model import load_model
    from ddi.positions import build_positions

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = args.output_root / f"v61_fp64_calibration_{timestamp}"
    (artifact_dir / "states").mkdir(parents=True, exist_ok=False)
    (artifact_dir / "outputs").mkdir(exist_ok=False)

    print(f"[calibrate] loading {args.checkpoint} (bf16, sdpa) ...", flush=True)
    model, tokenizer = load_model(str(args.checkpoint), torch.bfloat16, "sdpa")
    model.eval().to(args.device)

    input_ids, input_meta = build_input_ids(tokenizer, DOC_TEXTS, QUERY_TEXT)
    allow = build_mask(DOC_LENGTHS, QUERY_LEN, ANSWER_LEN, VARIANT)  # bool [T, T] on CPU
    positions = build_positions(DOC_LENGTHS, QUERY_LEN, ANSWER_LEN, LAYOUT)
    additive = to_additive(allow, torch.bfloat16)
    print(f"[calibrate] T={input_ids.shape[1]} captured states ...", flush=True)
    q, k, v, attn_meta = capture_layer0_states(model, input_ids, positions, additive)

    # Save the captured states (bf16, small) + the exact mask/positions for reproduction.
    torch.save(q.cpu(), artifact_dir / "states" / "q_rot_bf16.pt")
    torch.save(k.cpu(), artifact_dir / "states" / "k_rot_bf16.pt")
    torch.save(v.cpu(), artifact_dir / "states" / "v_bf16.pt")
    torch.save(allow, artifact_dir / "states" / "allow_bool.pt")
    torch.save(positions, artifact_dir / "states" / "positions.pt")

    scale = attn_meta["softmax_scale"]
    outputs = {
        "fp64_dense": dense_attention(q, k, v, allow, scale, torch.float64),
        "fp32_sdpa": sdpa_attention(q, k, v, allow, scale, torch.float32),
        "bf16_sdpa": sdpa_attention(q, k, v, allow, scale, torch.bfloat16),
        "bf16_fa2_segmented": fa2_segmented(q, k, v, scale),
    }
    for name, out in outputs.items():
        if not torch.isfinite(out).all():
            raise RuntimeError(f"{name} produced non-finite attention output")

    reference = outputs[REFERENCE_METHOD]
    errors = error_table(outputs, reference)
    pairwise = pairwise_argmax_disagreement(outputs)
    worst = max(errors[name]["rms_rel"] for name in GATED_METHODS)
    gate = derive_gate(worst)

    # Small fp32 outputs for later inspection: reference + the FA2 candidate.
    torch.save(reference.to(torch.float32).cpu(), artifact_dir / "outputs" / "fp64_dense_f32.pt")
    torch.save(
        outputs["bf16_fa2_segmented"].to(torch.float32).cpu(),
        artifact_dir / "outputs" / "bf16_fa2_segmented_f32.pt",
    )

    config = model.config
    summary = {
        "schema": SCHEMA,
        "generated_utc": timestamp,
        "runtime": _runtime_info(),
        "bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "checkpoint": {
            "path": str(args.checkpoint),
            "dtype": "bfloat16",
            "backend": "sdpa",
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(
                getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            ),
            "softmax_scale": scale,
        },
        "input": {
            **input_meta,
            "doc_lengths": list(DOC_LENGTHS),
            "query_len": QUERY_LEN,
            "answer_len": ANSWER_LEN,
            "layout": LAYOUT,
            "variant": VARIANT,
            "tokens": input_ids[0].tolist(),
            "positions_sha256": _sha256_tensor(positions),
            "mask_spec": {
                "implementation": "ddi.masks.build_mask",
                "variant": VARIANT,
                "doc_lengths": list(DOC_LENGTHS),
                "query_length": QUERY_LEN,
                "answer_length": ANSWER_LEN,
            },
            "mask_sha256": _sha256_tensor(allow),
        },
        "artifacts": {
            "states/q_rot_bf16.pt": _sha256_tensor(q.cpu()),
            "states/k_rot_bf16.pt": _sha256_tensor(k.cpu()),
            "states/v_bf16.pt": _sha256_tensor(v.cpu()),
            "outputs/fp64_dense_f32.pt": _sha256_tensor(reference.to(torch.float32).cpu()),
            "outputs/bf16_fa2_segmented_f32.pt": _sha256_tensor(
                outputs["bf16_fa2_segmented"].to(torch.float32).cpu()
            ),
        },
        "errors_vs_fp64_dense": errors,
        "pairwise_argmax_disagreement": pairwise,
        "gate_proposal": gate,
        "limitations": [
            "layer 0 only, one fixed input, T=320 tokens, one checkpoint",
            "argmax disagreement is over the attention-output feature dim, not the vocabulary",
            "fp64 dense removes kernel numerics but not the bf16 rounding of the captured states",
            "gate is a PROPOSAL pending owner ratification",
        ],
    }
    summary_path = artifact_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    (artifact_dir / "CALIBRATION.md").write_text(render_calibration_md(summary), encoding="utf-8")

    print(f"\n[calibrate] error vs {REFERENCE_METHOD} (attention-output level):")
    for name, row in errors.items():
        print(
            f"  {name:>20}: rms_rel={row['rms_rel']:.3e}  max_abs={row['max_abs']:.3e}  "
            f"worst_head_rms_rel={row['worst_head_rms_rel']:.3e}"
        )
    print("[calibrate] pairwise argmax disagreement:")
    for pair, rate in pairwise.items():
        print(f"  {pair:>45}: {rate:.4f}")
    print(f"[calibrate] PROPOSED gate (not ratified): rms_rel <= {gate['proposed_max']:g}")
    print(f"[calibrate] artifacts: {artifact_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
