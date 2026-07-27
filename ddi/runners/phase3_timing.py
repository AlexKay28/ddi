"""Phase 3 timing sweep runner — one (N, L, backend) grid cell per invocation.

Covers issues #18 (N-sweep), #19 (L-sweep), and #20 (backend cross-product): the same runner
handles all three because the difference is entirely in the config the orchestrator renders.
Every row carries the (N, L, Q) shape in `run_id`, the analytical speedup target from
RESEARCH_PLAN §7, and the mask sparsity, so Phase 3.5 (issue #21) can compute
realized-vs-analytical per cell without joining across additional metadata.

Timing primitives (`measure_prefill` / `measure_decode`) come from `ddi.timing` (issue #17);
this runner is glue: it builds the mask, times prefill + decode, attaches the analytical
target, and writes rows. It does NOT load a dataset — a timing sweep is not sensitive to
token content, only to shape and mask sparsity, so we use random token IDs.
"""

from __future__ import annotations

from typing import Any

import torch

from ddi.cli import register_runner
from ddi.masks import build_mask, to_additive
from ddi.model import load_model
from ddi.provenance import capture
from ddi.store import DEFAULT_STORE, append
from ddi.timing import append_timing, measure_decode, measure_prefill


def _analytical_speedup(variant: str, N: int, L: int, Q: int) -> float:
    """Full-causal cost / DDI cost from RESEARCH_PLAN §7. `full` returns 1.0 exactly.

    Blind / bridge_k are approximated by `full` so the column stays populated for every row —
    Phase 3.5 filters on `mask_variant` before reading this value.
    """
    T = N * L + Q
    full_flops = 0.5 * T * T
    if variant == "full":
        return 1.0
    if variant == "ddi_qf":
        our = 0.5 * N * L * L + Q * N * L + 0.5 * Q * Q
    elif variant == "ddi_strict":
        our = 0.5 * N * L * L + 0.5 * Q * Q + T
    else:
        our = full_flops
    return full_flops / our


def _sparsity(allow: torch.Tensor) -> float:
    """Fraction of full-causal attention pairs removed by the mask.

    Reported relative to the causal-pair count (not to T²) so the number lines up with the
    RESEARCH_PLAN §7 table (e.g. 0.906 at N=13, L=512, Q=64). This is what "sparsity" means
    in the paper and in the F3 realization plot.
    """
    T = allow.shape[-1]
    causal_pairs = T * (T + 1) / 2
    return 1.0 - float(allow.sum().item()) / causal_pairs


@register_runner("phase3_timing")
def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run one (N, L, backend) grid cell, one seed — see module docstring."""
    if "mask" not in config:
        raise ValueError("phase3_timing config missing required key: mask")
    mask_cfg = config["mask"]
    if "variant" not in mask_cfg:
        raise ValueError("phase3_timing config.mask missing required key: variant")

    variant: str = mask_cfg["variant"]
    bridge_k = mask_cfg.get("bridge_k")

    N = int(config["N"])
    L = int(config["L"])
    Q = int(config["Q"])
    # `answer_len` is kept in the config schema for future FTT-with-answer runs; unused here.
    _ = int(config.get("answer_len", 16))
    backend_name = config.get("backend", "sdpa")
    model_name = config["model"]
    n_repeats = int(config.get("n_repeats", 5))
    warmup = int(config.get("warmup", 3))
    n_new_tokens = int(config.get("n_new_tokens", 16))

    # WHY first: FA2 kernels are CUDA-only, and the timing sweep is meaningless on CPU with an
    # FA2 request. Failing before load_model saves a multi-GB download when the orchestrator
    # accidentally schedules an FA2 cell onto a CPU-only worker.
    if backend_name == "fa2" and not torch.cuda.is_available():
        raise RuntimeError(
            "backend='fa2' requires CUDA; torch.cuda.is_available() is False. "
            "Route this cell to a GPU worker or select backend='sdpa'."
        )

    # Honour the config's backend: HF picks its attn kernel from `attn_implementation` at
    # load time, so one runner invocation = one backend on one shape. Matches the serverless
    # "one cell" idiom and keeps the Phase 3.4 cross-product a pure orchestrator concern.
    # Tokenizer is discarded — timing does not decode text, only counts kernel time.
    model, _tokenizer = load_model(model_name, attn_implementation=backend_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    vocab_size = model.config.vocab_size
    torch.manual_seed(seed)
    doc_lengths = [L] * N
    T_prefill = N * L + Q
    prefill_ids = torch.randint(0, vocab_size, (1, T_prefill), device=model.device)

    allow = build_mask(
        doc_lengths, query_len=Q, answer_len=0, variant=variant, bridge_k=bridge_k
    )
    additive = (
        to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)
    )

    prov = capture(
        config_hash=config_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision=f"synthetic_timing_N{N}_L{L}_Q{Q}",
    )
    variant_tag = variant if bridge_k is None else f"{variant}_k{bridge_k}"
    run_id = (
        f"phase3_timing:{model_name}:N={N}:L={L}:Q={Q}:{variant_tag}:{backend_name}:"
        f"seed={seed}:{prov.git_sha[:8]}"
    )

    sparsity = _sparsity(allow)
    append(run_id, "mask_sparsity", sparsity, prov)

    ftt = measure_prefill(model, prefill_ids, additive, n_repeats=n_repeats, warmup=warmup)
    append_timing(store=DEFAULT_STORE, run_id=run_id, metric="ftt", timing=ftt, provenance=prov)

    tpot = measure_decode(
        model,
        prefill_ids,
        additive,
        n_new_tokens=n_new_tokens,
        n_repeats=n_repeats,
        warmup=warmup,
    )
    append_timing(store=DEFAULT_STORE, run_id=run_id, metric="tpot", timing=tpot, provenance=prov)

    analytical_speedup = _analytical_speedup(variant, N, L, Q)
    append(run_id, "analytical_speedup", analytical_speedup, prov)

    # Backend as a label row so Phase 3.4 can group by it without parsing run_id.
    append(run_id, "backend_used", backend_name, prov)
