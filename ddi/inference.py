"""Classic HF wrapper with DDI mask injection (issue #84, Phase 1 layer 2).

The only HF-interaction surface in the `ddi/` package: load a model, encode
N documents + a query, run a masked forward pass, or greedy-generate under a
mask variant. Every piece of the pipeline is delegated to an existing module —
`ddi.masks` (variant tensor), `ddi.positions` (layout), `ddi.model` (HF load
and mask-injecting forward), `ddi.cache` (KV cache handling via HF's own path).
No new attention logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ddi.masks import Variant, build_mask, to_additive
from ddi.model import load_model
from ddi.positions import Layout, build_positions

_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class BatchEncoding:
    """Encoded (docs + query) sample. `doc_lengths` is what mask/position builders key on."""

    input_ids: torch.Tensor  # [1, T] long
    position_ids: torch.Tensor  # [1, T] long
    doc_lengths: list[int]
    query_len: int
    layout: str


@dataclass(frozen=True)
class ForwardOutput:
    """One masked forward pass. `hidden_states` is populated when the model returns them."""

    logits: torch.Tensor  # [1, T, V]
    hidden_states: tuple[torch.Tensor, ...] | None
    mask_variant: str


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {tuple(_DTYPES)}")
    return _DTYPES[dtype]


def load(
    model_id: str,
    dtype: str = "bf16",
    backend: str = "sdpa",
    checkpoint_dir: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load an HF causal-LM + tokenizer with the chosen attention backend.

    `backend` maps directly to HF's `attn_implementation` (`sdpa`, `flash_attention_2`,
    `eager`). Delegates to `ddi.model.load_model`, which also installs the DDI mask
    patch so 4D float masks pass through `_update_causal_mask` unchanged.

    If ``checkpoint_dir`` is set, weights are loaded from that local directory
    (an HF-layout snapshot written by e.g. `ddi.finetune`) rather than the
    ``model_id`` hub repo. The `model_id` is still surfaced for provenance so a
    downstream reader knows which base architecture the checkpoint was trained
    from.
    """
    torch_dtype = _resolve_dtype(dtype)
    attn_impl = "flash_attention_2" if backend == "fa2" else backend
    load_name = checkpoint_dir if checkpoint_dir is not None else model_id
    return load_model(load_name, dtype=torch_dtype, attn_implementation=attn_impl)


def encode(
    tokenizer: PreTrainedTokenizerBase,
    docs: list[str],
    query: str,
    layout: str = "sequential",
) -> BatchEncoding:
    """Encode N docs + a query into a single BatchEncoding with the chosen position layout.

    Documents are tokenised individually so per-document lengths survive as the mask's
    block structure; the query is concatenated after all docs. `layout` is delegated
    to `ddi.positions.build_positions`.
    """
    if layout not in ("sequential", "restart"):
        raise ValueError(f"unknown layout {layout!r}; expected 'sequential' or 'restart'")

    doc_id_tensors = [
        tokenizer(doc, return_tensors="pt", add_special_tokens=False).input_ids for doc in docs
    ]
    doc_lengths = [int(t.shape[-1]) for t in doc_id_tensors]

    query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids
    query_len = int(query_ids.shape[-1])

    all_ids = doc_id_tensors + [query_ids] if doc_id_tensors else [query_ids]
    input_ids = torch.cat(all_ids, dim=-1)

    positions = build_positions(
        doc_lengths, query_len=query_len, answer_len=0, layout=layout  # type: ignore[arg-type]
    ).unsqueeze(0)

    return BatchEncoding(
        input_ids=input_ids,
        position_ids=positions,
        doc_lengths=doc_lengths,
        query_len=query_len,
        layout=layout,
    )


def _build_additive(
    doc_lengths: list[int],
    query_len: int,
    answer_len: int,
    mask_variant: str,
    mask_kwargs: dict | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a 4D additive mask via ddi.masks; mask_kwargs supplies variant-specific args."""
    kwargs = dict(mask_kwargs) if mask_kwargs else {}
    allow = build_mask(
        doc_lengths,
        query_len=query_len,
        answer_len=answer_len,
        variant=mask_variant,  # type: ignore[arg-type]
        **kwargs,
    )
    return to_additive(allow, dtype=dtype).to(device).unsqueeze(0).unsqueeze(0)


def forward_with_mask(
    model: PreTrainedModel,
    batch: BatchEncoding,
    mask_variant: str,
    mask_kwargs: dict | None = None,
) -> ForwardOutput:
    """Single forward pass under the given mask variant. Returns logits + hidden states.

    Delegates mask construction to `ddi.masks` and relies on the `_update_causal_mask`
    patch installed by `ddi.model.load_model` to route the 4D float mask through HF's
    forward unchanged. `output_hidden_states=True` is set so downstream analyses (delta,
    per-layer probes) do not need a second pass.
    """
    input_ids = batch.input_ids.to(model.device)
    positions = batch.position_ids.to(model.device)
    additive = _build_additive(
        batch.doc_lengths,
        batch.query_len,
        answer_len=0,
        mask_variant=mask_variant,
        mask_kwargs=mask_kwargs,
        dtype=model.dtype,
        device=model.device,
    )
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=additive,
            position_ids=positions,
            use_cache=False,
            output_hidden_states=True,
        )
    return ForwardOutput(
        logits=out.logits,
        hidden_states=tuple(out.hidden_states) if getattr(out, "hidden_states", None) else None,
        mask_variant=mask_variant,
    )


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch: BatchEncoding,
    mask_variant: str,
    max_new_tokens: int = 64,
    mask_kwargs: dict | None = None,
) -> list[str]:
    """Generate text under the given mask variant. Returns decoded strings.

    Same shape as `phase2_train_free._generate_with_prefill_mask`: the DDI mask is
    applied only to the prefill; each decode step then uses HF's own KV-cached path
    (no hand-rolled attention). Returns a length-batch list; batch=1 in Phase 1.
    """
    prefill_ids = batch.input_ids.to(model.device)
    positions = batch.position_ids.to(model.device)
    additive = _build_additive(
        batch.doc_lengths,
        batch.query_len,
        answer_len=0,
        mask_variant=mask_variant,
        mask_kwargs=mask_kwargs,
        dtype=model.dtype,
        device=model.device,
    )
    eos_id = tokenizer.eos_token_id
    generated: list[int] = []

    with torch.no_grad():
        out = model(
            input_ids=prefill_ids,
            attention_mask=additive,
            position_ids=positions,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)

        for _ in range(max_new_tokens):
            tok_id = int(next_token.item())
            generated.append(tok_id)
            if eos_id is not None and tok_id == eos_id:
                break
            step = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = step.past_key_values
            next_token = step.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)

    return [tokenizer.decode(generated, skip_special_tokens=True).strip()]


# Re-exports so a caller can `from ddi.inference import Variant, Layout` without a
# second import from the delegated modules.
__all__ = [
    "BatchEncoding",
    "ForwardOutput",
    "Layout",
    "Variant",
    "encode",
    "forward_with_mask",
    "generate",
    "load",
]
