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
from ddi.prompt import encode_segments

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
    if backend == "flex":
        attn_impl = "flex"
    load_name = checkpoint_dir if checkpoint_dir is not None else model_id
    return load_model(load_name, dtype=torch_dtype, attn_implementation=attn_impl)


def encode(
    tokenizer: PreTrainedTokenizerBase,
    docs: list[str],
    query: str,
    layout: str = "sequential",
) -> BatchEncoding:
    """Encode N docs + a query into a single BatchEncoding with the chosen position layout.

    Tokenisation is delegated to `ddi.prompt.encode_segments` — the one encoder shared with
    the loss, training and perplexity paths — with no answer text, since this is the
    generation path and the answer is what we are about to produce. It takes texts rather
    than a record because callers legitimately generate from a query they built themselves
    (`ddi/run.py` already has the templated query in hand, and the mask tests drive it with
    synthetic strings); the record-level wrapper is `ddi.prompt.encode_record`.

    Documents are tokenised individually so per-document lengths survive as the mask's
    block structure; the query is concatenated after all docs. `layout` is delegated
    to `ddi.positions.build_positions`.
    """
    if layout not in ("sequential", "restart"):
        raise ValueError(f"unknown layout {layout!r}; expected 'sequential' or 'restart'")

    input_ids, doc_lengths, query_len, _ = encode_segments(tokenizer, list(docs), query)

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

    The DDI mask covers the prefill *and every decode step*. Restricting it to the
    prefill — as this function used to, following
    `phase2_train_free._generate_with_prefill_mask` — silently breaks `blind`:
    `ddi.masks.build_mask` sets `answer_sees_doc` to all-False for that variant
    (masks.py:101-102), but a decode step called without an `attention_mask` gets
    HF's default full-causal mask over the whole KV cache, so every generated token
    after the first could read the documents anyway. `blind` is the null channel
    that every parity claim is measured against, so a leaky `blind` collapses the
    `full - blind` headroom in exactly the generation metrics (`em`, `f1`,
    `presence`) the research depends on — while `loss`, which goes through a
    properly masked single forward, kept showing the real separation. That
    disagreement between the two measurement paths is what exposed the bug
    (VIBEQUEUE-2 review, issue R2).

    Returns a length-batch list; batch=1 in Phase 1.
    """
    prefill_ids = batch.input_ids.to(model.device)
    positions = batch.position_ids.to(model.device)
    prefix_len = int(prefill_ids.shape[-1])

    # The mask is indexed by (doc_lengths, query_len); the token stream is indexed by
    # position. Nothing else in this function ties the two together, so a `BatchEncoding`
    # whose block structure disagrees with its own ids — hand-built, mutated, or produced
    # by an encoder that stopped going through `ddi.prompt.encode_segments` — would slide
    # every block boundary against the tokens and still run to completion, producing
    # numbers that look like a mask effect. Both mask bugs this layer has had were silent
    # in exactly that way, so the misalignment gets an error rather than a plot.
    expected_prefix = sum(batch.doc_lengths) + batch.query_len
    if prefix_len != expected_prefix:
        raise ValueError(
            f"BatchEncoding is inconsistent: input_ids has {prefix_len} tokens but "
            f"doc_lengths={batch.doc_lengths} + query_len={batch.query_len} = "
            f"{expected_prefix}; the mask would be misaligned against the token stream"
        )

    # One allow-matrix covering prefill + the whole decode budget. Rows below
    # `prefix_len` are independent of `answer_len`, so the top-left block is exactly
    # the prefill mask we used before; rows at and beyond it are the answer rows we
    # now need one at a time.
    allow_full = build_mask(
        batch.doc_lengths,
        query_len=batch.query_len,
        answer_len=max_new_tokens,
        variant=mask_variant,  # type: ignore[arg-type]
        **(dict(mask_kwargs) if mask_kwargs else {}),
    )
    additive_full = to_additive(allow_full, dtype=model.dtype).to(model.device)
    # `.clone()` is load-bearing, not defensive copying. Slicing `additive_full` yields a
    # VIEW into the big matrix: the prefill block is non-contiguous (row stride is the full
    # width, not `prefix_len`), and each decode row is contiguous but carries a storage
    # offset of `row * width`. Neither owns an allocation whose base pointer is aligned.
    #
    # In bf16 a tensor-core GEMM wants its operands 16-byte aligned — 8 elements — so a
    # mask whose data pointer sits at an arbitrary element offset makes cuBLAS fail with
    # CUBLAS_STATUS_EXECUTION_FAILED, and every CUDA call after it reports the sticky
    # `misaligned address`. That is exactly how this presented: on 2026-08-06 all 15 runs
    # died at the first decode step, on `full` as well as the DDI variants, because the
    # pre-review code passed NO mask during decode and so never exercised this path.
    # `.contiguous()` is NOT sufficient for the decode row — it is already contiguous by
    # stride and returns self, offset and all. Only a fresh allocation fixes the base
    # pointer.
    additive = additive_full[:prefix_len, :prefix_len].clone().unsqueeze(0).unsqueeze(0)

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

        for i in range(max_new_tokens):
            tok_id = int(next_token.item())
            generated.append(tok_id)
            if eos_id is not None and tok_id == eos_id:
                break
            # This forward processes the token at absolute position prefix_len + i,
            # attending over the prefix_len + i + 1 entries already in the cache.
            kv_len = prefix_len + i + 1
            # `.clone()` before reshape: see the note on `additive` above. The row is
            # contiguous but offset, and reshape of a contiguous view is a view, so
            # without this the mask reaches cuBLAS at an unaligned address.
            step_mask = (
                additive_full[prefix_len + i, :kv_len]
                .clone()
                .reshape(1, 1, 1, kv_len)
            )
            step = model(
                input_ids=next_token,
                attention_mask=step_mask,
                past_key_values=past,
                use_cache=True,
            )
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
