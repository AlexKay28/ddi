"""V71 serving paths: precomputed-layout request execution for the trained contract.

The V70 harness callables (``independent_forward_from_entries`` /
``generate_independent_ids``) rebuild the O(P^2) Python layout and its additive
mask tensors on every call and every decode step; at serving shapes that
overhead dominates the model work. This module precomputes, once per unit and
untimed (mirroring arm A's untimed batch construction), every structure the
request needs, then executes the identical model calls.

Numerics claim: the executed model calls receive identical ``input_ids``,
``attention_mask`` values, ``position_ids`` and ``cache_position`` as the V70
harness path — the precomputed rows are slices of a layout with the maximum
answer budget whose prefix rows are identical to the no-answer layout (prefix
rows never attend later answer tokens; asserted in tests). Byte-identity of
the fast path against the V70 path is asserted on the real model in the V71
smoke and guarded per repeat by the CP≡BP gate during the sweep.
"""

from __future__ import annotations

from typing import Any

import torch

from ddi.link_token_contract import build_request_layout
from scripts.v70_harness import (
    EncodedRequest,
    _assemble_with_grad,
    layout_tensors,
)


class PrecomputedRequest:
    """Everything a trained-contract request needs, built once per unit."""

    def __init__(
        self,
        model: Any,
        contract: Any,
        request: EncodedRequest,
        *,
        k: int,
        max_answer_tokens: int = 64,
    ):
        if max_answer_tokens < 1:
            raise ValueError("max_answer_tokens must be >= 1")
        self.request = request
        self.k = k
        self.device = model.device
        self.dtype = model.dtype

        self.layout, self.ids, self.mask, self.positions = self._prefill_structures(
            model, contract, request, k=k
        )
        self.doc_total = sum(len(document.token_ids) for document in request.documents)
        self.prefix_len = len(self.layout.physical_token_ids)

        dummy_answer = tuple(1 for _ in range(max_answer_tokens))
        full_layout = build_request_layout(
            contract, request.documents, request.query_token_ids, dummy_answer, k=k
        )
        _full_ids, full_mask, full_positions = layout_tensors(
            full_layout, self.dtype, self.device
        )
        self._prefix_rows_identical(full_mask, full_positions, contract, request, k=k)

        self.step_masks = [
            full_mask[
                :,
                :,
                self.prefix_len + index : self.prefix_len + index + 1,
                : self.prefix_len + index + 1,
            ]
            for index in range(max_answer_tokens)
        ]
        self.step_positions = [
            full_positions[:, self.prefix_len + index : self.prefix_len + index + 1]
            for index in range(max_answer_tokens)
        ]
        self._build_full_causal_structures(contract, request, k=k)

    def _build_full_causal_structures(
        self, contract: Any, request: EncodedRequest, *, k: int
    ) -> None:
        """Arm-DP structures: trained full-causal no-link monolithic prefill."""
        mono_layout = build_request_layout(
            contract, request.documents, request.query_token_ids, (), k=k
        )
        mono_ids, _unused, mono_positions = layout_tensors(
            mono_layout, self.dtype, self.device
        )
        length = mono_ids.shape[1]
        causal = torch.ones((length, length), dtype=torch.bool, device=self.device).tril()
        self.dc_ids = mono_ids
        self.dc_positions = mono_positions
        self.dc_mask = torch.zeros_like(causal, dtype=self.dtype).masked_fill_(
            ~causal, -torch.inf
        )[None, None]
        self.dc_len = length

    def _prefill_structures(
        self, model: Any, contract: Any, request: EncodedRequest, *, k: int
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = build_request_layout(
            contract, request.documents, request.query_token_ids, (), k=k
        )
        ids, mask, positions = layout_tensors(layout, model.dtype, model.device)
        return layout, ids, mask, positions

    def _prefix_rows_identical(
        self,
        full_mask: torch.Tensor,
        full_positions: torch.Tensor,
        contract: Any,
        request: EncodedRequest,
        *,
        k: int,
    ) -> None:
        """Prefix mask/position rows must not depend on the answer budget."""
        if (
            not torch.equal(self.mask, full_mask[:, :, : self.prefix_len, : self.prefix_len])
            or not torch.equal(self.positions, full_positions[:, : self.prefix_len])
        ):
            raise ValueError(
                "prefixed layout disagrees with the no-answer layout; "
                "precomputed serving rows would not mirror the V70 path"
            )

    def step(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(attention_mask row, position_ids row, cache_position) for decode step."""
        if not 0 <= index < len(self.step_masks):
            raise ValueError(f"decode step {index} outside the precomputed budget")
        cache_position = torch.tensor(
            [self.prefix_len + index], device=self.device
        )
        return self.step_masks[index], self.step_positions[index], cache_position


@torch.no_grad()
def ttft_fast(
    model: Any,
    request: EncodedRequest,
    entries: list[Any],
    pre: PrecomputedRequest,
) -> int:
    """First predicted token via the precomputed structures (timed callable)."""
    cache = _assemble_with_grad(entries, pre.device)
    out = _prefill_forward(model, pre, cache)
    return int(out.logits[0, -1].argmax())


def _prefill_forward(model: Any, pre: PrecomputedRequest, cache: Any) -> Any:
    return model(
        input_ids=pre.ids[:, pre.doc_total : pre.prefix_len],
        attention_mask=pre.mask[:, :, pre.doc_total : pre.prefix_len, : pre.prefix_len],
        position_ids=pre.positions[:, pre.doc_total : pre.prefix_len],
        cache_position=torch.arange(pre.doc_total, pre.prefix_len, device=pre.device),
        past_key_values=cache,
        use_cache=True,
    )


@torch.no_grad()
def generate_fast(
    model: Any,
    request: EncodedRequest,
    entries: list[Any],
    pre: PrecomputedRequest,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    """Fixed-work greedy decode mirroring the V70 ``_decode_from_prefill`` numerics."""
    if max_new_tokens < 0 or request.answer_token_ids:
        raise ValueError("generation requires nonnegative budget and no supplied answer")
    if max_new_tokens > len(pre.step_masks):
        raise ValueError("decode budget exceeds the precomputed answer structures")
    if max_new_tokens == 0:
        return []
    cache = _assemble_with_grad(entries, pre.device)
    out = _prefill_forward(model, pre, cache)
    next_token = out.logits[:, -1].argmax(-1)
    generated: list[int] = []
    for index in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == eos_token_id or index + 1 == max_new_tokens:
            break
        step_mask, step_positions, step_cache_position = pre.step(index)
        out = model(
            input_ids=torch.tensor([[token_id]], device=pre.device),
            attention_mask=step_mask,
            position_ids=step_positions,
            cache_position=step_cache_position,
            past_key_values=out.past_key_values,
            use_cache=True,
        )
        next_token = out.logits[:, -1].argmax(-1)
    return generated


@torch.no_grad()
def dp_ttft_fast(model: Any, pre: PrecomputedRequest) -> Any:
    """Arm-DP: trained full-causal no-link prefill via precomputed structures."""
    return model(
        input_ids=pre.dc_ids,
        attention_mask=pre.dc_mask,
        position_ids=pre.dc_positions,
        cache_position=torch.arange(pre.dc_len, device=pre.device),
        use_cache=True,
    )
