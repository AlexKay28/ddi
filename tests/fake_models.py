"""A pure-torch causal-LM stand-in that records every call made to it.

Importable from any test module; imports `torch` and nothing else. That constraint is the
point, not tidiness: the suites that cover the real decode loops (`test_inference`,
`test_model`) need `transformers` to be *collected* and HF Hub access to run, so on an
ordinary dev host they report "skipped" and a decode loop that attends to everything looks
exactly like one that does not. Both mask-alignment bugs shipped through that gap. A fake
that records shapes, dtypes and storage offsets pins the contract without either dependency.

The recorded `storage_offset` is not bookkeeping either — it is the failure signature of the
2026-08-06 outage (see the post-mortem in `ddi.inference.generate`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch


@dataclass(frozen=True)
class TensorRecord:
    """What a tensor looked like at the moment it was handed to the model."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    storage_offset: int
    is_contiguous: bool
    values: torch.Tensor  # detached clone, so later in-place edits cannot rewrite history

    @classmethod
    def of(cls, t: torch.Tensor) -> TensorRecord:
        return cls(
            shape=tuple(t.shape),
            dtype=t.dtype,
            storage_offset=t.storage_offset(),
            is_contiguous=t.is_contiguous(),
            values=t.detach().clone(),
        )


@dataclass(frozen=True)
class RecordedCall:
    """One `model(...)` call. `is_decode` is True for a single-token step."""

    input_ids: TensorRecord
    attention_mask: TensorRecord | None
    position_ids: TensorRecord | None
    past_len: int | None
    use_cache: bool
    extra_kwargs: tuple[str, ...] = ()

    @property
    def is_decode(self) -> bool:
        return self.input_ids.shape[-1] == 1 and self.past_len is not None

    @property
    def kv_len(self) -> int | None:
        """Cache length this call attends over, as claimed by the mask it was given."""
        if self.attention_mask is None:
            return None
        return int(self.attention_mask.shape[-1])


@dataclass
class FakeCache:
    """Opaque `past_key_values` stand-in — only its length is ever inspected."""

    length: int = 0

    def __len__(self) -> int:
        return self.length

    def get_seq_length(self, layer_idx: int = 0) -> int:  # noqa: ARG002 — HF-shaped accessor
        return self.length


class RecordingCausalLM(torch.nn.Module):
    """`nn.Module` with an HF-shaped `forward`, recording every argument it is given.

    Returns fixed-shape logits whose argmax is always `next_token_id`, so the decode loop
    under test is deterministic and never wanders into an eos.
    """

    def __init__(
        self,
        vocab_size: int = 16,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        next_token_id: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.next_token_id = next_token_id
        self.calls: list[RecordedCall] = []
        self._dtype = dtype
        self._device = torch.device(device)

    # `device` / `dtype` are read off real HF models by `ddi.inference`; mirror them.
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def decode_calls(self) -> list[RecordedCall]:
        return [c for c in self.calls if c.is_decode]

    @property
    def prefill_calls(self) -> list[RecordedCall]:
        return [c for c in self.calls if not c.is_decode]

    def reset(self) -> None:
        self.calls.clear()

    def forward(  # noqa: PLR0913 — mirrors the HF signature the code under test calls
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        **kw: object,
    ) -> SimpleNamespace:
        past_len = None
        if past_key_values is not None:
            past_len = len(past_key_values) if hasattr(past_key_values, "__len__") else 0

        self.calls.append(
            RecordedCall(
                input_ids=TensorRecord.of(input_ids),
                attention_mask=(
                    None if attention_mask is None else TensorRecord.of(attention_mask)
                ),
                position_ids=None if position_ids is None else TensorRecord.of(position_ids),
                past_len=past_len,
                use_cache=bool(use_cache),
                extra_kwargs=tuple(sorted(kw)),
            )
        )

        seq_len = int(input_ids.shape[-1])
        logits = torch.zeros(1, seq_len, self.vocab_size, dtype=torch.float32)
        logits[..., self.next_token_id] = 1.0
        cache = FakeCache((past_len or 0) + seq_len)
        return SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=None)


class SlowCausalLM(RecordingCausalLM):
    """`RecordingCausalLM` whose forward costs `ms_per_input_token` per token it is fed.

    A prefill of `T` tokens therefore costs `T * ms_per_input_token` while a decode step — one
    token against the cache — costs `ms_per_input_token` flat, independent of `T`. That is the
    separation a timing test needs: sweeping the prompt length moves the prefill cost and nothing
    else, so any `tpot` that tracks it is reporting prefill time under a decode label.

    `time.sleep` rather than real arithmetic on purpose — the quantity under test is which region
    the timer brackets, and a sleep makes the two regions' costs exactly known instead of merely
    large. Keep `ms_per_input_token` at 1.0 or above: below roughly 0.1 ms the scheduler's wakeup
    granularity dominates and the arms stop being distinguishable.
    """

    def __init__(self, *, ms_per_input_token: float = 1.0, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.ms_per_input_token = float(ms_per_input_token)

    def forward(  # noqa: PLR0913 — mirrors the HF signature it delegates to
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        **kw: object,
    ) -> SimpleNamespace:
        time.sleep(int(input_ids.shape[-1]) * self.ms_per_input_token / 1000.0)
        return super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kw,
        )


@dataclass
class RecordingTokenizer:
    """Minimal tokenizer surface: `eos_token_id` + `decode`, plus word-level encoding.

    `eos_token_id` defaults to a token the model never emits, so `generate` runs its full
    budget instead of stopping early and hiding un-exercised decode steps.
    """

    eos_token_id: int | None = 0
    decoded: list[list[int]] = field(default_factory=list)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:  # noqa: ARG002
        self.decoded.append(list(ids))
        return " ".join(str(i) for i in ids)

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",  # noqa: ARG002
        add_special_tokens: bool = False,  # noqa: ARG002
    ) -> SimpleNamespace:
        ids = torch.tensor([[len(w) for w in text.split()]], dtype=torch.long)
        return SimpleNamespace(input_ids=ids)
