"""Every decode step must carry its own row of the DDI mask.

A decode step called with no `attention_mask` gets HF's default full-causal mask over the whole
KV cache. Nothing crashes, nothing warns, and the numbers still look plausible — `full`, `ddi_qf`
and `blind` simply become the same computation from the first generated token onward. That is
worth one class of bug in quality (`ddi.inference.generate`, fixed, post-mortem in its docstring)
and one in speed (`ddi.timing.measure_decode`, fixed here): a `tpot` sweep over `masks:` where
every row measures the same thing.

The suites that would have caught either need `transformers` to be collected and HF Hub access
to run, so on a dev host they skip and the gap stays open. These tests use the pure-torch
recording model in `tests.fake_models` instead, derive what the mask *should* say straight from
`ddi.masks.build_mask`, and therefore fail if the loop ever stops passing a mask, passes the
wrong row, or passes an unaligned one.
"""

from __future__ import annotations

import pytest
import torch

from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from ddi.timing import measure_decode
from tests.fake_models import RecordingCausalLM, RecordingTokenizer

DOC_LENGTHS = [3, 4, 2]
QUERY_LEN = 5
PREFIX_LEN = sum(DOC_LENGTHS) + QUERY_LEN
BUDGET = 6
DTYPE = torch.float32

VARIANTS = ["full", "ddi_qf", "ddi_strict", "blind"]


def _prompt_ids() -> torch.Tensor:
    return torch.zeros(1, PREFIX_LEN, dtype=torch.long)


def _expected_additive(variant: str, answer_len: int) -> torch.Tensor:
    """The mask the model *should* see, built from ddi.masks — never from the code under test."""
    allow = build_mask(
        DOC_LENGTHS,
        query_len=QUERY_LEN,
        answer_len=answer_len,
        variant=variant,  # type: ignore[arg-type]
    )
    return to_additive(allow, dtype=DTYPE)


def _prefill_mask(variant: str) -> torch.Tensor:
    """The [1, 1, T, T] prefill mask `ddi.run._timing_cell` builds and passes in."""
    return _expected_additive(variant, answer_len=0).unsqueeze(0).unsqueeze(0)


def _run_measure_decode(variant: str, mask: torch.Tensor | None = None) -> RecordingCausalLM:
    model = RecordingCausalLM()
    measure_decode(
        model,
        _prompt_ids(),
        _prefill_mask(variant) if mask is None else mask,
        n_new_tokens=BUDGET,
        n_repeats=1,
        warmup=0,
    )
    return model


# ---- invariant 1: every decode step is masked, over a cache that grows by one ---------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_measure_decode_masks_every_step(variant: str) -> None:
    model = _run_measure_decode(variant)
    steps = model.decode_calls
    assert len(steps) == BUDGET

    for i, call in enumerate(steps):
        assert call.attention_mask is not None, (
            f"decode step {i} ran unmasked — HF will fill in a full-causal mask over the "
            f"whole KV cache and {variant!r} silently becomes 'full'"
        )
        assert call.attention_mask.shape == (1, 1, 1, PREFIX_LEN + i + 1)


@pytest.mark.parametrize("variant", VARIANTS)
def test_measure_decode_kv_len_grows_by_one_per_step(variant: str) -> None:
    kv_lens = [c.kv_len for c in _run_measure_decode(variant).decode_calls]
    assert kv_lens == [PREFIX_LEN + i + 1 for i in range(BUDGET)]


def test_measure_decode_prefill_is_masked() -> None:
    call = _run_measure_decode("ddi_qf").prefill_calls[0]
    assert call.attention_mask is not None
    assert call.attention_mask.shape == (1, 1, PREFIX_LEN, PREFIX_LEN)
    assert torch.equal(
        call.attention_mask.values[0, 0], _expected_additive("ddi_qf", answer_len=0)
    )


# ---- invariant 2: the mask at step i is the right row, not merely non-None ------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_measure_decode_step_masks_are_the_build_mask_rows(variant: str) -> None:
    expected = _expected_additive(variant, answer_len=BUDGET)
    steps = _run_measure_decode(variant).decode_calls

    for i, call in enumerate(steps):
        kv_len = PREFIX_LEN + i + 1
        want = expected[PREFIX_LEN + i, :kv_len].reshape(1, 1, 1, kv_len)
        assert call.attention_mask is not None
        assert torch.equal(call.attention_mask.values, want), f"wrong mask row at decode step {i}"


def test_measure_decode_blind_never_lets_the_answer_see_a_document() -> None:
    """`blind` is the null channel every parity claim is measured against.

    Stated separately from the row-equality test above because this is the property the
    research depends on, and a future mask refactor that quietly re-enabled doc columns for
    the answer would still be "the row build_mask emits".
    """
    doc_total = sum(DOC_LENGTHS)
    for call in _run_measure_decode("blind").decode_calls:
        assert call.attention_mask is not None
        row = call.attention_mask.values.flatten()
        fmin = torch.finfo(row.dtype).min
        assert (row[:doc_total] <= fmin).all(), "blind decode can read docs"
        assert not (row[doc_total:] <= fmin).any()


def test_measure_decode_full_lets_the_answer_see_everything() -> None:
    for call in _run_measure_decode("full").decode_calls:
        assert call.attention_mask is not None
        vals = call.attention_mask.values
        assert not (vals <= torch.finfo(vals.dtype).min).any()


def test_measure_decode_variants_do_not_all_agree() -> None:
    """The whole point: masks that differ must reach the model differently.

    With the pre-fix loop every variant produced the identical (absent) decode mask, so a
    `tpot` sweep over `masks: [full, ddi_qf, blind]` compared a computation with itself.
    """
    per_variant = {
        v: [
            c.attention_mask.values
            for c in _run_measure_decode(v).decode_calls
            if c.attention_mask is not None
        ]
        for v in ("full", "blind")
    }
    assert not any(
        torch.equal(a, b) for a, b in zip(per_variant["full"], per_variant["blind"], strict=True)
    )


# ---- invariant 3: nothing reaches the model at a storage offset -----------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_measure_decode_hands_over_base_aligned_tensors(variant: str) -> None:
    """A bf16 tensor-core GEMM needs 16-byte-aligned operands; a sliced mask row is contiguous
    but offset, and `.contiguous()` returns it unchanged. See the post-mortem in
    `ddi.inference.generate` and `tests/test_inference_mask_alignment.py`."""
    for n, call in enumerate(_run_measure_decode(variant).calls):
        for name, rec in (
            ("input_ids", call.input_ids),
            ("attention_mask", call.attention_mask),
            ("position_ids", call.position_ids),
        ):
            if rec is None:
                continue
            assert rec.storage_offset == 0, f"call {n}: {name} points into another allocation"
            assert rec.is_contiguous, f"call {n}: {name} is not contiguous"


# ---- invariant 4: the loop runs exactly the requested budget --------------------------------


@pytest.mark.parametrize(("warmup", "n_repeats"), [(0, 1), (1, 3), (2, 2)])
def test_measure_decode_call_count(warmup: int, n_repeats: int) -> None:
    model = RecordingCausalLM()
    measure_decode(
        model,
        _prompt_ids(),
        _prefill_mask("ddi_qf"),
        n_new_tokens=BUDGET,
        n_repeats=n_repeats,
        warmup=warmup,
    )
    iterations = warmup + n_repeats
    assert len(model.decode_calls) == BUDGET * iterations
    assert len(model.prefill_calls) == iterations


# ---- the two accepted mask shapes must agree ------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_prefill_only_and_full_masks_produce_identical_calls(variant: str) -> None:
    """Passing the prebuilt `answer_len=n_new_tokens` matrix is the explicit form; passing the
    prefill-only mask (what `_timing_cell` does today) reconstructs it. They must not diverge."""
    from_prefill = _run_measure_decode(variant)
    from_full = _run_measure_decode(variant, mask=_expected_additive(variant, answer_len=BUDGET))

    assert len(from_prefill.calls) == len(from_full.calls)
    for a, b in zip(from_prefill.calls, from_full.calls, strict=True):
        assert (a.attention_mask is None) == (b.attention_mask is None)
        if a.attention_mask is not None and b.attention_mask is not None:
            assert torch.equal(a.attention_mask.values, b.attention_mask.values)


# ---- position ids: the same either-width contract as the mask -------------------------------
#
# The timing path carried no `position_ids` at all while `ddi.inference.generate` passes them, so
# the sweep could not exercise the `restart` layout axis: every timing cell silently ran HF's
# default sequential positions. Shape-identical, so no published number moved — but `restart` is
# what makes a document's KV block a pure function of its content (positions.py), and it was
# unmeasurable here.


@pytest.mark.parametrize("layout", ["sequential", "restart"])
def test_measure_decode_forwards_the_right_position_per_step(layout: str) -> None:
    expected = build_positions(DOC_LENGTHS, QUERY_LEN, BUDGET, layout)  # type: ignore[arg-type]
    model = RecordingCausalLM()
    measure_decode(
        model,
        _prompt_ids(),
        _prefill_mask("ddi_qf"),
        n_new_tokens=BUDGET,
        position_ids=expected[:PREFIX_LEN].unsqueeze(0),
        n_repeats=1,
        warmup=0,
    )

    prefill = model.prefill_calls[0]
    assert prefill.position_ids is not None
    assert torch.equal(prefill.position_ids.values, expected[:PREFIX_LEN].unsqueeze(0))

    for i, call in enumerate(model.decode_calls):
        assert call.position_ids is not None, f"decode step {i} got no position_ids"
        want = expected[PREFIX_LEN + i].reshape(1, 1)
        assert torch.equal(call.position_ids.values, want), f"wrong position at decode step {i}"


@pytest.mark.parametrize("layout", ["sequential", "restart"])
def test_prompt_only_positions_reconstruct_the_full_vector(layout: str) -> None:
    """Handing over the prompt-only positions must equal handing over the full ones.

    `_decode_positions` extends the prompt vector by continuing from its last entry. That is exact
    for both layouts because `build_positions` writes the query+answer tail as a single contiguous
    `arange` — but it is an assumption about another module, so it gets pinned rather than trusted.
    """
    full = build_positions(DOC_LENGTHS, QUERY_LEN, BUDGET, layout)  # type: ignore[arg-type]
    prompt_only = build_positions(DOC_LENGTHS, QUERY_LEN, 0, layout)  # type: ignore[arg-type]

    def _run(positions: torch.Tensor) -> RecordingCausalLM:
        model = RecordingCausalLM()
        measure_decode(
            model,
            _prompt_ids(),
            _prefill_mask("ddi_qf"),
            n_new_tokens=BUDGET,
            position_ids=positions.unsqueeze(0),
            n_repeats=1,
            warmup=0,
        )
        return model

    for a, b in zip(_run(prompt_only).calls, _run(full).calls, strict=True):
        assert a.position_ids is not None and b.position_ids is not None
        assert torch.equal(a.position_ids.values, b.position_ids.values)


def test_measure_decode_positions_are_base_aligned() -> None:
    """Same allocation hazard as the mask rows — a sliced position row is offset into the vector."""
    model = RecordingCausalLM()
    positions = build_positions(DOC_LENGTHS, QUERY_LEN, BUDGET, "restart")
    measure_decode(
        model,
        _prompt_ids(),
        _prefill_mask("ddi_qf"),
        n_new_tokens=BUDGET,
        position_ids=positions.unsqueeze(0),
        n_repeats=1,
        warmup=0,
    )
    for n, call in enumerate(model.calls):
        assert call.position_ids is not None
        assert call.position_ids.storage_offset == 0, f"call {n}: position_ids is offset"
        assert call.position_ids.is_contiguous


def test_measure_decode_rejects_a_mask_of_the_wrong_width() -> None:
    model = RecordingCausalLM()
    bad = torch.zeros(PREFIX_LEN + 1, PREFIX_LEN + 1, dtype=DTYPE)
    with pytest.raises(ValueError, match="matches neither"):
        measure_decode(model, _prompt_ids(), bad, n_new_tokens=BUDGET, n_repeats=1, warmup=0)


# ---- the sibling loop: ddi.inference.generate ------------------------------------------------
#
# `generate` was fixed first and has been unpinned ever since (its own tests need transformers
# plus Hub access). The same recording model covers it, at the cost of an import guard: the
# module imports `transformers` at top level.


def _generate():
    pytest.importorskip("transformers", reason="ddi.inference imports transformers at module scope")
    from ddi.inference import BatchEncoding, generate

    return BatchEncoding, generate


def _batch(BatchEncoding):  # noqa: N803 — it is a class
    return BatchEncoding(
        input_ids=_prompt_ids(),
        position_ids=torch.arange(PREFIX_LEN, dtype=torch.long).unsqueeze(0),
        doc_lengths=list(DOC_LENGTHS),
        query_len=QUERY_LEN,
        layout="sequential",
    )


def _batch_for_layout(BatchEncoding, layout: str):  # noqa: N803 — it is a class
    return BatchEncoding(
        input_ids=_prompt_ids(),
        position_ids=build_positions(DOC_LENGTHS, QUERY_LEN, 0, layout).unsqueeze(0),  # type: ignore[arg-type]
        doc_lengths=list(DOC_LENGTHS),
        query_len=QUERY_LEN,
        layout=layout,
    )


def _run_generate(variant: str) -> RecordingCausalLM:
    BatchEncoding, generate = _generate()
    model = RecordingCausalLM()
    generate(model, RecordingTokenizer(), _batch(BatchEncoding), variant, max_new_tokens=BUDGET)
    return model


@pytest.mark.parametrize("variant", VARIANTS)
def test_generate_masks_every_decode_step(variant: str) -> None:
    steps = _run_generate(variant).decode_calls
    assert len(steps) == BUDGET
    for i, call in enumerate(steps):
        assert call.attention_mask is not None, f"generate decode step {i} ran unmasked"
        assert call.attention_mask.shape == (1, 1, 1, PREFIX_LEN + i + 1)


@pytest.mark.parametrize("variant", VARIANTS)
def test_generate_step_masks_are_the_build_mask_rows(variant: str) -> None:
    expected = _expected_additive(variant, answer_len=BUDGET)
    for i, call in enumerate(_run_generate(variant).decode_calls):
        kv_len = PREFIX_LEN + i + 1
        want = expected[PREFIX_LEN + i, :kv_len].reshape(1, 1, 1, kv_len)
        assert call.attention_mask is not None
        assert torch.equal(call.attention_mask.values, want), f"wrong mask row at decode step {i}"


def test_generate_blind_never_lets_the_answer_see_a_document() -> None:
    doc_total = sum(DOC_LENGTHS)
    for call in _run_generate("blind").decode_calls:
        assert call.attention_mask is not None
        row = call.attention_mask.values.flatten()
        fmin = torch.finfo(row.dtype).min
        assert (row[:doc_total] <= fmin).all(), "blind decode can read docs"


@pytest.mark.parametrize("layout", ["sequential", "restart"])
def test_generate_forwards_the_right_position_per_step(layout: str) -> None:
    BatchEncoding, generate = _generate()
    model = RecordingCausalLM()
    batch = _batch_for_layout(BatchEncoding, layout)
    generate(model, RecordingTokenizer(), batch, "ddi_qf", max_new_tokens=BUDGET)

    expected = build_positions(DOC_LENGTHS, QUERY_LEN, BUDGET, layout)  # type: ignore[arg-type]
    for i, call in enumerate(model.decode_calls):
        assert call.position_ids is not None, f"decode step {i} got no position_ids"
        want = expected[PREFIX_LEN + i].reshape(1, 1)
        assert torch.equal(call.position_ids.values, want), f"wrong position at decode step {i}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_generate_hands_over_base_aligned_tensors(variant: str) -> None:
    for n, call in enumerate(_run_generate(variant).calls):
        for name, rec in (
            ("input_ids", call.input_ids),
            ("attention_mask", call.attention_mask),
            ("position_ids", call.position_ids),
        ):
            if rec is None:
                continue
            assert rec.storage_offset == 0, f"call {n}: {name} points into another allocation"
            assert rec.is_contiguous, f"call {n}: {name} is not contiguous"
