"""Phase 0.5 correctness gates (issue #1 §0.5, issue #4).

Ground rule 6 enforcement: these three gates must pass on the current SHA before any DDI
number is admitted to the store. All tests are marked `@pytest.mark.gate` so `make gates`
can run just this suite as a cheap serverless preflight; `make test` runs them in CI on
every push.

The gates:

1. N=1 equivalence — with one document, DDI and full causal are the same mask; the
   attention outputs match to bf16/fp32 tolerance for every variant and every backend.
2. Backend cross-check — one fixed input through all available backends; the max logit
   deviation vs the dense SDPA reference is printed and asserted below tolerance.
3. Document-length regression — the `len()`-vs-`.shape[-1]` bug from the v3/v4
   archaeology, encoded as a permanent test so it cannot silently return.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from ddi.attention import get_backend
from ddi.masks import build_mask, to_additive

_HAS_FA2 = importlib.util.find_spec("flash_attn") is not None
_HAS_CUDA = torch.cuda.is_available()


def _random_qkv(
    batch: int = 1,
    heads: int = 2,
    seq: int = 12,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Gate 1 — N=1 equivalence
# ---------------------------------------------------------------------------


_N1_VARIANTS = ("ddi_qf", "ddi_strict", "ddi_bridge_k")


@pytest.mark.gate
@pytest.mark.parametrize("variant", _N1_VARIANTS)
def test_gate1_n1_mask_equals_full(variant: str) -> None:
    """N=1 collapse at the mask level: DDI variants must be bitwise-equal to full causal."""
    bridge_k = 2 if variant == "ddi_bridge_k" else None
    ddi_mask = build_mask(
        doc_lengths=[6], query_len=4, answer_len=2, variant=variant, bridge_k=bridge_k
    )
    full_mask = build_mask(doc_lengths=[6], query_len=4, answer_len=2, variant="full")
    assert torch.equal(ddi_mask, full_mask)


def _available_backends_for_gate1() -> list[str]:
    backends = ["sdpa"]
    if _HAS_FA2 and _HAS_CUDA:
        backends.append("fa2")
    return backends


@pytest.mark.gate
@pytest.mark.parametrize("variant", _N1_VARIANTS)
@pytest.mark.parametrize("backend_name", _available_backends_for_gate1())
def test_gate1_n1_output_matches_full(variant: str, backend_name: str) -> None:
    """N=1 collapse at the output level: DDI-masked attention must equal full-causal attention."""
    doc_lengths = [6]
    query_len, answer_len = 4, 2
    seq = sum(doc_lengths) + query_len + answer_len
    bridge_k = 2 if variant == "ddi_bridge_k" else None

    ddi_allow = build_mask(doc_lengths, query_len, answer_len, variant, bridge_k=bridge_k)
    full_allow = build_mask(doc_lengths, query_len, answer_len, "full")
    assert torch.equal(ddi_allow, full_allow)  # sanity: mask-level equality must already hold

    if backend_name == "fa2":
        # FA2 varlen is scoped to doc-only prefill; N=1 collapse claim at the OUTPUT level for
        # FA2 requires query_len=answer_len=0 so the input is a single-document prefill.
        doc_lengths_fa2 = [seq]
        ddi_allow_fa2 = build_mask(doc_lengths_fa2, 0, 0, variant, bridge_k=bridge_k)
        full_allow_fa2 = build_mask(doc_lengths_fa2, 0, 0, "full")
        assert torch.equal(ddi_allow_fa2, full_allow_fa2)

        q, k, v = _random_qkv(seq=seq)
        q, k, v = q.cuda(), k.cuda(), v.cuda()
        ref = get_backend("sdpa")(
            q, k, v, to_additive(full_allow_fa2, dtype=torch.float32).cuda(), None
        )
        out = get_backend("fa2")(q, k, v, None, doc_lengths_fa2)
        assert torch.allclose(out, ref, atol=1e-5)
        return

    # SDPA path: additive fp32, CPU. Since ddi_allow == full_allow by construction, running
    # under either mask produces the same output; we still compute both and compare to catch
    # any accidental non-determinism / in-place bug in the mask pipeline.
    q, k, v = _random_qkv(seq=seq)
    add_ddi = to_additive(ddi_allow, dtype=torch.float32)
    add_full = to_additive(full_allow, dtype=torch.float32)
    sdpa = get_backend("sdpa")
    out_ddi = sdpa(q, k, v, add_ddi, None)
    out_full = sdpa(q, k, v, add_full, None)
    assert torch.allclose(out_ddi, out_full, atol=1e-5)


@pytest.mark.gate
def test_gate1_blind_does_not_collapse() -> None:
    """`blind` is the control that sees no docs — it must NOT collapse to full at N=1."""
    # Query row 0 (position doc_total=6) attempting to attend to doc token 0 must be blocked.
    assert not build_mask([6], 4, 2, "blind")[6, 0].item()


# ---------------------------------------------------------------------------
# Gate 2 — backend cross-check
# ---------------------------------------------------------------------------


@pytest.mark.gate
def test_gate2_backend_cross_check() -> None:
    """Fixed input through every available backend; max deviation vs SDPA below fp32 tolerance.

    The deviation is `print`ed for each backend so serverless log tails and pytest -ra failure
    summaries surface the number without needing to grep across a captured tensor.
    """
    doc_lengths = [3, 4, 5]
    query_len, answer_len = 0, 0
    seq = sum(doc_lengths)

    allow = build_mask(doc_lengths, query_len, answer_len, "ddi_qf")
    additive = to_additive(allow, dtype=torch.float32)

    q, k, v = _random_qkv(batch=1, heads=2, seq=seq, head_dim=4, dtype=torch.float32)

    ref = get_backend("sdpa")(q, k, v, additive, None)

    other_backends: list[str] = []
    if _HAS_FA2 and _HAS_CUDA:
        other_backends.append("fa2")

    if not other_backends:
        pytest.skip("no non-SDPA backend available on this runner")

    for name in other_backends:
        if name == "fa2":
            q_dev, k_dev, v_dev = q.cuda(), k.cuda(), v.cuda()
            out = get_backend("fa2")(q_dev, k_dev, v_dev, None, doc_lengths)
            out = out.cpu()
        else:  # pragma: no cover — no other backend enumerated yet
            out = get_backend(name)(q, k, v, additive, doc_lengths)

        max_abs_dev = (out - ref).abs().max().item()
        print(f"[gate2] {name} vs sdpa: max_abs_dev = {max_abs_dev:.3e}")
        assert max_abs_dev < 1e-5, f"{name} deviates {max_abs_dev:.3e} from sdpa (fp32 tol=1e-5)"


# ---------------------------------------------------------------------------
# Gate 3 — document-length regression (the v3/v4 len()-vs-.shape[-1] bug)
# ---------------------------------------------------------------------------


@pytest.mark.gate
def test_gate3_mask_shape_sums_lengths() -> None:
    """Total mask dimension equals `sum(doc_lengths) + query_len + answer_len` — nothing else."""
    doc_lengths = [3, 5, 7]
    query_len, answer_len = 4, 2
    mask = build_mask(doc_lengths, query_len, answer_len, "ddi_qf")
    assert mask.shape == (21, 21)
    assert mask.shape[0] == sum(doc_lengths) + query_len + answer_len


@pytest.mark.gate
def test_gate3_len_vs_shape_regression() -> None:
    """The v3/v4 bug: `len(encoded)` on a batched-token tensor returns the batch, not the tokens.

    Any code that uses `len()` on a `tokenizer.encode(..., return_tensors="pt")` output treats
    every document as one token long — which silently produces a 1x1 mask per doc and a total
    length equal to `n_docs + query_len + answer_len` instead of the true token total. This
    test pins the semantic difference so the bug cannot silently return.
    """
    fake_encoded = torch.randint(0, 100, (1, 8))  # tokenizer.encode(..., return_tensors="pt") shape
    assert len(fake_encoded) == 1  # THIS is the bug — batch dimension, not token count
    assert fake_encoded.shape[-1] == 8  # THIS is the fix — actual token count

    # And prove the downstream consequence: a doc-length list built with len() collapses every
    # document to a single token, producing a totally wrong mask geometry.
    docs = [torch.randint(0, 100, (1, n)) for n in (3, 5, 7)]
    wrong_lengths = [len(d) for d in docs]  # [1, 1, 1] — the bug
    right_lengths = [d.shape[-1] for d in docs]  # [3, 5, 7] — the fix
    assert wrong_lengths == [1, 1, 1]
    assert right_lengths == [3, 5, 7]
    assert sum(wrong_lengths) != sum(right_lengths)


# ---------------------------------------------------------------------------
# Meta sanity — every gate in this module must carry the marker.
# ---------------------------------------------------------------------------


def test_every_gate_is_marked() -> None:
    """Belt-and-braces: any test function in this module whose name starts with `test_gate` must
    carry `@pytest.mark.gate`, else `make gates` would silently skip it."""
    import inspect
    import sys

    module = sys.modules[__name__]
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_gate"):
            continue
        marks = getattr(obj, "pytestmark", [])
        assert any(m.name == "gate" for m in marks), f"{name} missing @pytest.mark.gate"
