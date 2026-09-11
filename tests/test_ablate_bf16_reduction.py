from types import SimpleNamespace

import pytest
import torch

import scripts.ablate_bf16_reduction as ablation
from scripts.ablate_bf16_reduction import (
    bf16_reduction_flag,
    capture_projection,
    diagnose,
    set_bf16_reduction,
    traced_prefill,
)


def test_bf16_flag_helper_records_unavailable_and_transition():
    unavailable = SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()))
    )
    assert bf16_reduction_flag(unavailable) == {"available": False, "value": None}
    assert set_bf16_reduction(False, unavailable)["unavailable"]
    matmul = SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
    fake = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)))
    transition = set_bf16_reduction(False, fake)
    assert transition["before"]["value"] is True
    assert transition["after"]["value"] is False


def test_projection_hook_records_without_replacing_live_output():
    trace = {}
    output = torch.tensor([1.0])

    assert capture_projection(trace, "q_proj")(None, None, output) is None
    assert torch.equal(trace["q_proj"], output)
    assert trace["q_proj"].device.type == "cpu"


def test_traced_prefill_uses_one_model_device_for_rope_inputs(monkeypatch):
    """A changing device lookup must not make Qwen's RoPE cos/sin CPU-only."""

    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = None

        def forward(self, hidden_states, position_ids):
            self.inputs = (hidden_states, position_ids)
            assert hidden_states.device == position_ids.device
            return hidden_states, hidden_states

    class Model:
        dtype = torch.float32

        def __init__(self):
            self._parameter = torch.nn.Parameter(torch.ones(1))
            attention = SimpleNamespace(
                q_proj=torch.nn.Identity(),
                k_proj=torch.nn.Identity(),
                v_proj=torch.nn.Identity(),
            )
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(self_attn=attention)], rotary_emb=Rotary()
            )

        @property
        def device(self):
            return torch.device("meta")

        def parameters(self):
            return iter([self._parameter])

        def __call__(self, *, input_ids, position_ids, attention_mask, **kwargs):
            assert input_ids.device == position_ids.device == attention_mask.device
            hidden_states = torch.zeros((*input_ids.shape, 1), device=input_ids.device)
            self.model.rotary_emb(hidden_states, position_ids)
            return SimpleNamespace()

    model = Model()
    # The regression mock needs only the mask/position helpers, not the optional
    # full model stack imported by runtime_dependencies().
    monkeypatch.setattr(ablation, "CacheEntry", object, raising=False)
    monkeypatch.setattr(
        ablation,
        "build_positions",
        lambda lengths, query_length, *_: torch.arange(sum(lengths) + query_length),
        raising=False,
    )
    monkeypatch.setattr(
        ablation,
        "build_mask",
        lambda lengths, query_length, *_: torch.ones(
            sum(lengths) + query_length, sum(lengths) + query_length
        ),
        raising=False,
    )
    monkeypatch.setattr(ablation, "to_additive", lambda mask, dtype: mask.to(dtype), raising=False)
    traced_prefill(model, [2, 3, 4], [3], 0)
    hidden_states, position_ids = model.model.rotary_emb.inputs
    assert hidden_states.device == position_ids.device == torch.device("cpu")


def test_tiny_qwen_compares_projections_logits_and_disk_controls(tmp_path):
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        pytest.skip(f"tiny Qwen unavailable in this CPU environment: {error}")
    from ddi.model import patch_model

    torch.manual_seed(58)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = patch_model(Qwen2ForCausalLM(config).eval())
    report = diagnose(model, [[2, 3, 4], [5, 6, 7]], [8, 9], tmp_path)
    assert {row["operation"] for row in report["layer0_preassembly"]} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "k_post_rope",
        "v_post_rope",
    }
    assert report["query_logits"]["A_vs_B"]["all"]["allclose_1e4"]
    assert report["query_logits"]["B_vs_C"]["all"]["equal"]
    assert all(row["equal"] for checks in report["assembly"].values() for row in checks)
    assert all(
        step["comparisons"]["B_vs_C"]["equal"] for step in report["teacher_forced_greedy"]["steps"]
    )
    assert report["exact_gate"] == {"atol": 1e-4, "rtol": 1e-4, "relaxed_gate_set": False}
