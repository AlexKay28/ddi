import pytest
import torch

try:
    from scripts.validate_canonical_smoke import diagnose, fixed_cases
except ModuleNotFoundError as error:
    # This sandbox has an incompatible torchvision/Transformers optional stack; the
    # parent environment runs the real tiny-Qwen test below.
    if "PreTrainedModel" not in str(error):
        raise
    pytest.skip(f"canonical-cache runtime unavailable: {error}", allow_module_level=True)


def _model(backend: str):
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from ddi.model import patch_model
    except Exception as error:  # Local optional vision stack can make Qwen lazy import unavailable.
        pytest.skip(f"tiny Qwen runtime unavailable: {error}")
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
        pad_token_id=0,
    )
    config._attn_implementation = backend
    return patch_model(Qwen2ForCausalLM(config).eval())


def test_fixed_cases_are_token_native_and_complete():
    cases = fixed_cases([list(range(2, 34)), list(range(34, 66))], list(range(66, 74)))
    assert [case["name"] for case in cases] == [
        "base_cold",
        "base_warm",
        "reordered",
        "duplicate",
        "partial_changed_doc_one_token",
        "changed_query_one_token",
        "neighbor_shortened31",
    ]
    assert len(cases[-1]["documents"][1]) == 31


def test_tiny_real_canonical_cache_smoke(tmp_path):
    torch.set_num_threads(1)
    docs, query = [list(range(2, 34)), list(range(34, 66))], list(range(66, 74))
    report = diagnose(
        _model("sdpa"),
        _model("eager"),
        docs,
        query,
        tmp_path,
        {"weights_identity": "tiny-test", "tokenizer_identity": "tiny-tokenizer"},
    )
    assert report["canonical_api"] == "CanonicalCache"
    assert report["exact_bc_gate"]
    assert len(report["cases"]) == 7
    for case in report["cases"]:
        assert case["b_vs_c_query"]["equal"]
        assert all(
            step["b_vs_c_logits"]["equal"] and step["b_greedy_id"] == step["c_greedy_id"]
            for step in case["teacher_forced"]
        )
        assert case["natural_generation_up_to4"]["exact"]
        assert all(row["equal"] for row in case["repeat_encoding_exact"])
        assert all(row["equal"] for row in case["assembly_b_exact"] + case["assembly_c_exact"])
