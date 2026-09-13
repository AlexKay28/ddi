import json
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from ddi.model import patch_model
from scripts.diagnose_canonical_kv import (
    diagnose,
    sha,
    trace_comparisons,
    verify_sources,
    write_json,
)


def test_generated_module_identity_ignores_temporary_path(tmp_path, monkeypatch):
    snapshot = tmp_path / "generated_source"
    snapshot.mkdir()
    source = snapshot / "transient.py"
    source.write_text("value = 1\n")
    current = tmp_path / "new_temporary_location.py"
    current.write_text(source.read_text())
    monkeypatch.setitem(sys.modules, "transient", SimpleNamespace(__file__=str(current)))
    manifest = dict(sources={}, external_files={}, generated_modules={"transient": sha(source)})
    assert verify_sources(tmp_path, manifest) == []
    current.write_text("value = 2\n")
    assert verify_sources(tmp_path, manifest) == ["transient"]


def test_operation_major_order():
    zero, one = torch.zeros(1, 1, 2), torch.ones(1, 1, 2)
    combined = [
        (0, "q_proj", torch.cat([zero, zero], 1)),
        (1, "q_proj", torch.cat([zero, zero], 1)),
    ]
    independent = [
        [(0, "q_proj", zero), (1, "q_proj", one)],
        [(0, "q_proj", one), (1, "q_proj", zero)],
    ]
    rows = trace_comparisons(combined, independent, [1, 1])
    first = next(r for r in rows if not r["equal"])
    assert first["layer"] == 0 and first["document"] == 1


def test_no_overwrite(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"old": True})
    with pytest.raises(FileExistsError):
        write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}


def test_tiny_qwen_controls(tmp_path):
    torch.manual_seed(58)
    torch.set_num_threads(1)
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
    assert all(r["finite"] for r in report["operations"])
    operations = {r["operation"] for r in report["operations"]}
    assert {
        "embedding",
        "input_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "k_post_rope",
        "v_cache",
        "attention_before_o_proj",
        "attention_output",
        "post_attention_norm",
        "mlp_output",
    } <= operations
    assert report["query_logits"]["B_vs_C"]["all"]["equal"]
    assert report["query_logits"]["A_vs_A_replay"]["all"]["allclose_1e4"]
    assert all(r["equal"] for checks in report["assembly"].values() for r in checks)
    steps = report["teacher_forced"]["steps"]
    assert len(steps) >= 3
    for i, step in enumerate(steps):
        assert step["shared_prefix"] == [s["a_selected_token"] for s in steps[:i]]
        assert step["comparisons"]["B_vs_C"]["equal"]
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
