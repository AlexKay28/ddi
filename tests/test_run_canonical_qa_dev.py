import json
from types import SimpleNamespace

import pytest
import torch

from scripts.run_canonical_qa_dev import ROOT, _quality, main


class _StubTokenizer:
    eos_token_id = None
    chat_template = None

    def apply_chat_template(self, dialog, tokenize=False, add_generation_prompt=True, **kwargs):
        assert tokenize is False
        return dialog[-1]["content"]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = [[1 + (ord(ch) % 90) for ch in text] or [1]]
        return SimpleNamespace(input_ids=torch.tensor(ids))

    def decode(self, ids, skip_special_tokens=True):
        return "Gian Carlo Menotti" if ids and ids[0] == 5 else "other answer"


def _frozen_record_ids(count: int) -> list[str]:
    manifest = json.loads(
        (ROOT / "results/v58_cache_qa_100_20260908T122202Z/manifest.json").read_text()
    )
    return manifest["config"]["record_ids"][:count]


def _records(dataset: str, seed: int) -> dict:
    assert dataset == "hotpot_qa" and seed == 0
    docs = [f"document {index} text" for index in range(2)]
    return {
        record_id: {
            "id": record_id,
            "info_contexts": docs,
            "possible_answers": ["Gian Carlo Menotti"],
            "dialog": [
                {"role": "user", "content": "who wrote the consulate?"},
                {"role": "assistant", "content": "Gian Carlo Menotti"},
            ],
        }
        for record_id in _frozen_record_ids(2)
    }


def _models(checkpoint: str, device: str):
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except Exception as error:
        pytest.skip(f"tiny Qwen runtime unavailable: {error}")
    from ddi.model import patch_model

    torch.manual_seed(60)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
    )

    def build(backend: str):
        config._attn_implementation = backend
        return patch_model(Qwen2ForCausalLM(config).eval()).to(device)

    return build("sdpa"), build("eager")


def test_quality_metrics_match_v57_definitions():
    assert _quality("Gian Carlo Menotti", ["Gian Carlo Menotti"]) == {
        "em": 1.0,
        "f1": 1.0,
        "presence": 1.0,
    }
    scores = _quality("Gian Carlo Menotti wrote it", ["Gian Carlo Menotti"])
    assert scores["em"] == 0.0 and scores["presence"] == 1.0 and 0.0 < scores["f1"] < 1.0


def test_tiny_qwen_dev_qa_matrix_end_to_end(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    output = tmp_path / "artifact"
    summary = main(
        [
            "--checkpoint",
            "unused-tiny",
            "--record-count",
            "2",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--natural-max-new-tokens",
            "4",
            "--fixed-new-tokens",
            "3",
        ],
        load_models=lambda _checkpoint, _device: _models("unused-tiny", "cpu"),
        record_loader=_records,
        tokenizer=_StubTokenizer(),
    )
    assert summary["aggregate"]["b_vs_c_equal_all"] is True
    assert summary["data_role"] == "development"
    for key in ("A", "B", "C_cold", "C_all_hit", "D"):
        assert set(summary["aggregate"]["quality_mean"][key]) == {"em", "f1", "presence"}
        assert summary["aggregate"]["ttft_median_seconds"][key] > 0.0

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["data_role"] == "development"
    assert "never serve as untouched confirmation" in manifest["data_role_note"]
    assert manifest["source_manifest"]["record_ids"] == _frozen_record_ids(2)
    cells = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert len(cells) == 2
    for cell in cells:
        assert all(cell["b_vs_c_equal"].values())
        assert set(cell["arms"]) == {"A", "B", "C", "D"}
        assert cell["arms"]["B"]["quality"] == cell["arms"]["C"]["cold"]["quality"]

    with pytest.raises(FileExistsError):
        main(
            [
                "--checkpoint",
                "unused-tiny",
                "--record-count",
                "2",
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
            load_models=lambda _checkpoint, _device: _models("unused-tiny", "cpu"),
            record_loader=_records,
            tokenizer=_StubTokenizer(),
        )
