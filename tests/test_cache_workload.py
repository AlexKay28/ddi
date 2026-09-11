"""CPU gate for the opt-in V53 diagnostic; never downloads a model."""

import json

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from ddi.cache import DocCache
from ddi.inference import BatchEncoding, generate
from ddi.model import patch_model
from ddi.positions import build_positions
from ddi.run import _time_full_causal
from scripts.validate_cache_qa import bootstrap_difference, bootstrap_ratio
from scripts.validate_cache_workload import TokenInput, compare, request, workload


def test_exact_paired_workload():
    docs, query, stream = workload(0, 1000)
    assert list(map(len, docs)) == [500] * 32
    assert len(query) == 32 and len(stream) == 60
    assert all(len(set(indices)) == 5 for indices in stream)
    assert (docs, query, stream) == workload(0, 1000)


def test_fixed_work_cache_and_natural_eos(tmp_path):
    torch.manual_seed(53)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    patch_model(model)
    docs, query, _ = workload(0, 64, n=2, length=8, query=4, top=4, requests=4)
    cache = DocCache(tmp_path)
    for indices, expected in [([0, 1], [False, False]), ([0, 1], [True, True]),
                              ([1, 0], [True, True]), ([1, 2], [True, False])]:
        fresh, a = request(model, docs, query, indices, "ddi_recompute", None,
                           "test", budget=4, diagnostic=True)
        cached, b = request(model, docs, query, indices, "ddi_cache", cache,
                            "test", budget=4, diagnostic=True)
        assert cached["hit_state"] == expected
        assert fresh["generated_ids"] == cached["generated_ids"]
        assert compare(a, b)["numerical_pass"]
        assert fresh["decode_forwards"] == cached["decode_forwards"] == 3
        for mode in ["ddi_recompute", "ddi_cache", "full_causal"]:
            fixed, _ = request(model, docs, query, indices, mode, cache, "test", budget=4)
            natural, _ = request(model, docs, query, indices, mode, cache, "test", budget=4,
                                 eos=fixed["generated_ids"][0])
            assert fixed["generated_count"] == 4 and fixed["decode_forwards"] == 3
            assert natural["generated_count"] == 1 and natural["decode_forwards"] == 0
    isolated, _ = request(model, docs, query, [0, 1], "ddi_cache", cache, "other", budget=4)
    assert isolated["hit_state"] == [False, False]

    # Reproduce the original asymmetry without changing shipped EOS behavior.
    tokenizer = TokenInput()
    tokenizer.eos_token_id = isolated["generated_ids"][0]
    tokenizer.decode = lambda ids, **kwargs: str(ids)
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    _time_full_causal(model, tokenizer, [json.dumps(docs[0]), json.dumps(docs[1])],
                      json.dumps(query), 4)
    assert len(calls) == 5  # prefill plus four forwards, even if first token is EOS
    calls.clear()
    batch = BatchEncoding(torch.tensor([docs[0] + docs[1] + query]),
                          build_positions([8, 8], 4, 0, "restart")[None], [8, 8], 4, "restart")
    generate(model, tokenizer, batch, "ddi_qf", max_new_tokens=4)
    assert len(calls) == 1  # DDI stops immediately after prefill's EOS
    hook.remove()


def test_numeric_gate_does_not_confuse_argmax_with_identity():
    a = torch.tensor([[0.0, 2.0]])
    b = torch.tensor([[0.0, 2.1]])
    result = compare(a, b)
    assert result["argmax_agreement"] == [True]
    assert not result["numerical_pass"]


def test_cache_qa_paired_bootstrap_direction():
    difference = bootstrap_difference([0.0, 0.5], [0.5, 1.0])
    ratio = bootstrap_ratio([2.0, 4.0], [1.0, 2.0])
    assert difference["mean"] == 0.5
    assert difference["ci95_paired_bootstrap"] == [0.5, 0.5]
    assert ratio["mean"] == 2.0
    assert ratio["ci95_paired_bootstrap"] == [2.0, 2.0]
