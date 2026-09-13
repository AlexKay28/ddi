"""CPU tests for the V61 timing-scope decomposition harness: no GPU, no FA2 kernels.

Covers the preregistration surface (reduced grid, scope/ratio definitions, manifest written
before measurement), the scope-timer helpers (module-hook timer, attention-kernel
classification on synthetic profiler events, decode mask/position builders), the paired
ratio statistics on synthetic numbers, cell validation and summary entries. The only
model-level test runs the SDPA-arm prefill + request scopes on a tiny CPU Qwen2 — it never
touches CUDA and skips when the tiny-model runtime is unavailable.
"""

from __future__ import annotations

import hashlib
import statistics

import pytest
import torch
from torch.autograd import DeviceType

from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from scripts.run_v61_timing_scopes import (
    BOOTSTRAP_DRAWS,
    DECODE_NEW_TOKENS,
    DECODE_STEPS,
    DOC_COUNTS,
    DOC_LENGTHS,
    KERNEL_PATTERNS,
    QUERY_LENGTH,
    REPEATS,
    ROOT,
    SCHEMA_VERSION,
    SCOPES,
    SEED,
    WARMUPS,
    ModuleCallTimer,
    _summary_entry,
    attention_modules,
    build_cell_specs,
    build_decode_positions,
    build_manifest,
    classify_attention_kernels,
    decode_step_mask,
    paired_ratio_stats,
    run_prefill_scopes,
    run_request_scope,
    validate_scope_cell,
)


def test_schema_constants_pinned():
    assert SCHEMA_VERSION == "v61_timing_scopes_v1"
    assert DOC_COUNTS == (1, 2, 5, 10)
    assert DOC_LENGTHS == (512,)  # preregistered reduction: L=128 dropped
    assert QUERY_LENGTH == 64
    assert SEED == 47
    assert WARMUPS == 2
    assert REPEATS == 10
    assert BOOTSTRAP_DRAWS == 10_000
    assert DECODE_NEW_TOKENS == DECODE_STEPS == 16
    assert SCOPES == ("kernel", "module", "prefill", "request")
    assert KERNEL_PATTERNS == ("flash", "fmha", "attention")


def test_grid_is_reduced_to_l512_only():
    specs = build_cell_specs()
    assert [(spec["n_docs"], spec["doc_length"]) for spec in specs] == [
        (1, 512),
        (2, 512),
        (5, 512),
        (10, 512),
    ]
    for spec in specs:
        assert spec["doc_lengths"] == [spec["doc_length"]] * spec["n_docs"]
        assert spec["query_length"] == QUERY_LENGTH


class _Event:
    def __init__(self, name: str, device_type: object, micros: float) -> None:
        self.name = name
        self.device_type = device_type
        self.self_device_time_total = micros


def test_classify_attention_kernels_synthetic():
    flash_name = "void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, ...>>"
    events = [
        _Event(flash_name, "DeviceType.CUDA", 100.0),
        _Event("fmha_cutlassF_bf16_aligned_64x128_rf_sm80(...)", DeviceType.CUDA, 50.0),
        _Event(flash_name, DeviceType.CUDA, 25.0),
        _Event("aten::scaled_dot_product_attention", DeviceType.CPU, 999.0),  # CPU op: excluded
        _Event("ampere_bf16_s16816gemm_256x128_ldg8", DeviceType.CUDA, 500.0),  # GEMM: excluded
    ]
    attention, all_cuda, matched = classify_attention_kernels(events)
    assert attention == pytest.approx(175.0 / 1e6)
    assert all_cuda == pytest.approx(675.0 / 1e6)
    assert len(matched) == 2
    assert matched["fmha_cutlassF_bf16_aligned_64x128_rf_sm80(...)"]["count"] == 1
    assert matched[flash_name]["count"] == 2
    assert matched[flash_name]["total_us"] == pytest.approx(125.0)


def test_classify_attention_kernels_empty_and_nameless():
    attention, all_cuda, matched = classify_attention_kernels([])
    assert attention == 0.0 and all_cuda == 0.0 and matched == {}
    attention, _, _ = classify_attention_kernels([_Event("", DeviceType.CUDA, 10.0)])
    assert attention == 0.0


class _Attn(torch.nn.Module):
    def forward(self, x):
        return x + 1.0


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()

    def forward(self, x):
        return self.self_attn(x)


class _Tiny(torch.nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_module_call_timer_sums_resets_and_removes():
    model = _Tiny(3)
    # attention_modules() is HF-shaped (model.model.layers); the toy uses .layers directly.
    timer = ModuleCallTimer([layer.self_attn for layer in model.layers])
    assert timer.module_count == 3
    model(torch.zeros(1))
    assert timer.calls == 3
    assert timer.seconds > 0.0
    timer.reset()
    assert timer.calls == 0 and timer.seconds == 0.0
    timer.remove()
    model(torch.zeros(1))
    assert timer.calls == 0  # hooks gone after remove()


def test_paired_ratio_stats_median_mean_and_ci():
    reference = [1.0, 1.1, 0.95, 1.2, 1.05, 0.9, 1.15, 1.0, 1.3, 0.85]
    candidate = [0.9, 0.92, 0.88, 0.95, 0.9, 0.87, 0.93, 0.9, 1.0, 0.86]
    stats = paired_ratio_stats(reference, candidate)
    ratios = [r / c for r, c in zip(reference, candidate, strict=True)]
    assert stats["seed"] == SEED and stats["draws"] == BOOTSTRAP_DRAWS
    assert stats["median"] == pytest.approx(statistics.median(ratios))
    assert stats["mean"] == pytest.approx(sum(ratios) / len(ratios))
    assert stats["ci95_low"] <= stats["mean"] <= stats["ci95_high"]
    assert stats["repeats"] == ratios
    assert paired_ratio_stats(reference, candidate) == paired_ratio_stats(reference, candidate)


def test_paired_ratio_stats_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        paired_ratio_stats([], [])
    with pytest.raises(ValueError):
        paired_ratio_stats([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_ratio_stats([1.0], [0.0])
    with pytest.raises(ValueError):
        paired_ratio_stats([-1.0], [1.0])


def test_decode_positions_continue_restart_layout():
    positions = build_positions([8, 8], 4, 0, "restart")
    assert positions[-1].item() == 11  # max(doc_lengths) + query_len - 1
    decode_positions = build_decode_positions(int(positions[-1].item()), 3)
    assert decode_positions.shape == (1, 3)
    assert decode_positions.tolist() == [[12, 13, 14]]
    with pytest.raises(ValueError):
        build_decode_positions(11, 0)
    with pytest.raises(ValueError):
        build_decode_positions(-1, 2)


def test_decode_step_mask_rows_and_validation():
    doc_lengths = [8, 8]
    prefix_len = 20  # 16 doc tokens + 4 query tokens
    full_additive = to_additive(build_mask(doc_lengths, 4, 2, "ddi_qf"), dtype=torch.float32)
    assert full_additive.shape == (22, 22)

    mask = decode_step_mask(full_additive, prefix_len, step=0, kv_len=21)
    assert mask.shape == (1, 1, 1, 21)
    # Answer row 0 sees both docs, the whole query, and itself — nothing beyond kv_len.
    allow = build_mask(doc_lengths, 4, 2, "ddi_qf")
    assert allow[20, :21].all()
    assert not allow[20, 21:].any()

    mask_last = decode_step_mask(full_additive, prefix_len, step=1, kv_len=22)
    assert mask_last.shape == (1, 1, 1, 22)
    assert allow[21, :22].all()

    with pytest.raises(ValueError):
        decode_step_mask(full_additive, prefix_len, step=0, kv_len=99)
    with pytest.raises(ValueError):
        decode_step_mask(full_additive, prefix_len, step=2, kv_len=23)  # beyond answer rows


def _synthetic_scope_repeats(base: float) -> list[float]:
    return [base * (1.0 + 0.01 * index) for index in range(REPEATS)]


def _synthetic_ok_cell() -> dict:
    reference_scopes = {}
    candidate_scopes = {}
    for index, scope in enumerate(SCOPES):
        reference_repeats = _synthetic_scope_repeats(1.0 + 0.1 * index)
        candidate_repeats = _synthetic_scope_repeats(0.9)
        reference_scopes[scope] = {
            "repeats_seconds": reference_repeats,
            "median_seconds": statistics.median(reference_repeats),
        }
        candidate_scopes[scope] = {
            "repeats_seconds": candidate_repeats,
            "median_seconds": statistics.median(candidate_repeats),
        }
    reference_scopes["kernel"]["all_cuda_kernel_repeats_seconds"] = [
        value * 2.0 for value in reference_scopes["kernel"]["repeats_seconds"]
    ]
    candidate_scopes["kernel"]["all_cuda_kernel_repeats_seconds"] = [
        value * 2.0 for value in candidate_scopes["kernel"]["repeats_seconds"]
    ]
    reference_scopes["module"]["module_calls"] = [28] * REPEATS
    candidate_scopes["module"]["module_calls"] = [28] * REPEATS
    peak = {"peak_allocated_bytes": 32_000_000_000, "peak_reserved_bytes": 33_000_000_000}
    return {
        "status": "ok",
        "n_docs": 2,
        "doc_length": 512,
        "doc_lengths": [512, 512],
        "query_length": QUERY_LENGTH,
        "total_tokens": 2 * 512 + QUERY_LENGTH,
        "tokens_sha256": "0" * 64,
        "positions_sha256": "0" * 64,
        "reference": {
            "backend": "sdpa",
            "scopes": reference_scopes,
            "peak_memory": peak,
            "decode_tokens_sha256": "a" * 64,
        },
        "candidate": {
            "backend": "segmented_fa2",
            "scopes": candidate_scopes,
            "peak_memory": peak,
            "decode_tokens_sha256": "a" * 64,
        },
        "scopes": {
            scope: paired_ratio_stats(
                reference_scopes[scope]["repeats_seconds"],
                candidate_scopes[scope]["repeats_seconds"],
            )
            for scope in SCOPES
        },
        "decode_tokens_match": True,
    }


def test_validate_scope_cell_accepts_synthetic_ok_cell():
    validate_scope_cell(_synthetic_ok_cell())  # must not raise


def test_validate_scope_cell_rejects_broken_cells():
    ok = _synthetic_ok_cell()
    with pytest.raises(ValueError):
        validate_scope_cell({key: value for key, value in ok.items() if key != "scopes"})
    with pytest.raises(ValueError):
        validate_scope_cell(
            {
                **ok,
                "reference": {
                    **ok["reference"],
                    "scopes": {
                        **ok["reference"]["scopes"],
                        "prefill": {
                            **ok["reference"]["scopes"]["prefill"],
                            "repeats_seconds": ok["reference"]["scopes"]["prefill"][
                                "repeats_seconds"
                            ][:5],
                        },
                    },
                },
            }
        )
    with pytest.raises(ValueError):
        validate_scope_cell(
            {
                **ok,
                "reference": {
                    **ok["reference"],
                    "scopes": {
                        **ok["reference"]["scopes"],
                        "kernel": {
                            **ok["reference"]["scopes"]["kernel"],
                            "repeats_seconds": [0.0] * REPEATS,
                        },
                    },
                },
            }
        )
    with pytest.raises(ValueError):
        validate_scope_cell(
            {
                **ok,
                "scopes": {
                    **ok["scopes"],
                    "request": {**ok["scopes"]["request"], "median": 99.0},
                },
            }
        )
    with pytest.raises(ValueError):
        validate_scope_cell(
            {
                **ok,
                "scopes": {
                    **ok["scopes"],
                    "module": {
                        **ok["scopes"]["module"],
                        "ci95_low": ok["scopes"]["module"]["ci95_high"] + 1.0,
                    },
                },
            }
        )
    with pytest.raises(ValueError):
        validate_scope_cell({**ok, "decode_tokens_match": "yes"})


def test_summary_entry_ok_and_error():
    ok = _summary_entry(_synthetic_ok_cell())
    assert ok["status"] == "ok"
    assert set(ok["scopes"]) == set(SCOPES)
    for scope in SCOPES:
        entry = ok["scopes"][scope]
        assert entry["ratio_ci95"][0] <= entry["ratio_mean"] <= entry["ratio_ci95"][1]
    assert ok["peak_memory"]["reference"]["peak_allocated_bytes"] > 0
    assert ok["decode_tokens_match"] is True

    error_cell = _synthetic_ok_cell()
    error_cell["status"] = "error"
    error_cell["error"] = "RuntimeError: CUDA out of memory"
    error_entry = _summary_entry(error_cell)
    assert error_entry["status"] == "error"
    assert error_entry["error"] == "RuntimeError: CUDA out of memory"
    assert "scopes" not in error_entry


def test_manifest_preregisters_definitions_before_measurement(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen2"}')
    manifest = build_manifest(tmp_path)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["prior_artifact"] == "results/v61_timing_20260910T071010Z"

    pre = manifest["preregistration"]
    assert set(pre["scope_definitions"]) == set(SCOPES)
    assert all(isinstance(text, str) and text for text in pre["scope_definitions"].values())
    assert "torch.profiler" in pre["scope_definitions"]["kernel"]
    assert "with_stack=False" in pre["scope_definitions"]["kernel"]
    assert "hook" in pre["scope_definitions"]["module"]
    assert "use_cache=True" in pre["scope_definitions"]["request"]
    assert "seed 47" in pre["ratio_definitions"] and "10000" in pre["ratio_definitions"]
    assert "OUT OF SCOPE" in pre["temperature_labels"]
    assert "no KV is reused" in pre["temperature_labels"]
    assert "L=128" in pre["scope_reduction"]
    assert pre["grid"]["doc_lengths"] == [512]
    assert pre["grid"]["decode_new_tokens"] == 16

    # Cells carry only INPUTS (tokens/positions) — no measurements yet.
    assert len(manifest["cells"]) == 4
    for cell in manifest["cells"]:
        assert "tokens" in cell and "positions" in cell
        assert "reference" not in cell and "scopes" not in cell

    assert (
        manifest["touched_sources"]["ddi/segmented_fa2.py"]
        == hashlib.sha256((ROOT / "ddi" / "segmented_fa2.py").read_bytes()).hexdigest()
    )
    assert "scripts/run_v61_timing_scopes.py" in manifest["touched_sources"]


def test_prefill_and_request_scopes_smoke_on_cpu_tiny_model():
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except Exception as error:  # Local optional vision stack can break lazy Qwen imports.
        pytest.skip(f"tiny Qwen runtime unavailable: {error}")

    from ddi.model import patch_model

    torch.manual_seed(SEED)
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
    )
    model = patch_model(Qwen2ForCausalLM(config).eval())

    doc_lengths = [8, 8]
    query_len = 4
    prefix_len = sum(doc_lengths) + query_len
    generator = torch.Generator().manual_seed(SEED)
    tokens = torch.randint(1, 90, (prefix_len,), generator=generator)
    positions = build_positions(doc_lengths, query_len, 0, "restart").unsqueeze(0)
    prefill_additive = (
        to_additive(build_mask(doc_lengths, query_len, 0, "ddi_qf"), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    full_additive = to_additive(
        build_mask(doc_lengths, query_len, DECODE_NEW_TOKENS, "ddi_qf"), dtype=torch.float32
    )
    decode_positions = build_decode_positions(int(positions[0, -1].item()), DECODE_NEW_TOKENS)

    timer = ModuleCallTimer(attention_modules(model))
    prefill, module, module_calls = run_prefill_scopes(
        model, timer, tokens.unsqueeze(0), prefill_additive, positions, warmups=0, repeats=1
    )
    assert module_calls[0] == timer.module_count
    assert prefill[0] > 0.0
    # On CPU everything is synchronous: hook time is a subset of the forward's wall time.
    assert module[0] <= prefill[0] + 1e-6

    request, kept = run_request_scope(
        model,
        tokens.unsqueeze(0),
        prefill_additive,
        positions,
        full_additive,
        decode_positions,
        prefix_len=prefix_len,
        decode_steps=2,
        warmups=0,
        repeats=1,
    )
    # No request>prefill inequality here: on a tiny model both are sub-ms wall times and the
    # comparison is noise, not signal. The invariants that matter are below.
    assert request[0] > 0.0
    assert len(kept) == 2
    assert all(1 <= token < 97 for token in kept)
