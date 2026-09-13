"""CPU checks for the opt-in V59 TTFT benchmark."""

import sys
from types import ModuleType, SimpleNamespace

import torch

from scripts.analyze_cache_ttft import (
    bootstrap_mean,
    memory_diagnostics,
    natural_decode_diagnostics,
)
from scripts.audit_cache_ttft import audit_schema_fields
from scripts.benchmark_cache_ttft import (
    crossover,
    fresh_ttft,
    paired_overhead,
    partial_hit_schedule,
    peak_memory_block,
    stage_accounting,
    summarize,
)


def test_crossover_computes_feasible_thresholds():
    result = crossover(fresh_ms=100.0, cold_ms=300.0, warm_ms=50.0, build_ms=250.0)
    assert result == {"hit_rate": 0.8, "reuse_count": 5.0}


def test_crossover_reports_when_warm_cache_cannot_win():
    result = crossover(fresh_ms=100.0, cold_ms=200.0, warm_ms=125.0, build_ms=75.0)
    assert result == {"hit_rate": None, "reuse_count": None}


def test_summary_retains_raw_repeats():
    assert summarize([1.0, 2.0, 3.0]) == {
        "raw_ms": [1.0, 2.0, 3.0],
        "median_ms": 2.0,
        "iqr_ms": [1.5, 2.5],
    }


def test_bootstrap_mean_retains_paired_seed_values():
    result = bootstrap_mean([2.0, 2.0, 2.0], seed=59)
    assert result == {"values": [2.0, 2.0, 2.0], "mean": 2.0, "ci95": [2.0, 2.0]}


def test_partial_hit_schedule_has_every_requested_count_and_is_reproducible():
    schedules = [partial_hit_schedule(5, hit_count, seed=59) for hit_count in range(6)]
    assert [sum(hits) for hits in schedules] == list(range(6))
    assert partial_hit_schedule(5, 3, seed=59) == schedules[3]
    assert partial_hit_schedule(5, 3, seed=60) != schedules[3]


def test_stage_accounting_and_paired_overhead_are_explicit():
    accounting = stage_accounting(10.0, {"lookup_ms": 3.0, "forward_ms": 6.5})
    assert accounting == {"stage_total_ms": 9.5, "residual_ms": 0.5}
    assert paired_overhead({"total_ms": 12.5}, {"total_ms": 10.0}) == {
        "instrumented_ms": 12.5,
        "control_ms": 10.0,
        "overhead_ms": 2.5,
    }


def _instrumented_repeat() -> dict:
    stages = {"lookup_ms": 3.0, "forward_ms": 6.5}
    return {
        "total_ms": 10.0,
        "stages": stages,
        "accounting": stage_accounting(10.0, stages),
        "metrics": {"ttft_ms": 10.0, "request_total_ms": None, "tpot_ms": None},
        "measurement": {"instrumented": True, "pair_id": None},
        "hits": [True],
    }


def _fresh_repeat() -> dict:
    return {"total_ms": 10.0, "token": 1}


def _enabled_repeat(*, instrumented: bool = True, pair_id: str | None = None) -> dict:
    stages = {"lookup_ms": 3.0, "forward_ms": 6.5}
    repeat = {
        "total_ms": 10.0,
        "token": 1,
        "metrics": {
            "ttft_ms": 10.0,
            "decode_total_ms": 4.0,
            "request_total_ms": 14.0,
            "tpot_ms": 2.0,
        },
        "decode": {"length": 2, "work_count": 2, "token_ids_sha256": "fresh-digest"},
        "measurement": {"instrumented": instrumented, "pair_id": pair_id},
        "hits": [True],
    }
    if instrumented:
        repeat |= {"stages": stages, "accounting": stage_accounting(10.0, stages)}
    return repeat


def test_audit_rejects_malformed_v2_accounting():
    repeat = _instrumented_repeat()
    repeat["accounting"]["residual_ms"] = 1.0
    row = {
        "N": 1,
        "fresh": {"repeats": [_fresh_repeat()]},
        "cold": {"repeats": [repeat]},
        "warm": {"repeats": [repeat]},
        "partial_hit": [],
    }
    assert "retained arm: residual mismatch" in audit_schema_fields(row, 2, repeats=1)


def test_fresh_decode_uses_the_first_token_as_every_forced_input(monkeypatch):
    workload = ModuleType("scripts.validate_cache_workload")
    workload.token_digest = lambda tokens: f"digest-{tokens}"
    monkeypatch.setitem(sys.modules, "scripts.validate_cache_workload", workload)

    class Model:
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            logits = torch.zeros((1, kwargs["input_ids"].shape[-1], 8))
            logits[..., 5] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    model = Model()
    result = fresh_ttft(model, [[1, 2]], [3], decode_length=3)
    assert result["decode"]["length"] == 3
    assert result["decode"]["work_count"] == 3
    assert result["metrics"]["request_total_ms"] == (
        result["metrics"]["ttft_ms"] + result["metrics"]["decode_total_ms"]
    )
    assert [call["input_ids"].tolist() for call in model.calls[1:]] == [[[5]], [[5]], [[5]]]


def test_fresh_natural_decode_records_generated_tokens_and_eos(monkeypatch):
    workload = ModuleType("scripts.validate_cache_workload")
    workload.token_digest = lambda tokens: f"digest-{tokens}"
    monkeypatch.setitem(sys.modules, "scripts.validate_cache_workload", workload)

    class Model:
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self):
            self.tokens = iter([2, 7])

        def __call__(self, **kwargs):
            logits = torch.zeros((1, kwargs["input_ids"].shape[-1], 8))
            logits[..., next(self.tokens)] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    result = fresh_ttft(
        Model(), [[1, 2]], [3], natural_decode_max_length=4, eos_token_id=7
    )
    natural = result["natural_decode"]
    assert natural["generated_token_ids"] == [2, 7]
    assert natural["token_ids_sha256"] == "digest-[2, 7]"
    assert natural["stop_reason"] == "eos"
    assert natural["decode_forward_count"] == 1
    assert natural["natural_request_total_ms"] == (
        result["total_ms"] + natural["natural_decode_total_ms"]
    )


def test_audit_accepts_enabled_decode_and_rejects_bad_pairing_and_totals():
    fresh = _enabled_repeat()
    fresh.pop("measurement")
    fresh.pop("hits")
    fresh.pop("stages")
    fresh.pop("accounting")
    cold = _enabled_repeat()
    warm = _enabled_repeat()
    row = {
        "N": 1,
        "fresh": {"repeats": [fresh]},
        "cold": {"repeats": [cold]},
        "warm": {"repeats": [warm]},
        "partial_hit": [],
    }
    assert audit_schema_fields(row, 2, repeats=1, decode_length=2) == []
    warm["decode"]["token_ids_sha256"] = "other-digest"
    warm["metrics"]["request_total_ms"] = 13.0
    failures = audit_schema_fields(row, 2, repeats=1, decode_length=2)
    assert "retained arm: forced decode tokens differ from fresh pair" in failures
    assert "retained arm: request total is inconsistent with TTFT and decode" in failures


def test_audit_rejects_partial_control_with_unpaired_forced_decode_tokens():
    fresh = _enabled_repeat()
    fresh.pop("measurement")
    fresh.pop("hits")
    fresh.pop("stages")
    fresh.pop("accounting")

    def pair(pair_id: str, hits: list[bool]) -> dict:
        control = _enabled_repeat(instrumented=False, pair_id=pair_id)
        instrumented = _enabled_repeat(pair_id=pair_id)
        control["hits"] = hits
        instrumented["hits"] = hits
        return {
            "pair_id": pair_id,
            "control": control,
            "instrumented": instrumented,
            "overhead": paired_overhead(instrumented, control),
        }

    row = {
        "N": 1,
        "fresh": {"repeats": [fresh]},
        "cold": {"repeats": [_enabled_repeat()]},
        "warm": {"repeats": [_enabled_repeat()]},
        "partial_hit": [
            {"hit_count": 0, "schedule_seed": 1, "hits": [False], "pairs": [pair("p0", [False])]},
            {"hit_count": 1, "schedule_seed": 2, "hits": [True], "pairs": [pair("p1", [True])]},
        ],
    }
    row["partial_hit"][0]["pairs"][0]["control"]["decode"]["token_ids_sha256"] = "other-digest"
    failures = audit_schema_fields(row, 2, repeats=1, decode_length=2)
    assert "partial hit control run: forced decode tokens differ from fresh pair" in failures


def test_audit_accepts_existing_v2_disabled_decode_shape():
    repeat = _instrumented_repeat()
    row = {
        "N": 1,
        "fresh": {"repeats": [_fresh_repeat()]},
        "cold": {"repeats": [repeat]},
        "warm": {"repeats": [repeat]},
        "partial_hit": [],
    }
    assert audit_schema_fields(row, 2, repeats=1) == []


def test_audit_rejects_unlabeled_or_malformed_natural_decode():
    fresh = _fresh_repeat()
    fresh["natural_decode"] = {"label": "not-diagnostic"}
    repeat = _instrumented_repeat()
    row = {
        "N": 1,
        "fresh": {"repeats": [fresh]},
        "cold": {"repeats": [repeat]},
        "warm": {"repeats": [repeat]},
        "partial_hit": [],
    }
    assert "fresh run: unlabeled natural decode data" in audit_schema_fields(row, 3, repeats=1)
    failures = audit_schema_fields(row, 3, repeats=1, natural_decode_max_length=2)
    assert "fresh run: malformed natural decode metadata" in failures


def test_audit_accepts_labeled_natural_eos_record(monkeypatch):
    workload = ModuleType("scripts.validate_cache_workload")
    workload.token_digest = lambda tokens: f"digest-{tokens}"
    monkeypatch.setitem(sys.modules, "scripts.validate_cache_workload", workload)
    natural = {
        "label": "natural_diagnostic_not_arm_comparable",
        "max_length": 3,
        "eos_token_id": 7,
        "generated_token_ids": [2, 7],
        "token_ids_sha256": "digest-[2, 7]",
        "stop_reason": "eos",
        "decode_forward_count": 1,
        "natural_decode_total_ms": 4.0,
        "natural_request_total_ms": 14.0,
    }
    fresh = _fresh_repeat() | {"natural_decode": natural}
    cold = _instrumented_repeat() | {"natural_decode": natural}
    warm = _instrumented_repeat() | {"natural_decode": natural}
    row = {
        "N": 1,
        "fresh": {"repeats": [fresh]},
        "cold": {"repeats": [cold]},
        "warm": {"repeats": [warm]},
        "partial_hit": [],
    }
    assert audit_schema_fields(row, 3, repeats=1, natural_decode_max_length=3) == []


def test_natural_analysis_is_separately_labeled_diagnostic():
    natural = {
        "natural_decode_total_ms": 4.0,
        "natural_request_total_ms": 14.0,
    }
    row = {
        arm: {"repeats": [{"natural_decode": natural}]}
        for arm in ("fresh", "cold", "warm")
    }
    assert natural_decode_diagnostics([row]) == {
        "label": "natural_diagnostic_not_arm_comparable",
        "arms": {
            arm: {"natural_decode_total_ms": 4.0, "natural_request_total_ms": 14.0}
            for arm in ("fresh", "cold", "warm")
        },
    }


def test_audit_skips_new_fields_for_retained_v1_artifacts():
    assert audit_schema_fields({"N": 1}, 1, repeats=10) == []


def _memory_row(memory: dict | None) -> dict:
    fresh = _fresh_repeat()
    cold = _instrumented_repeat()
    warm = _instrumented_repeat()
    if memory is not None:
        fresh |= {"memory": memory}
        cold |= {"memory": memory}
        warm |= {"memory": memory}
    return {
        "N": 1,
        "fresh": {"repeats": [fresh]},
        "cold": {"repeats": [cold]},
        "warm": {"repeats": [warm]},
        "partial_hit": [],
    }


def test_audit_accepts_valid_peak_memory_block():
    memory = {"peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048}
    assert audit_schema_fields(_memory_row(memory), 3, repeats=1) == []


def test_peak_memory_block_stores_nulls_without_a_device():
    assert peak_memory_block(torch.device("cpu")) == {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }


def test_audit_rejects_malformed_peak_memory_blocks():
    short = _memory_row({"peak_allocated_bytes": 1024})
    assert "fresh run: malformed peak memory block" in audit_schema_fields(short, 3, repeats=1)
    not_a_dict = _memory_row(None)
    not_a_dict["cold"]["repeats"][0]["memory"] = [1024, 2048]
    failures = audit_schema_fields(not_a_dict, 3, repeats=1)
    assert "retained arm: malformed peak memory block" in failures
    non_integer = _memory_row({"peak_allocated_bytes": 10.5, "peak_reserved_bytes": 2048})
    failures = audit_schema_fields(non_integer, 3, repeats=1)
    assert "retained arm: peak memory values must be positive integers or null" in failures


def test_audit_rejects_null_memory_when_cuda_is_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    failures = audit_schema_fields(
        _memory_row({"peak_allocated_bytes": None, "peak_reserved_bytes": None}), 3, repeats=1
    )
    assert "retained arm: null peak memory requires a host without CUDA devices" in failures


def test_audit_accepts_absent_memory_on_every_schema_version():
    row = _memory_row(None)
    assert audit_schema_fields(row, 2, repeats=1) == []
    assert audit_schema_fields(row, 3, repeats=1) == []


def test_analyzer_emits_peak_memory_medians_when_present():
    fresh_blocks = [
        {"peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048},
        {"peak_allocated_bytes": 2048, "peak_reserved_bytes": 4096},
    ]
    row = {
        "fresh": {"repeats": [{"memory": block} for block in fresh_blocks]},
        "cold": {
            "repeats": [{"memory": {"peak_allocated_bytes": 512, "peak_reserved_bytes": 1024}}]
        },
        "warm": {
            "repeats": [{"memory": {"peak_allocated_bytes": 256, "peak_reserved_bytes": 512}}]
        },
    }
    assert memory_diagnostics([row]) == {
        "arms": {
            "fresh": {"peak_allocated_bytes": 1536.0, "peak_reserved_bytes": 3072.0},
            "cold": {"peak_allocated_bytes": 512, "peak_reserved_bytes": 1024},
            "warm": {"peak_allocated_bytes": 256, "peak_reserved_bytes": 512},
        }
    }
    missing = {
        arm: {"repeats": [{"total_ms": 1.0}]} for arm in ("fresh", "cold", "warm")
    }
    assert memory_diagnostics([missing]) is None
