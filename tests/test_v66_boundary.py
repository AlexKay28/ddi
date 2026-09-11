"""Pure-helper tests for the V66 boundary sweep (no GPU, no model load)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from scripts.analyze_v66_boundary import crossover_h, nominal_rows_for_cell
from scripts.run_v66_boundary import (
    LOSE_CI_HIGH,
    WIN_CI_LOW,
    aggregate_identity,
    build_monolithic_batch,
    hit_count_for,
    h_labels_for,
    identity_verdict,
    materiality_verdict,
    pack_documents,
    prune_store,
    ratio_summary,
    reset_store,
    unit_hit_counts,
    verify_store,
)


# ---------------------------------------------------------------------------
# grid expansion
# ---------------------------------------------------------------------------


def test_hit_count_for_rounds_half_up_and_clamps():
    assert hit_count_for(0.0, 10) == 0
    assert hit_count_for(1.0, 10) == 10
    assert hit_count_for(0.25, 10) == 3  # floor(2.5 + 0.5)
    assert hit_count_for(0.5, 10) == 5
    assert hit_count_for(0.9, 10) == 9
    assert hit_count_for(0.25, 2) == 1  # floor(0.5 + 0.5)
    assert hit_count_for(0.0, 5) == 0
    with pytest.raises(ValueError):
        hit_count_for(1.5, 5)
    with pytest.raises(ValueError):
        hit_count_for(0.5, 0)


def test_unit_hit_counts_are_unique_descending_and_match_prereg():
    assert unit_hit_counts(2) == [2, 1, 0]
    assert unit_hit_counts(5) == [5, 4, 3, 1, 0]
    assert unit_hit_counts(10) == [10, 9, 8, 7, 6, 5, 3, 0]
    for n in (2, 5, 10):
        counts = unit_hit_counts(n)
        assert counts == sorted(set(counts), reverse=True)
        assert counts[0] == n and counts[-1] == 0


def test_h_labels_for_covers_the_full_frozen_grid_exactly_once():
    for n in (2, 5, 10):
        labelled: list[float] = []
        for hit_count in unit_hit_counts(n):
            labels = h_labels_for(hit_count, n)
            assert all(hit_count_for(h, n) == hit_count for h in labels)
            labelled.extend(labels)
        assert sorted(labelled) == [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_pack_documents_exact_length_cursor_and_remainder_discard():
    pool = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10], [11]]
    docs, cursor = pack_documents(pool, 4, 2)
    assert cursor == 3  # three paragraphs consumed, remainder of para 3 discarded
    assert docs[0].dtype == torch.long
    assert docs[0].shape == (1, 4)
    assert docs[0].tolist() == [[1, 2, 3, 4]]
    assert docs[1].tolist() == [[6, 7, 8, 9]]
    with pytest.raises(ValueError, match="pool exhausted"):
        pack_documents(pool, 4, 2, start=3)
    with pytest.raises(ValueError):
        pack_documents(pool, 0, 2)


def test_monolithic_batch_matches_inference_encode_semantics():
    tokenizer = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        Path(__file__).resolve().parents[1]
        / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
    )
    from ddi import inference
    from ddi.prompt import build_query_text

    record = {
        "id": "x",
        "info_contexts": ["Alpha beta gamma delta epsilon.", "Second document text."],
        "dialog": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A"}],
        "possible_answers": ["A"],
    }
    query_text = build_query_text(tokenizer, record)
    encoded = inference.encode(
        tokenizer, record["info_contexts"], query_text, layout="restart"
    )
    streams = []
    for text in record["info_contexts"]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        streams.append(torch.tensor([ids], dtype=torch.long))
    query_ids = torch.tensor(
        [tokenizer(query_text, add_special_tokens=False)["input_ids"]], dtype=torch.long
    )
    batch = build_monolithic_batch(streams, query_ids)
    assert batch.doc_lengths == encoded.doc_lengths
    assert batch.query_len == encoded.query_len
    assert batch.layout == "restart"
    assert torch.equal(batch.input_ids, encoded.input_ids)
    assert torch.equal(batch.position_ids, encoded.position_ids)


# ---------------------------------------------------------------------------
# materiality + identity verdicts
# ---------------------------------------------------------------------------


def test_materiality_verdict_frozen_thresholds():
    assert materiality_verdict(1.06, 1.20) == "win"
    assert materiality_verdict(WIN_CI_LOW + 1e-9, 2.0) == "win"
    assert materiality_verdict(0.90, LOSE_CI_HIGH - 1e-9) == "lose"
    assert materiality_verdict(0.90, 0.96) == "tie"
    assert materiality_verdict(1.05, 1.10) == "tie"  # strict inequality at the bound
    assert materiality_verdict(0.90, 1 / 1.05) == "tie"


def test_crossover_h_takes_smallest_winning_nominal_h():
    rows = [
        {"h": 0.0, "wall_verdict": "lose"},
        {"h": 0.5, "wall_verdict": "tie"},
        {"h": 0.8, "wall_verdict": "win"},
        {"h": 1.0, "wall_verdict": "win"},
    ]
    assert crossover_h(rows, "wall")["h"] == 0.8
    assert crossover_h(rows, "wall")["achiever"] == {
        "N": None, "hit_count": None, "h_eff": None,
    }
    assert crossover_h(rows[:2], "wall") == {"h": None, "status": "none_in_grid",
                                             "achiever": None}


def test_nominal_rows_for_cell_expands_shared_hit_counts():
    rows = nominal_rows_for_cell([0.5, 0.6], 3, 5, {"ratio_point": 1.5})
    assert [row["h"] for row in rows] == [0.5, 0.6]
    assert all(row["hit_count"] == 3 and row["h_eff"] == 0.6 for row in rows)
    assert all(row["ratio_point"] == 1.5 for row in rows)


def test_identity_verdict_and_aggregation():
    assert identity_verdict([1, 2, 3], [1, 2, 3]) is True
    assert identity_verdict([1, 2], [1, 3]) is False
    assert identity_verdict((1, 2), [1, 2]) is True
    ok = aggregate_identity([True, True, True])
    assert ok == {"repeats": 3, "all_identical": True, "mismatch_repeat_indices": []}
    bad = aggregate_identity([True, False, True])
    assert bad["all_identical"] is False
    assert bad["mismatch_repeat_indices"] == [1]
    assert aggregate_identity([])["all_identical"] is True


def test_ratio_summary_point_and_verdict_are_consistent():
    fast_reuse = ratio_summary([200.0] * 15, [80.0] * 15)
    assert fast_reuse["ratio_point"] == pytest.approx(2.5)
    assert fast_reuse["verdict"] == "win"
    identical = ratio_summary([100.0, 102.0, 98.0] * 5, [100.0, 102.0, 98.0] * 5)
    assert identical["ratio_point"] == pytest.approx(1.0)
    assert identical["verdict"] == "tie"
    slower = ratio_summary([80.0] * 15, [200.0] * 15)
    assert slower["verdict"] == "lose"
    assert math.isfinite(slower["ratio_ci_low"]) and math.isfinite(slower["ratio_ci_high"])
    with pytest.raises(ValueError):
        ratio_summary([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# store hygiene
# ---------------------------------------------------------------------------


def test_reset_and_prune_store_hold_exact_pre_state(tmp_path: Path):
    store = tmp_path / "store"
    store.mkdir()
    hit_keys = ["aaa", "bbb"]
    for key in ("aaa", "bbb", "ccc"):
        (store / f"{key}.pt").write_bytes(b"x")
    report = reset_store(store, hit_keys)
    assert report["removed"] == 1
    assert report["missing"] == []
    assert report["files"] == 2
    verify_store(store, hit_keys)
    assert prune_store(store, hit_keys) == 0
    (store / "ddd.pt").write_bytes(b"x")
    assert prune_store(store, hit_keys) == 1
    verify_store(store, hit_keys)
    report = reset_store(store, ["aaa", "bbb", "eee"])
    assert report["missing"] == ["eee"]
    with pytest.raises(RuntimeError, match="pre/post-state mismatch"):
        verify_store(store, ["aaa", "bbb", "eee"])
