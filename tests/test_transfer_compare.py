from ddi.transfer_compare import compare


def _result(*, closure: list[float], admissible: bool = True, blind: bool = True):
    return {
        "cells": [
            {
                "dataset": "dataset",
                "metric": "f1",
                "admissible": admissible,
                "blind_stability_pass": blind,
                "verdict": "test",
                "aggregate": {"normalized_closure": {"seed_values": closure}},
            }
        ]
    }


def test_compare_requires_valid_controls() -> None:
    result = compare(
        _result(closure=[0.5, 0.5, 0.5]),
        _result(closure=[0.8, 0.8, 0.8], blind=False),
        left_name="A",
        right_name="B",
        repeats=100,
    )
    assert result["cells"][0]["verdict"] == "UNAVAILABLE"


def test_compare_detects_practical_difference() -> None:
    result = compare(
        _result(closure=[0.5, 0.5, 0.5]),
        _result(closure=[0.7, 0.7, 0.7]),
        left_name="A",
        right_name="B",
        repeats=100,
    )
    cell = result["cells"][0]
    assert cell["mean"] == 0.19999999999999996
    assert cell["practically_distinguished"]
