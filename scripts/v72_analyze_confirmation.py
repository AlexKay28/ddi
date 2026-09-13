"""V72 confirmation-gate analysis: runs the V70 fail-closed analysis at k=0.

V72's frozen selection selected the k0 arm for seed 1, while the inherited
analysis script pins EXPECTED_K = 5 (V70's selection). This wrapper runs the
identical fail-closed checks with the V72 expected arm and checkpoint pin;
every other expectation (schema, split, data role, record count, confirmation
manifest SHAs, gates, margins, bootstrap) is inherited unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v70_analyze_confirmation import EXPECTED_K  # noqa: E402


def run_v72(argv: list[str] | None = None) -> int:
    import scripts.v70_analyze_confirmation as analysis

    analysis.EXPECTED_K = 0
    assert analysis.EXPECTED_K != EXPECTED_K
    return analysis.main(argv)


if __name__ == "__main__":
    raise SystemExit(run_v72())
