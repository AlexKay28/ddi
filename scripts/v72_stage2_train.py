"""V72 stage 2: Llama-3.1-8B k=5 transfer under the frozen V70 protocol.

Reuses the V70 harness by patching MODEL_ID and BASE_CHECKPOINT at import
time — no V70 file is modified. The training protocol (1000 steps, top-8
layers, AdamW8bit, lr 1e-5, 15% independent presentations, seed 47) is
inherited unchanged. Dev eval and confirmation follow the same V70 path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v70_run as v70  # noqa: E402

LLAMA_CHECKPOINT = ROOT / "results/checkpoints/meta-llama__Llama-3.1-8B-Instruct__ddi_qf__8L"
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("k0", "k5"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    v70.MODEL_ID = LLAMA_MODEL_ID
    v70.BASE_CHECKPOINT = LLAMA_CHECKPOINT
    v70.train_arm(args.arm, args.steps, args.checkpoint_dir, args.seed)


if __name__ == "__main__":
    main()
