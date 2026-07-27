"""Phase 2.2 full-SFT runner (issue #14).

Registers `task: phase2_finetune_train`. Wraps `ddi.train.train` so a YAML config can
launch a full-SFT run with the DDI mask present throughout. Only the training rung is
implemented here — the LoRA / light-adaptation rung of issue #14 is deferred (marked
TODO in the example config).

Evaluating the resulting checkpoint reuses `phase2_train_free` (issue #13): point that
runner's `model:` at the checkpoint directory produced here. No separate eval runner
exists or is needed — the eval code path is identical to stock-weights inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ddi.cli import register_runner
from ddi.provenance import capture
from ddi.train import TrainConfig, train


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run full SFT with the mask variant present throughout; see module docstring."""
    if "model" not in config:
        raise ValueError("phase2_finetune_train: missing required key 'model'")
    if "dataset" not in config:
        raise ValueError("phase2_finetune_train: missing required key 'dataset'")
    if "mask" not in config or "variant" not in config["mask"]:
        raise ValueError(
            "phase2_finetune_train: missing required key 'mask.variant' — even 'full' must be explicit"
        )

    cfg = TrainConfig(
        model_name=config["model"],
        dataset=config["dataset"],
        mask_variant=config["mask"]["variant"],
        bridge_k=config["mask"].get("bridge_k"),
        split=config.get("split", "train"),
        lr=float(config.get("lr", 1e-5)),
        warmup_steps=int(config.get("warmup_steps", 50)),
        max_steps=int(config.get("max_steps", 500)),
        grad_accum_steps=int(config.get("grad_accum_steps", 8)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        validate_every=int(config.get("validate_every", 100)),
        output_dir=Path(config.get("output_dir", "checkpoints/phase2_finetune")),
    )
    prov = capture(
        config_hash=config_hash,
        seed=seed,
        model_revision=cfg.model_name,
        dataset_revision=cfg.dataset,
    )
    ckpt = train(cfg, seed, prov)
    print(f"[phase2_finetune_train] saved checkpoint to {ckpt}")


register_runner("phase2_finetune_train")(_run)
