"""GPU discovery and setup.

Called once at CLI startup (via `ddi.cli.main`) so every runner benefits without needing to
duplicate the check. Also usable directly as `ddi.gpu.require_cuda()` in code paths that must
have a GPU (e.g. Phase 3 FA2 timings).
"""

from __future__ import annotations

import torch


def is_available() -> bool:
    return torch.cuda.is_available()


def require_cuda() -> None:
    """Raise RuntimeError if CUDA is not available. Use in paths that are pointless on CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available on this runner. "
            "Fix the container image (torch was likely installed without CUDA support), "
            "or use `make setup` on a CPU-only preflight and skip GPU-only sweeps."
        )


def gpu_info() -> dict[str, object]:
    """Return a plain dict of GPU facts — used at startup + by provenance capture."""
    if not torch.cuda.is_available():
        return {"cuda": False, "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1e9, 1),
        "bf16_supported": _bf16_supported(props),
    }


def _bf16_supported(props: torch.cuda.get_device_properties.__annotations__.get("return", object)) -> bool:
    # Ampere (SM 8.0) and later have native bf16. Older GPUs (SM 7.x and below) fall back to
    # fp16 to avoid slow bf16 emulation.
    return props.major >= 8


def configure(*, verbose: bool = True) -> dict[str, object]:
    """Print a one-line device summary, enable TF32 on Ampere+, return the info dict.

    Idempotent — safe to call multiple times. Prints go to stderr so they don't pollute
    stdout-piped tools (e.g. `ddi-report --list`).
    """
    info = gpu_info()
    if verbose:
        _print_summary(info)
    if info["cuda"]:
        # TF32 gives a large fp32 speedup on Ampere+ with negligible accuracy loss for our path
        # (attention is bf16 anyway). Safe on older GPUs; no-op there.
        torch.set_float32_matmul_precision("high")
    return info


def _print_summary(info: dict[str, object]) -> None:
    import sys

    if not info["cuda"]:
        print(f"[gpu] cpu-only (torch={info['torch']})", file=sys.stderr)
        return
    print(
        f"[gpu] {info['device_name']} "
        f"cc={info['compute_capability']} "
        f"vram={info['vram_gb']}GB "
        f"cuda={info['cuda_version']} torch={info['torch']} "
        f"bf16={'yes' if info['bf16_supported'] else 'no'}",
        file=sys.stderr,
    )
