"""Phase 4.2 per-document KV cache correctness runner (issue #23).

Registers `task: phase4_cache_correctness`. For each document in the eval slice this runner
verifies the two properties required for a per-document cache to be admissible under DDI's
ground rules:

  (a) determinism — computing a document's KV block twice from a cold start yields byte-equal
      (within bf16 tolerance) tensors. Without this a cache is unsound in principle.
  (b) round-trip fidelity — writing the block into `DocCache` and reading it back yields the
      same tensors. Together with (a) this establishes cache-hit == cold-compute at the
      KV-block level, which is Phase 4.2's acceptance criterion (issue #1 §4.2, #23).

Injection of cached blocks into a full multi-doc + query + answer forward is deliberately NOT
here — that is Phase 4.3 (#24 order invariance) and #25 (end-to-end throughput). Keeping this
runner narrow means the correctness gate is decidable from these tensor comparisons alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ddi.cache import DocCache, compute_doc_kv
from ddi.cli import register_runner
from ddi.data import load
from ddi.model import load_model
from ddi.positions import Layout
from ddi.provenance import capture
from ddi.store import append


def _all_close(a_list: list[torch.Tensor], b_list: list[torch.Tensor], atol: float) -> bool:
    if len(a_list) != len(b_list):
        return False
    return all(torch.allclose(a, b, atol=atol) for a, b in zip(a_list, b_list))


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    dataset = config["dataset"]
    model_name = config["model"]

    layout_raw = config.get("layout", "restart")
    if layout_raw not in ("sequential", "restart"):
        raise ValueError(f"unknown layout {layout_raw!r}; expected 'sequential' or 'restart'")
    layout: Layout = layout_raw  # type: ignore[assignment]

    cache_dir = Path(config.get("cache_dir", "cache/doc_kv"))
    max_records = int(config.get("max_records", 10))
    atol = float(config.get("atol", 1e-4))

    model, tokenizer = load_model(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    cache = DocCache(cache_dir)
    prov = capture(
        config_hash=config_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision=dataset,
    )
    run_id_base = (
        f"phase4_cache:{model_name}:{dataset}:{layout}:seed={seed}:{prov.git_sha[:8]}"
    )

    for i, record in enumerate(load(dataset, "val", seed)):
        if i >= max_records:
            break
        if not record["info_contexts"]:
            continue

        for doc_idx, doc_text in enumerate(record["info_contexts"]):
            e1 = compute_doc_kv(model, tokenizer, doc_text, layout)
            e2 = compute_doc_kv(model, tokenizer, doc_text, layout)

            deterministic = e1.key == e2.key and _all_close(
                e1.k_per_layer, e2.k_per_layer, atol
            ) and _all_close(e1.v_per_layer, e2.v_per_layer, atol)

            cache.put(e1)
            hit = cache.get(e1.key)
            if hit is None:
                raise RuntimeError(
                    f"cache miss immediately after put for key={e1.key}; "
                    f"cache_dir={cache_dir}"
                )
            roundtrip = _all_close(hit.k_per_layer, e1.k_per_layer, atol) and _all_close(
                hit.v_per_layer, e1.v_per_layer, atol
            )

            rid = f"{run_id_base}:{record['id']}:doc{doc_idx}"
            append(rid, "kv_deterministic", float(deterministic), prov)
            append(rid, "kv_cache_roundtrip", float(roundtrip), prov)

    print(
        f"[phase4_cache_correctness] cache now holds {cache.size()} entries at {cache_dir}"
    )


register_runner("phase4_cache_correctness")(_run)
