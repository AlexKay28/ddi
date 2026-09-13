"""Localize combined-fresh versus independently-prefilled document KV divergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ddi.cache import CacheEntry, assemble_dynamic_cache, forward_query_with_doc_cache
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions


def _layers(cache):
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache, strict=True))
    return list(cache)


@torch.no_grad()
def diagnose(model, docs: list[list[int]], query: list[int]) -> dict:
    device = model.device
    lengths = [len(doc) for doc in docs]
    prefix = sum(lengths) + len(query)
    combined_ids = torch.tensor([sum(docs, []) + query], device=device)
    positions = build_positions(lengths, len(query), 0, "restart").to(device)[None]
    allow = build_mask(lengths, len(query), 0, "ddi_qf")
    additive = to_additive(allow, model.dtype).to(device)[None, None]
    fresh = model(
        input_ids=combined_ids,
        position_ids=positions,
        attention_mask=additive,
        use_cache=True,
    )
    fresh_layers = _layers(fresh.past_key_values)
    entries = []
    document_errors = []
    offset = 0
    for doc_index, doc in enumerate(docs):
        ids = torch.tensor([doc], device=device)
        doc_positions = build_positions([len(doc)], 0, 0, "restart").to(device)[None]
        doc_allow = build_mask([len(doc)], 0, 0, "full")
        doc_mask = to_additive(doc_allow, model.dtype).to(device)[None, None]
        out = model(
            input_ids=ids,
            position_ids=doc_positions,
            attention_mask=doc_mask,
            use_cache=True,
        )
        independent_layers = _layers(out.past_key_values)
        k_values, v_values = [], []
        for layer_index, ((fresh_k, fresh_v), (doc_k, doc_v)) in enumerate(
            zip(fresh_layers, independent_layers, strict=True)
        ):
            expected_k = fresh_k[..., offset : offset + len(doc), :]
            expected_v = fresh_v[..., offset : offset + len(doc), :]
            k_error = (expected_k.float() - doc_k.float()).abs()
            v_error = (expected_v.float() - doc_v.float()).abs()
            document_errors.append(
                {
                    "document": doc_index,
                    "layer": layer_index,
                    "k_equal": bool(torch.equal(expected_k, doc_k)),
                    "v_equal": bool(torch.equal(expected_v, doc_v)),
                    "k_max_abs": float(k_error.max()),
                    "v_max_abs": float(v_error.max()),
                }
            )
            k_values.append(doc_k.detach().cpu())
            v_values.append(doc_v.detach().cpu())
        entries.append(
            CacheEntry(
                key=f"doc-{doc_index}",
                k_per_layer=k_values,
                v_per_layer=v_values,
                doc_len=len(doc),
                layout="restart",
                tensor_dtype=str(k_values[0].dtype),
            )
        )
        offset += len(doc)
    assembled = assemble_dynamic_cache(entries, device)
    assembled_layers = _layers(assembled)
    assembly_equal = all(
        torch.equal(
            torch.cat([entry.k_per_layer[layer] for entry in entries], dim=-2),
            assembled_layers[layer][0].cpu(),
        )
        and torch.equal(
            torch.cat([entry.v_per_layer[layer] for entry in entries], dim=-2),
            assembled_layers[layer][1].cpu(),
        )
        for layer in range(len(assembled_layers))
    )
    cached = forward_query_with_doc_cache(model, torch.tensor([query]), entries)
    fresh_last = fresh.logits[:, -1].float()
    cached_last = cached.logits[:, -1].float()
    first = next(
        (row for row in document_errors if not row["k_equal"] or not row["v_equal"]), None
    )
    return {
        "dtype": str(model.dtype),
        "shape": {"doc_lengths": lengths, "query_length": len(query), "prefix": prefix},
        "first_document_kv_divergence": first,
        "document_kv": document_errors,
        "assembly_exact": assembly_equal,
        "fresh_cache_length": int(fresh.past_key_values.get_seq_length()),
        "assembled_cache_length": int(assembled.get_seq_length()),
        "query": {
            "finite": bool(torch.isfinite(fresh_last).all() and torch.isfinite(cached_last).all()),
            "max_abs": float((fresh_last - cached_last).abs().max()),
            "allclose_1e4": bool(torch.allclose(fresh_last, cached_last, atol=1e-4, rtol=1e-4)),
            "argmax_equal": bool(torch.equal(fresh_last.argmax(-1), cached_last.argmax(-1))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--doc-length", type=int, default=32)
    parser.add_argument("--query-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    model, tokenizer = load(str(args.checkpoint), dtype=args.dtype)
    model = model.to("cuda").eval()
    docs = [
        torch.randint(0, tokenizer.vocab_size, (args.doc_length,)).tolist()
        for _ in range(2)
    ]
    query = torch.randint(0, tokenizer.vocab_size, (args.query_length,)).tolist()
    result = diagnose(model, docs, query)
    result.update(
        checkpoint=str(args.checkpoint.resolve()),
        seed=args.seed,
        docs=docs,
        query_ids=query,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
