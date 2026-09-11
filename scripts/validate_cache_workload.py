"""V53 local diagnostic: exact inputs, equal work, separate cache behavior checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers

from ddi.cache import DocCache, compute_doc_kv, forward_query_with_doc_cache
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from ddi.run import _sample_doc_indices

MODES = ["ddi_cache", "ddi_recompute", "full_causal"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True).encode()


def token_digest(ids):
    return digest(np.asarray(ids, dtype=np.int64).tobytes())


class TokenInput:
    """Adapt exact IDs to the shipped single-document KV prefill, without text conversion."""

    def __call__(self, ids, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([json.loads(ids)], dtype=torch.long))


def workload(seed, vocab, n=5, length=500, query=32, top=32, requests=60):
    rng = np.random.default_rng(seed)
    docs = rng.integers(0, vocab, (top, length)).tolist()
    query_ids = rng.integers(0, vocab, query).tolist()
    stream = _sample_doc_indices(np.random.default_rng(seed), requests, n, top, "zipf", 1.5)
    return docs, query_ids, stream


@torch.no_grad()
def request(model, docs, query, indices, mode, cache, identity, budget=16,
            eos=None, diagnostic=False, forced=None):
    """A emitted tokens = one prefill + A-1 decode forwards; EOS ignored unless supplied.

    Timed scope includes mask construction, disk IO, transfers and cold document computation.
    Logit copies and diagnostics occur only in the separate untimed validation pass.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if mode not in MODES:
        raise ValueError(mode)
    selected = [docs[i] for i in indices]
    lengths = list(map(len, selected))
    prefix = sum(lengths) + len(query)
    device = model.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    hits = []
    entries = []
    if mode == "ddi_cache":
        for ids in selected:
            key = digest(json_bytes([identity, ids]))
            ids_digest = token_digest(ids)
            entry = cache.get(
                key,
                expected_identity=identity,
                expected_token_ids_sha256=ids_digest,
            )
            hits.append(entry is not None)
            if entry is None:
                entry = replace(
                    compute_doc_kv(
                        model,
                        TokenInput(),
                        json.dumps(ids),
                        "restart",
                        cache_identity=identity,
                    ),
                    key=key,
                )
                cache.put(entry)
            entries.append(entry)
    variant = "full" if mode == "full_causal" else "ddi_qf"
    layout = "sequential" if mode == "full_causal" else "restart"
    positions = build_positions(lengths, len(query), budget, layout).to(device)
    additive = to_additive(build_mask(lengths, len(query), budget, variant), model.dtype)
    additive = additive.to(device)
    if mode == "ddi_cache":
        out = forward_query_with_doc_cache(model, torch.tensor([query]), entries)
    else:
        ids = torch.tensor([sum(selected, []) + query], device=device)
        out = model(input_ids=ids, position_ids=positions[:prefix].unsqueeze(0),
                    attention_mask=additive[:prefix, :prefix].clone()[None, None],
                    use_cache=True)
    generated, logits = [], []
    forwards = 0
    for i in range(budget):
        last = out.logits[0, -1]
        token = int(last.argmax().item())
        generated.append(token)
        if diagnostic:
            logits.append(last.detach().float().cpu())
        if token == eos or i == budget - 1:
            break
        fed = token if forced is None else forced[i]
        out = model(input_ids=torch.tensor([[fed]], device=device),
                    position_ids=positions[prefix + i:prefix + i + 1].reshape(1, 1),
                    cache_position=torch.tensor([prefix + i], device=device),
                    attention_mask=additive[prefix + i, :prefix + i + 1].clone()[None, None, None],
                    past_key_values=out.past_key_values, use_cache=True)
        forwards += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    row = dict(indices=indices, doc_lengths=lengths, query_len=len(query),
               generated_ids=generated, generated_count=len(generated), decode_forwards=forwards,
               hit_state=hits, elapsed_ms=elapsed, mode=mode)
    return row, torch.stack(logits) if diagnostic else None


def compare(a, b):
    delta = (a - b).abs()
    return dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                numerical_pass=bool(torch.allclose(a, b, atol=1e-4, rtol=1e-4)),
                argmax_agreement=(a.argmax(-1) == b.argmax(-1)).tolist(),
                fresh_margins=(a.topk(2).values[:, 0] - a.topk(2).values[:, 1]).tolist(),
                cache_margins=(b.topk(2).values[:, 0] - b.topk(2).values[:, 1]).tolist(),
                per_step_max_abs=delta.amax(-1).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--n-docs", type=int, default=5)
    parser.add_argument("--doc-length", type=int, default=500)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--answer-length", type=int, default=16)
    parser.add_argument("--top-docs", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = args.output / "source"
    source.mkdir()
    hashes = {}
    for path in [*sorted((root / "ddi").rglob("*.py")), Path(__file__).resolve()]:
        relative = path.relative_to(root)
        data = path.read_bytes()
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[str(relative)] = digest(data)
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    (args.output / "working.patch").write_bytes(diff)
    checkpoint_hashes = {}
    for path in sorted(args.checkpoint.iterdir()):
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    h.update(chunk)
            checkpoint_hashes[path.name] = h.hexdigest()
    config = dict(
        seeds=args.seeds,
        warmups=args.warmups,
        measured=args.measured,
        N=args.n_docs,
        L=args.doc_length,
        Q=args.query_length,
        A=args.answer_length,
        top=args.top_docs,
        zipf=1.5,
        atol=1e-4,
        rtol=1e-4,
        backend="sdpa",
        dtype="bf16",
        layout="restart",
        decode=(
            f"{args.answer_length} emitted tokens; {args.answer_length - 1} forwards; ignore EOS"
        ),
    )
    manifest = dict(config=config, checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_files=checkpoint_hashes, source_hashes=hashes,
                    source_sha256=digest(json_bytes(hashes)), diff_sha256=digest(diff),
                    base_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
                    .decode().strip(), clean_revision=False, admissible_paper_evidence=False,
                    torch=torch.__version__, transformers=transformers.__version__,
                    python=platform.python_version(), gpu=subprocess.check_output(["nvidia-smi"])
                    .decode(), started_unix=time.time())
    identity = digest(json_bytes(manifest))
    manifest["cache_identity"] = identity
    (args.output / "manifest.json").write_bytes(json_bytes(manifest))
    model, tokenizer = load(str(args.checkpoint))
    model = model.to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    all_rows, validations = [], []
    with (args.output / "timing.jsonl").open("w") as timing, \
            (args.output / "validation.jsonl").open("w") as validation:
        for seed_index, seed in enumerate(config["seeds"]):
            docs, query, stream = workload(
                seed,
                tokenizer.vocab_size,
                n=config["N"],
                length=config["L"],
                query=config["Q"],
                top=config["top"],
                requests=config["warmups"] + config["measured"],
            )
            (args.output / f"inputs_seed{seed}.json").write_bytes(
                json_bytes(dict(docs=docs, query=query, stream=stream)))
            # Validation namespaces cannot prewarm the timed cache.
            scenario_cache = DocCache(args.output / f"validation_cache_{seed}")
            cases = [
                ("cold", stream[0], [False] * config["N"]),
                ("warm", stream[0], [True] * config["N"]),
                ("permuted", list(reversed(stream[0])), [True] * config["N"]),
            ]
            missing = next(i for i in range(config["top"]) if i not in stream[0])
            cases.append(
                ("partial", stream[0][:-1] + [missing], [True] * (config["N"] - 1) + [False])
            )
            duplicate_indices = [stream[0][0], stream[0][0], *stream[0][2:]]
            duplicate_cache = DocCache(args.output / f"duplicate_cache_{seed}")
            duplicate_expected = [False, True, *([False] * (config["N"] - 2))]
            validation_cases = [
                *((case, indices, scenario_cache, expected) for case, indices, expected in cases),
                ("duplicate", duplicate_indices, duplicate_cache, duplicate_expected),
            ]
            stream_cache = DocCache(args.output / f"stream_cache_{seed}")
            seen = set()
            for i, indices in enumerate(stream):
                expected = [index in seen for index in indices]
                seen.update(indices)
                validation_cases.append((f"stream_{i}", indices, stream_cache, expected))
            for case, indices, cache, expected_hits in validation_cases:
                fresh, a = request(model, docs, query, indices, "ddi_recompute", None,
                                   identity, budget=config["A"], diagnostic=True)
                cached, b = request(model, docs, query, indices, "ddi_cache", cache,
                                    identity, budget=config["A"], diagnostic=True)
                # Same-prefix numerical comparison, even after greedy trajectories diverge.
                _, forced = request(model, docs, query, indices, "ddi_cache", cache,
                                    identity, budget=config["A"], diagnostic=True,
                                    forced=fresh["generated_ids"])
                natural_fresh, _ = request(model, docs, query, indices, "ddi_recompute", None,
                                           identity, budget=config["A"], eos=tokenizer.eos_token_id)
                natural_cache, _ = request(model, docs, query, indices, "ddi_cache", cache,
                                           identity, budget=config["A"], eos=tokenizer.eos_token_id)
                row = dict(seed=seed, case=case, fresh=fresh, cache=cached,
                           expected_hit_state=expected_hits,
                           hit_state_pass=cached["hit_state"] == expected_hits,
                           comparison=compare(a, forced),
                           greedy_equal=fresh["generated_ids"] == cached["generated_ids"],
                           natural_fresh=natural_fresh, natural_cache=natural_cache,
                           natural_equal=natural_fresh["generated_ids"]
                           == natural_cache["generated_ids"])
                # Full decision logits retained, not merely aggregate errors.
                torch.save(dict(fresh=a, cache=b, cache_forced=forced),
                           args.output / f"logits_{seed}_{case}.pt")
                validation.write(json.dumps(row) + "\n")
                validation.flush()
                validations.append(row)
            print(f"seed {seed}: validation complete", flush=True)
            order = MODES[seed_index:] + MODES[:seed_index]
            cache = DocCache(args.output / f"timing_cache_{seed}")
            for mode in order:
                for i, indices in enumerate(stream):
                    row, _ = request(
                        model,
                        docs,
                        query,
                        indices,
                        mode,
                        cache,
                        identity,
                        budget=config["A"],
                    )
                    row.update(seed=seed, request=i, warmup=i < config["warmups"], order=order)
                    assert row["doc_lengths"] == [config["L"]] * config["N"]
                    assert row["query_len"] == config["Q"]
                    assert row["generated_count"] == config["A"]
                    assert row["decode_forwards"] == config["A"] - 1
                    timing.write(json.dumps(row) + "\n")
                    timing.flush()
                    all_rows.append(row)
                print(f"seed {seed}: {mode} complete", flush=True)
    qps = {mode: [config["measured"] / (sum(r["elapsed_ms"] for r in all_rows if r["mode"] == mode
                            and r["seed"] == seed and not r["warmup"]) / 1000)
                  for seed in config["seeds"]] for mode in MODES}
    ratios = {}
    for baseline in MODES[1:]:
        values = np.array(qps["ddi_cache"]) / np.array(qps[baseline])
        radius = (
            4.302652729911275 * values.std(ddof=1) / np.sqrt(len(values))
            if len(values) == 3
            else float("nan")
        )
        ratios[baseline] = dict(per_seed=values.tolist(), mean=float(values.mean()),
                                ci95_t=[float(values.mean() - radius),
                                        float(values.mean() + radius)])
    summary = dict(qps=qps, paired_speedup=ratios, validation_count=len(validations),
                   greedy_failures=sum(not r["greedy_equal"] for r in validations),
                   natural_failures=sum(not r["natural_equal"] for r in validations),
                   numerical_failures=sum(not r["comparison"]["numerical_pass"]
                                          for r in validations),
                    finite=all(r["comparison"]["finite"] for r in validations),
                    hit_state_failures=sum(not r["hit_state_pass"] for r in validations),
                    max_abs=max(r["comparison"]["max_abs"] for r in validations),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    wall_seconds=time.time() - manifest["started_unix"],
                    admissible_paper_evidence=False)
    summary["correctness_pass"] = bool(
        summary["finite"]
        and summary["greedy_failures"] == 0
        and summary["natural_failures"] == 0
        and summary["numerical_failures"] == 0
        and summary["hit_state_failures"] == 0
    )
    (args.output / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
