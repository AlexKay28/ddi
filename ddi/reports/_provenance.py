"""Provenance footer for rendered tables (issue #77).

Every table-emitting renderer in `ddi/reports/` appends the block returned by
`render_provenance_footer(store, rows)` to its Markdown output so reviewers can
audit each cell back to the store state at render time. The footer is intentionally
Markdown-safe (leading `<!-- provenance -->` marker + a blockquote) so it pastes
into GitHub/paper toolchains without a wrapper.

Fields (all present, in this order):
- UTC render timestamp (ISO8601, seconds precision — matches `ddi.provenance`)
- git SHA at render time (`git rev-parse HEAD`, or the literal `not-a-git-repo`)
- store path + first 8 hex chars of sha256(store bytes)
- row count backing the render, distinct run_ids, model list, seed list

Design notes:
- Stdlib only (hashlib, subprocess, datetime). No dependency on `ddi.provenance`
  itself — the footer stamps the store, not the runner process, so pulling in the
  torch-bound Provenance dataclass would just add import weight for nothing.
- Row inputs are the exact dicts the renderer aggregated; models/seeds are read
  from each row's nested `provenance` dict (fields `model_revision`, `seed`),
  falling back to `"unknown"` / `-1` so a malformed row never crashes rendering.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

# Public marker: renderer tests grep for this string rather than matching volatile
# timestamp/SHA fields. Keep it single-line and stable — LaTeX \input relies on it.
FOOTER_MARKER = "<!-- provenance -->"


def _git_sha() -> str:
    """Return current HEAD SHA, or the literal `"not-a-git-repo"` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not-a-git-repo"
    sha = result.stdout.strip()
    return sha if sha else "not-a-git-repo"


def _store_hash(store: Path) -> str:
    """First 8 hex chars of sha256(store bytes); `"missing"` if the file is absent."""
    try:
        data = store.read_bytes()
    except (FileNotFoundError, OSError):
        return "missing"
    return hashlib.sha256(data).hexdigest()[:8]


def _distinct(values: Iterable[object]) -> list[str]:
    """Sorted list of stringified distinct values, dropping None/empty."""
    seen: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v)
        if not s:
            continue
        seen.add(s)
    return sorted(seen)


def render_provenance_footer(store: Path, rows: list[dict]) -> str:
    """Return a Markdown block stamping `rows` and the store state at render time.

    The block is prefixed by `FOOTER_MARKER` on its own line so downstream tests and
    tooling can locate it robustly regardless of the volatile timestamp/SHA fields.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # ISO8601 renders `+00:00`; normalize to `Z` for the paper convention.
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[: -len("+00:00")] + "Z"
    git_sha = _git_sha()
    store_hash = _store_hash(Path(store))

    run_ids = _distinct(r.get("run_id") for r in rows)
    models = _distinct((r.get("provenance") or {}).get("model_revision") for r in rows)
    seeds = _distinct((r.get("provenance") or {}).get("seed") for r in rows)

    models_str = ", ".join(models) if models else "unknown"
    seeds_str = ", ".join(seeds) if seeds else "unknown"

    lines = [
        FOOTER_MARKER,
        f"> **Generated:** {timestamp} · git `{git_sha}` · store `{store}` (sha256 `{store_hash}`)",
        f"> **Backing rows:** {len(rows)} rows across {len(run_ids)} run_ids (models: {models_str})",
        f"> **Seeds:** {seeds_str}",
    ]
    return "\n".join(lines) + "\n"
