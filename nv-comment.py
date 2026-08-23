#!/usr/bin/env python3
"""Set — or read back — a Nirvana instance's comment, this project's canonical identity line.

    <entry-point> | <model> | <arc-ref[:10]> | <date> | <ticket>[ | <note>]

WHY THIS EXISTS. Every DDI row publishes into ONE workflow, so every instance carries the
same name — `ALIGNMENT-2445 | [DDI] Serverless Inference`. In the Nirvana UI, in
`nvt --json` and in the ticket, three concurrent runs are then indistinguishable except by
uuid. The per-instance `comment` is the field Nirvana provides for exactly this, and the
rest of this account already uses it that way.

WHAT IT REPLACES. `publish` already writes a comment of its own —
`<fully.qualified.entrypoint>@[r-1](<arcadia link>)`. The entry point half is genuinely
useful and is kept; the link half points at `rev=-1`, which does not resolve, and the model,
revision and date are missing entirely.

The model is read back from the *published instance* rather than passed in, so the comment
cannot disagree with what the job will actually run: it is parsed out of the block's own
`run-command`, the same string the shell executes.

API notes, verified 2026-08-07 — none of this is guessable and two of the three plausible
routes fail silently or loudly:

* `public/v1/addCommentToWorkflowInstance` {workflowInstanceId, comment} — the real setter.
  Despite the name it REPLACES rather than appends (verified: setting "PROBE-ONE" then
  "PROBE-TWO" leaves exactly "PROBE-TWO").
* `public/v1/editWorkflow` accepts `workflowParams={"comment": ...}` and returns "ok"
  WITHOUT changing anything. A silent no-op — do not use it for this.
* `public/v1/setWorkflowMetaData`, `front/setProcessInstanceComment`,
  `front/setWorkflowInstanceComment`, `front/updateProcessInstance` — all HTTP 404.

READING IT BACK (`--read`), verified 2026-08-08. This half exists because the comment is
the ONLY place a published number's revision survives. `ddi.aggregate` stamps every
summary JSON with `git_sha`, and on a Nirvana job that value is the literal string
`not-a-git-repo`: the job runs from an `arc export`, which is a plain directory tree with
no VCS metadata for `ddi/provenance.py` to read. So the summary cannot say what produced
it, and the instance comment — written here at publish time out of
`nirvana/common/constant.ARC_REFERENCE` — is the record. `make repro-pin` is the reader.

* `public/v1/getWorkflowMetaData` {workflowInstanceId} — returns `instanceComment`, plus
  `name` / `owner` / `started`. This is the getter.
* `front/getProcessInstance` wants `processInstanceId` and is the front-API equivalent;
  `public/v1/getWorkflowInstance` is HTTP 404. `front/findProcessInstances` (what `nvt`
  uses) also carries `comment`, but it filters by owner/name/status and has no by-id
  filter, so it is the wrong shape for "this one instance".
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

PUBLIC = "https://nirvana.yandex-team.ru/api/public/v1/"


def read_token() -> str:
    path = Path.home() / ".nirvana" / "token"
    if not path.is_file():
        sys.exit(f"no Nirvana token at {path}")
    return path.read_text().strip()


def call(token: str, method: str, params: dict) -> object:
    body = json.dumps({"jsonrpc": "2.0", "id": "1", "method": method, "params": params})
    req = urllib.request.Request(
        PUBLIC + method,
        data=body.encode(),
        headers={"Authorization": "OAuth " + token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        sys.exit(f"{method}: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        sys.exit(f"{method}: {exc.reason}")
    if "error" in payload:
        sys.exit(f"{method}: {payload['error']}")
    return payload.get("result")


# `MODEL="Qwen/..."` as written by scripts.finetune_script / eval_script.
_MODEL_RE = re.compile(r'^MODEL="([^"]+)"', re.MULTILINE)


def model_from_instance(token: str, instance: str) -> str | None:
    """The model this instance will actually run, parsed out of a block's run-command.

    Prefers a finetune block: a finetune graph has both, and the training cell is what the
    row is named for. Falls back to any block that names a model, so eval-only rows still
    get a comment.
    """
    blocks = call(token, "getBlockParameters", {"workflowInstanceId": instance}) or []
    found: dict[str, str] = {}
    for block in blocks:
        code = block.get("blockCode") or ""
        for param in block.get("parameters") or []:
            if param.get("parameter") != "run-command":
                continue
            match = _MODEL_RE.search(str(param.get("value") or ""))
            if match:
                found[code] = match.group(1)
    for code, model in found.items():
        if code.startswith("finetune"):
            return model
    return next(iter(found.values()), None)


# ---------------------------------------------------------------------------
# Read side. One parser for the format written above, so the two halves cannot drift.
# ---------------------------------------------------------------------------

_FIELDS = ("graph", "model", "arc-ref", "date", "ticket", "note")
_SEP = " | "
# What `--arc-ref` writes: `args.arc_ref[:10]` of an Arcadia commit hash.
_ARC_REF_RE = re.compile(r"^[0-9a-f]{7,40}$")


def parse_comment(comment: str) -> dict[str, str] | None:
    """Split a stamp into its fields, or None if this is not one of ours.

    The negative case is the one that matters: `publish` writes a comment of its own
    (`<fully.qualified.entrypoint>@[r-1](<arcadia link>)`) to anything published without
    this script, and an instance created that way carries NO revision at all. Returning
    None rather than a half-parsed dict is what lets `--read` say so instead of printing a
    confident `arc checkout` line built from a URL fragment.
    """
    parts = [p.strip() for p in comment.split(_SEP)]
    if len(parts) < 5:
        return None
    if not _ARC_REF_RE.match(parts[2]):
        return None
    parts = parts[:6]
    return dict(zip(_FIELDS, parts))


def comment_from_instance(token: str, instance: str) -> str:
    """The instance's current comment. See the module docstring for why this method."""
    meta = call(token, "getWorkflowMetaData", {"workflowInstanceId": instance})
    if not isinstance(meta, dict):
        sys.exit(f"getWorkflowMetaData: unexpected result for instance {instance}")
    return str(meta.get("instanceComment") or "").strip()


def arc_path() -> str:
    """The Arcadia subtree the graphs export, read from the graphs' own constant.

    Derived rather than restated so `arc checkout` advice cannot point at a path this
    project has moved away from. Falls back to a literal only if the constant is
    unreadable (e.g. this script copied out of the tree).
    """
    src = Path(__file__).resolve().parent / "nirvana" / "common" / "constant.py"
    try:
        match = re.search(r'^ARC_PATH = "(.*)"', src.read_text(), re.MULTILINE)
    except OSError:
        return "junk/alexkay28/ddi"
    return match.group(1) if match else "junk/alexkay28/ddi"


def read_mode(token: str, instance: str) -> int:
    comment = comment_from_instance(token, instance)
    if not comment:
        print(f"instance {instance} has NO comment — its revision is not recoverable.")
        print("Nothing else records it: a Nirvana job runs from an `arc export` with no")
        print("VCS metadata, so the summary JSON's git_sha reads 'not-a-git-repo'.")
        return 1

    print(f"instance : {instance}")
    print(f"comment  : {comment}")
    fields = parse_comment(comment)
    if fields is None:
        print("")
        print("NOT stamped by nv-comment.py — this comment carries no arc-ref field.")
        print("`publish` writes `<entrypoint>@[r-1](<link>)` on its own, and that link")
        print("resolves to rev=-1. The revision this instance ran is not recoverable")
        print("from Nirvana; check the ticket, or re-stamp future instances with")
        print("`make nv-publish` / `make nv-comment`.")
        return 1

    print("")
    for name in _FIELDS:
        if fields.get(name):
            print(f"{name:8} : {fields[name]}")
    print("")
    print("reproduce the code this instance ran:")
    print(f"  arc checkout {fields['arc-ref']}")
    print(f"  # exported subtree: {arc_path()}")
    print("  # abbreviated to 10 chars by the stamp; `arc checkout` accepts the prefix.")
    print("  # This, not the summary JSON, is the revision record — ddi.aggregate stamps")
    print("  #   git_sha='not-a-git-repo' for every Nirvana run.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", required=True, help="workflow instance id")
    ap.add_argument(
        "--read",
        action="store_true",
        help="print the instance's existing comment, parsed into fields, plus the "
             "`arc checkout` line that reproduces it. Sets nothing.",
    )
    # Required for a write, meaningless for a read — enforced in main() rather than by
    # argparse so `--read` needs nothing but `--instance`.
    ap.add_argument("--graph", help="entry-point function name (write mode)")
    ap.add_argument("--arc-ref", help="arc_reference the instance ships (write mode)")
    ap.add_argument("--ticket", default="VIBEQUEUE-2")
    ap.add_argument("--note", default="", help="extra field, e.g. a recipe caveat")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true", help="print, do not set")
    args = ap.parse_args()

    token = read_token()

    if args.read:
        sys.exit(read_mode(token, args.instance))

    if not args.graph or not args.arc_ref:
        ap.error("--graph and --arc-ref are required unless --read is given")

    model = model_from_instance(token, args.instance) or "model?"
    fields = [args.graph, model, args.arc_ref[:10], args.date, args.ticket]
    if args.note:
        fields.append(args.note)
    comment = _SEP.join(fields)

    if args.dry_run:
        print(comment)
        return

    call(token, "addCommentToWorkflowInstance",
         {"workflowInstanceId": args.instance, "comment": comment})
    print("comment set: " + comment)


if __name__ == "__main__":
    main()
