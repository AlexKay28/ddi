import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FROZEN_SHAS = {
    "experiments/v64_audit_preregistration.json": "3f159715d8a8f1901ca8e4caa951c6de3f54ccd0200c17424ccf7e419a898b1b",
    "results/v64_audit_20260910_Inventory/INVENTORY.json": "ba7850ddafc431d20b2c63eb6220615de6b2c400055e7e027bb2c626a5acb51b",
    "results/v64_audit_20260910T091231Z_QA_RECOMPUTATION/recomputation.json": "7850dbd4664ac50ff2f8d4f7d8de5420441bdb31da86d04aad7f2d911174fb63",
}

STEP3 = "results/v64_audit_20260910T091231Z_QA_RECOMPUTATION"
ARTDIR = "results/v60_confirmation24_20260910T093500Z"
INV = "results/v64_audit_20260910_Inventory/INVENTORY.json"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def verify_inputs():
    for rel, expected in FROZEN_SHAS.items():
        actual = sha256_file(ROOT / rel)
        if actual != expected:
            raise SystemExit(f"input gate FAILURE: {rel} sha {actual} != expected {expected}; stopping, nothing written")


def q(s):
    return json.dumps(s, ensure_ascii=False)


def render_md(sidecar):
    L = []
    a = L.append
    a("# V64 Evidence Audit — Step 4: Metadata Adjudication Sidecar")
    a("")
    a(f"- Created (UTC): {sidecar['created_utc']}")
    a(f"- Scope: adjudication of the V60 confirmation provenance defects per `experiments/v64_audit_preregistration.json` `metadata_adjudication` (§4). Verbatim strings recorded from the frozen step-1 inventory (`{INV}`, sha `ba7850dd…`); nothing re-derived, nothing edited, no GPU, no Tracker.")
    a(f"- Sidecar dir: `{sidecar['sidecar_dir']}` — IMMUTABLE once written: no edits, no overwrites; corrections are issued as a new sidecar directory with a new UTC timestamp. sha256 of both files is printed at generation and is not self-embedded (preregistration §7 convention).")
    a("")
    a("## Step-3 QA recomputation (embedded by sha)")
    a("")
    s3 = sidecar["step3_qa_recomputation"]
    a(f"- Artifact: `{s3['artifact']}` — sha256 `{s3['sha256']}` (created {s3['created_utc']}).")
    a(f"- Records: {s3['record_reconciliation']['matched']}/{s3['record_reconciliation']['frozen']} frozen ids matched, {s3['record_reconciliation']['exclusions']} exclusions.")
    hv = s3["headline_verdict"]
    a(f"- Headline verdict (primary contrast `{hv['contrast']}`): **{hv['verdict'].upper()}** — rule not satisfied ({hv['rule']}).")
    a(f"- Mean paired deltas: EM +0.041667, F1 +0.021982; 95% CI EM [-0.083333, 0.166667], F1 [-0.070833, 0.119518]; CI lower bounds EM -0.083333, F1 -0.070833 vs floor -0.02.")
    a("- Direction favorable (mean delta >= 0 both metrics) but interval too wide to clear the floor — underpowered; recorded as-is, test not altered (prereg §2, §6).")
    a("")
    for d in sidecar["defects"]:
        a(f"## {d['id']} — {d['title']}")
        a("")
        a(f"- Claims-table ref: **{d['claims_table_ref']}** (preregistration §5). Class: {d['class']}.")
        a("")
        a("### Verbatim record (from frozen inventory, file:key paths)")
        a("")
        a("| File | Key path | Verbatim value |")
        a("|---|---|---|")
        for v in d["verbatim"]:
            a(f"| `{v['file']}` | `{v['key_path']}` | {q(v['value'])} |")
        a("")
        a("### Adjudication (preregistration §4)")
        a("")
        for k, val in d["adjudication"].items():
            a(f"- **{k}:** {val}")
        a("")
        a(f"### Consequence\n")
        a(d["consequence"])
        a("")
        a("### Affected downstream claims")
        a("")
        for c in d["affected_downstream_claims"]:
            a(f"- **{c['ref']}:** {c['status']} — {c['note']}")
        a("")
    a("## Provenance of this sidecar")
    a("")
    a(f"- Generator: `scripts/build_v64_sidecar.py`; input gate re-verified the sha256 of the preregistration JSON twin, `{INV}` and `{STEP3}/recomputation.json` before anything was written; aborts on any mismatch.")
    a("- Verbatim strings were copied from fixed key paths of the sha-pinned inventory and preregistration; the step-3 verdict block was copied from the sha-pinned recomputation artifact. No metric was recomputed here; no existing file was modified; no GPU; no Tracker.")
    a("")
    return "\n".join(L) + "\n"


def build_sidecar(created_utc):
    inv = load_json(INV)
    prereg = load_json("experiments/v64_audit_preregistration.json")
    rec = load_json(f"{STEP3}/recomputation.json")

    inv_shas = {f["path"]: f["sha256"] for f in inv["files"]}
    mv = inv["v60_confirmation24"]["manifest_json_verbatim_strings"]
    arm_labels = inv["v60_confirmation24"]["per_arm_labels_verbatim"]
    runner = inv["script_run_canonical_qa_dev"]
    ratios = inv["recomputed_from_retained_per_record"]["derived_ratios_of_medians"]
    tracker_obj = inv["prior_reported_numbers"]["tracker_claim_B_to_C_1.77x"]
    c4 = next(r for r in prereg["claims_table_skeleton"]["rows"] if r["id"] == "C4")
    c5 = next(r for r in prereg["claims_table_skeleton"]["rows"] if r["id"] == "C5")
    c6 = next(r for r in prereg["claims_table_skeleton"]["rows"] if r["id"] == "C6")

    defect1_verbatim = [
        {"file": f"{ARTDIR}/manifest.json", "key_path": "data_role", "value": mv["manifest.json:data_role"]},
        {"file": f"{ARTDIR}/manifest.json", "key_path": "data_role_note", "value": mv["manifest.json:data_role_note"]},
        {"file": f"{ARTDIR}/result.json", "key_path": "data_role", "value": mv["result.json:data_role"]},
        {"file": "experiments/v60_confirmation_manifest.json", "key_path": "data_role", "value": inv["experiments_v60_confirmation_manifest"]["data_role"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L235 (manifest dict)", "value": runner["data_role_lines_verbatim"]["L235 (manifest dict)"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L443 (result dict)", "value": runner["data_role_lines_verbatim"]["L443 (result dict)"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L238-241 (data_role_note)", "value": runner["data_role_lines_verbatim"]["L238-241 (data_role_note)"]},
    ]
    defect2_verbatim = [
        {"file": f"{ARTDIR}/requests.jsonl", "key_path": "arms.A.backend", "value": arm_labels["arms.A"]["backend"]},
        {"file": f"{ARTDIR}/requests.jsonl", "key_path": "arms.B.backend", "value": arm_labels["arms.B"]["backend"]},
        {"file": f"{ARTDIR}/requests.jsonl", "key_path": "arms.C.cold.backend", "value": arm_labels["arms.C.cold"]["backend"]},
        {"file": f"{ARTDIR}/requests.jsonl", "key_path": "arms.C.all_hit.backend", "value": arm_labels["arms.C.all_hit"]["backend"]},
        {"file": f"{ARTDIR}/requests.jsonl", "key_path": "arms.D.backend", "value": arm_labels["arms.D"]["backend"]},
        {"file": f"{ARTDIR}/requests.jsonl", "key_path": "per_arm_labels_verbatim.uniformity", "value": arm_labels["uniformity"]},
        {"file": f"{ARTDIR}/result.json", "key_path": "timing_caveat", "value": mv["result.json:timing_caveat"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L222-225", "value": runner["backend_label_flow_verbatim"]["L222-225"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L227-228", "value": runner["backend_label_flow_verbatim"]["L227-228"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L307 (arm A, hardcoded)", "value": runner["backend_label_flow_verbatim"]["L307 (arm A, hardcoded)"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L321 (arm D, hardcoded)", "value": runner["backend_label_flow_verbatim"]["L321 (arm D, hardcoded)"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L336 (arm B)", "value": runner["backend_label_flow_verbatim"]["L336 (arm B)"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L352 (arm C cold)", "value": runner["backend_label_flow_verbatim"]["L352 (arm C cold)"]},
        {"file": "scripts/run_canonical_qa_dev.py", "key_path": "L367 (arm C all_hit)", "value": runner["backend_label_flow_verbatim"]["L367 (arm C all_hit)"]},
    ]
    defect3_verbatim = [
        {"file": INV, "key_path": "prior_reported_numbers.tracker_claim_B_to_C_1.77x", "value": tracker_obj},
        {"file": INV, "key_path": "recomputed_from_retained_per_record.derived_ratios_of_medians", "value": ratios},
        {"file": "experiments/v64_audit_preregistration.json", "key_path": "claims_table_skeleton.rows[C4].prior_text", "value": c4["prior_text"]},
    ]

    hv = rec["headline_verdict"]
    bma = rec["contrasts"]["B_minus_A"]

    sidecar = {
        "schema": "v64_audit_step4_metadata_sidecar_v1",
        "created_utc": created_utc,
        "purpose": "V64 CHECK-phase step 4: immutable adjudication sidecar for the V60 confirmation provenance defects (data_role, backend labels, B->C 1.77x timing claim), decided exactly per preregistration metadata_adjudication rules. Verbatim strings recorded from the frozen step-1 inventory; step-3 QA result embedded by sha; nothing re-derived.",
        "step_chain": {
            "step_1": "results/v64_audit_20260910_Inventory/ (immutable inventory, sha ba7850dd…)",
            "step_2": "experiments/v64_audit_preregistration.{md,json} (frozen methods, json sha 3f159715…)",
            "step_3": f"{STEP3}/ (QA recomputation, recomputation.json sha 7850dbd4…)",
            "step_4": "this sidecar",
        },
        "additivity": {"existing_files_modified": 0, "gpu_used": False, "tracker_writes": False},
        "recorded_from": {
            "verbatim_source": INV,
            "inventory_sha256": FROZEN_SHAS[INV],
            "preregistration_json_sha256": FROZEN_SHAS["experiments/v64_audit_preregistration.json"],
            "method": "fixed key paths into the sha-pinned inventory and preregistration; no value recomputed or re-derived in this sidecar",
        },
        "step3_qa_recomputation": {
            "artifact": f"{STEP3}/recomputation.json",
            "sha256": FROZEN_SHAS[f"{STEP3}/recomputation.json"],
            "created_utc": rec["created_utc"],
            "record_reconciliation": rec["record_reconciliation"],
            "stored_aggregate_reproduction": rec["aggregate_reproduction"]["status"],
            "b_vs_c_equal_all_raw_check": {"records_equal": rec["b_vs_c_equal_all_raw_check"]["records_equal"], "records_divergent": rec["b_vs_c_equal_all_raw_check"]["records_divergent"], "stored_flag": rec["b_vs_c_equal_all_raw_check"]["stored_flag"]},
            "headline_verdict": hv,
            "primary_contrast_B_minus_A": {"em": bma["em"], "f1": bma["f1"]},
            "note": "Embedded by sha with its verdict verbatim; this sidecar does not recompute QA outcomes. Analysis B (TTFT, prereg §3) is a separate preregistered artifact and is not produced here.",
        },
        "defects": [
            {
                "id": "DEFECT-1",
                "title": "data_role conflict: runner-hardcoded 'development' vs freeze-manifest 'confirmation_frozen'",
                "claims_table_ref": "C5",
                "class": "provenance / data_role mislabel (runner metadata defect)",
                "verbatim": defect1_verbatim,
                "adjudication": {
                    "authoritative_label": "confirmation_frozen",
                    "authority_basis": "Preregistration §4.1: the id-freeze preceded execution — freeze manifest (experiments/v60_confirmation_manifest.json, sha 2dffebf5…, created_utc 2026-09-10T09:30:00Z per preregistration §1) predates the run artifact (results/v60_confirmation24_20260910T093500Z, stamped 09:35:00Z). The freeze is the input that determines what was run; the run cannot retroactively reclassify its own inputs.",
                    "runner_mechanism": "The 'development' string is a hardcoded runner default (scripts/run_canonical_qa_dev.py L235/L443, unconditional dict literals) — recorded as a provenance defect of the runner metadata, NOT as evidence of record contamination.",
                    "prior_text_adjudicated": c5["prior_text"],
                    "scope_limit": "Whether the 24 ids are genuinely untouched (intersections vs V58/V60/V61 observed manifests, observed_exclusion_count: 100) is checklist item 2's separate question, decided on its own evidence; this adjudication only fixes which label is authoritative for the frozen-id question.",
                },
                "consequence": "The 24 records remain admissible as untouched confirmation DATA; the corrected label is confirmation_frozen. Any document citing results/v60_confirmation24_20260910T093500Z must carry this defect note (artifact self-describing metadata data_role/data_role_note is wrong and is preserved verbatim only in the inventory).",
                "affected_downstream_claims": [
                    {"ref": "C5 (data_role label)", "status": "adjudicated", "note": "Corrected text: data_role of the 24 frozen ids = confirmation_frozen; the artifact's 'development' label is a runner-hardcode provenance defect, not contamination evidence."},
                    {"ref": "Artifact self-metadata (manifest.json:data_role, manifest.json:data_role_note, result.json:data_role)", "status": "carry_defect_note", "note": "Wrong by hardcode; original strings preserved verbatim in INVENTORY.json; never silently edited."},
                    {"ref": "Docs citing the artifact as confirmation evidence (V59-V63-CONSOLIDATED-REVIEW.md, VIBEQUEUE-64 description, AUDIT_BASELINE.md §2.1)", "status": "must_carry_defect_note", "note": "Every citation of the 24-record artifact must carry this defect note."},
                    {"ref": "C1/C2/C3 (paired EM/F1 deltas, non-inferiority)", "status": "unaffected_in_admissibility", "note": "The 24 records remain admissible; step-3 recomputation (sha 7850dbd4…) executed on them."},
                    {"ref": "VIBEQUEUE-64 checklist item 2 (untouched status)", "status": "not_adjudicated_here", "note": "Decided on its own evidence (id intersections, policy-selection history); no contamination verdict is issued in this sidecar."},
                ],
            },
            {
                "id": "DEFECT-2",
                "title": "backend labels: executed backend was SDPA for all five arms; narrative 'eager' claims are metadata defects",
                "claims_table_ref": "C6",
                "class": "metadata defect (backend narrative vs per-request records)",
                "verbatim": defect2_verbatim,
                "adjudication": {
                    "authoritative_source": "Per-request records (requests.jsonl -> arms[*].backend, identical across all 24 records): executed backend was sdpa for all five arms (A, B, C_cold, C_all_hit, D), including the canonical arms B/C.",
                    "executed_backend": "sdpa (all five arms)",
                    "runner_mechanism": "Runner default canonical_backend = 'eager' (L222) overridden by --canonical-on-sdpa; retained outputs (B/C = 'sdpa') show --canonical-on-sdpa was in effect.",
                    "consequence_rule": "A-vs-B/C is a same-backend comparison (monolithic vs canonical execution); narrative 'eager' claims for the canonical arms are metadata defects wherever they appear (prose, comments, review); per-request records always outrank narrative.",
                    "corollary": "result.json:timing_caveat's 'cross-backend … comparisons are inadmissible' phrasing rests on a mislabeled premise (the comparison is same-backend); the caveat is preserved verbatim in the inventory and its wording is corrected here in step 4.",
                    "prior_text_adjudicated": c6["prior_text"],
                },
                "consequence": "All references to the canonical arms as 'eager' are metadata defects. The executed configuration was --canonical-on-sdpa: every arm, including B/C canonical, ran on SDPA. TTFT comparisons among A/B/C are same-backend (monolithic vs canonical execution on sdpa); whether specific timing comparisons are admissible remains bounded by checklist item 4 (cold/hit cache-state boundaries), which this sidecar does not adjudicate.",
                "affected_downstream_claims": [
                    {"ref": "C6 (backend narrative)", "status": "adjudicated", "note": "Corrected text: per-request records show backend=sdpa for all five arms; the run used --canonical-on-sdpa; any 'eager' narrative for canonical arms is a metadata defect."},
                    {"ref": "result.json:timing_caveat", "status": "premise_corrected", "note": "'cross-backend … inadmissible' rests on a mislabeled premise; comparison is same-backend; caveat preserved verbatim in INVENTORY.json, wording corrected in step 4."},
                    {"ref": "Prose/comments/review claiming canonical arms ran eager (VIBEQUEUE-60 comments, review prose)", "status": "metadata_defects", "note": "Entered as metadata defects; per-request records outrank narrative."},
                    {"ref": "C4 (1.77x timing claim interpretation)", "status": "affected_interpretation", "note": "B/A and C/A TTFT ratios are same-backend comparisons; the 'cross-backend inadmissibility' argument is void; cold/hit boundary scope questions (checklist item 4) remain open."},
                    {"ref": "C1/C2/C3/C7 (QA deltas, non-inferiority, b_vs_c_equal_all)", "status": "consistent", "note": "Computed on same-backend (sdpa) records; canonical consistency holds within one backend contract."},
                    {"ref": "V61 timing artifact (results/v61_timing_20260910T071010Z, sdpa vs segmented_fa2)", "status": "not_affected", "note": "Genuinely cross-backend artifact; outside this defect."},
                ],
            },
            {
                "id": "DEFECT-3",
                "title": "Tracker claim 'B→C 1.77x' has no artifact source; true B/C = 1.8876x; 1.77x matches C_all_hit/A = 1.7705x",
                "claims_table_ref": "C4",
                "class": "falsified external claim (contrast/reciprocal confusion)",
                "verbatim": defect3_verbatim,
                "adjudication": {
                    "no_artifact_source": "The 'B→C 1.77x' claim exists only as a VIBEQUEUE-60 comment; it is not stored in any artifact (preregistration C4 prior_text; inventory prior_reported_numbers.tracker_claim_B_to_C_1.77x.stored_in_artifacts = 'NOT in any artifact …').",
                    "true_B_over_C": "medianTTFT(B) / medianTTFT(C_all_hit) = 1.8876x (inventory derived_ratios_of_medians: B_over_C_all_hit = 1.8876273002662878; B/C_cold = 1.8864x).",
                    "origin_of_1.77x": "The 1.77x figure matches C_all_hit/A = 1.7705x (inventory: C_all_hit_over_A = 1.7704997002634049; C_cold/A = 1.7716x is also ≈1.77) — the claim confuses the B/C contrast with C/A (the reciprocal/contrast confusion preregistered in preregistration §3 A_over_C).",
                    "corrected_statement": "Cached canonical execution (C_all_hit) is ~1.7705x slower than original monolithic DDI (A); B (cache-disabled canonical) is ~1.8876x slower than C_all_hit. The claim 'B→C 1.77x' does not recompute as stated and is falsified by the retained per-record data.",
                    "estimand_caveat": "Ratios are of medians of per-record arms.<X>.ttft_median_seconds (each the median of 3 stored ttft_seconds repeats). The full preregistered bootstrap reconstruction (preregistration §3, Analysis B: B_over_C, C_over_A, A_over_C + CIs) is a separate artifact and is not produced in this sidecar; do not conflate ratio of medians with mean paired request ratios.",
                    "prior_text_adjudicated": c4["prior_text"],
                },
                "consequence": "'B→C 1.77x' must not be cited as a B/C result anywhere. The B/C ratio of medians is 1.8876x; the 1.77x figure belongs to C_all_hit/A = 1.7705x. Tracker correction (VIBEQUEUE-60) is checklist item 5/6 work and is out of scope for this sidecar (no Tracker writes).",
                "affected_downstream_claims": [
                    {"ref": "C4 (B→C 1.77x timing claim)", "status": "falsified_as_stated", "note": "Corrected text: medianTTFT(B)/medianTTFT(C_all_hit) = 1.8876x; the 1.77x figure matches medianTTFT(C_all_hit)/medianTTFT(A) = 1.7705x."},
                    {"ref": "VIBEQUEUE-60 comment (source of the claim)", "status": "no_artifact_source", "note": "Exists only as a Tracker comment; authoritative correction posted there is out of scope here (no Tracker writes in the additive audit)."},
                    {"ref": "VIBEQUEUE-64 description", "status": "already_carries_correction", "note": "Stores: 'B/C ratio of medians is ~1.888x, not comment's 1.77x; C/A is ~1.771x slower.'"},
                    {"ref": "V59-V63-CONSOLIDATED-REVIEW.md:60-61", "status": "already_carries_correction", "note": "Stores the corrected arithmetic ('the displayed median ratio B/C is about **1.888x**, not the comment's 1.77x. C/A is about **1.771x** …'); consistent with the inventory recomputation."},
                    {"ref": "Any paper/claim text quoting 1.77x as B→C", "status": "must_not_cite", "note": "1.77x may only be cited as C_all_hit/A ≈ 1.7705x; B/C is ≈1.8876x (ratios of medians, named estimand)."},
                ],
            },
        ],
        "claims_table_mapping": {"DEFECT-1": "C5", "DEFECT-2": "C6", "DEFECT-3": "C4", "embedded_by_sha": ["C1", "C2", "C3 via step3_qa_recomputation (sha 7850dbd4…)"]},
        "immutability": {
            "policy": "This sidecar is immutable once written: no in-place edits, no overwrites; corrections are issued as a new sidecar directory with a new UTC timestamp; prior strings remain preserved verbatim in the sha-pinned inventory.",
            "self_sha_note": "sha256 of SIDECAR.json and SIDECAR.md is printed by the generator at creation and is not self-embedded (preregistration §7 convention).",
            "generator": "scripts/build_v64_sidecar.py",
        },
    }
    return sidecar


def main():
    verify_inputs()
    ts = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_utc = datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = ROOT / "results" / f"v64_audit_{ts}_METADATA_SIDECAR"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing sidecar dir: {out}")
    out.mkdir(parents=True)

    sidecar = build_sidecar(created_utc)
    sidecar["sidecar_dir"] = f"results/v64_audit_{ts}_METADATA_SIDECAR"
    json_path = out / "SIDECAR.json"
    md_path = out / "SIDECAR.md"
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_md(sidecar), encoding="utf-8")

    s_json = sha256_file(json_path)
    s_md = sha256_file(md_path)
    print(f"sidecar dir: {sidecar['sidecar_dir']}")
    print(f"sha256  {s_json}  SIDECAR.json")
    print(f"sha256  {s_md}  SIDECAR.md")
    print(f"step-3 QA recomputation embedded by sha: {FROZEN_SHAS[f'{STEP3}/recomputation.json']} (verdict: {sidecar['step3_qa_recomputation']['headline_verdict']['verdict']})")
    print("additivity: existing_files_modified=0, gpu_used=False, tracker_writes=False")


if __name__ == "__main__":
    main()
