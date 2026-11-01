"""Write the build report: inventory tables, licence decisions, held sets with reasons, the QE audit summary and the unverified-provenance log.

    python -m sft_new.report_build --out /workspace/sft_build/out --final /workspace/sft_build/final --bins /workspace/sft_build/bins \
        --license-scan /workspace/sft_build/out/license_scan.json --provenance /workspace/sft_build/provenance.json --dest /workspace/sft_build/report

Writes DEST/SFT_BUILD.md (human report), DEST/license_scan.json (the scan + what was built from each repo + held / skipped with sizes),
DEST/provenance_unverified.json (sources whose machine-translation system could not be verified from their cards; USED, logged only, owner decision
2026-10-10) and DEST/qe_audit_summary.json.
"""
from __future__ import annotations
import os

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# sources whose MT system is not stated on their cards; used, only logged
UNVERIFIED = {
    (os.environ['HF_ORG_LEGACY'] + "/eng_sft"): "Alpaca / TinyStories / CoT machine-translated into Yoruba etc.; no card",
    (os.environ['HF_ORG_LEGACY'] + "/naija_sft"): "machine-translated instruction data; no card",
    (os.environ['HF_ORG_LEGACY'] + "/african_sft"): "machine-translated instruction data; card silent on the system",
    (os.environ['HF_ORG'] + "/tds"): "tagged translations scored by AfriCOMET at build time; MT system not recorded",
    (os.environ['HF_ORG'] + "/tds_long"): "as tds",
    (os.environ['HF_ORG'] + "/clean_data"): "tagged translations; MT system not recorded",
    (os.environ['HF_ORG'] + "/clean_data2"): "tagged translations; MT system not recorded",
    (os.environ['HF_ORG'] + "/translation"): "tagged translations; MT system not recorded",
    (os.environ['HF_ORG'] + "/processed_data"): "tagged translations scored by AfriCOMET; MT system not recorded",
    (os.environ['HF_ORG_LEGACY'] + "/Clean_translate"): "tagged translations; MT system not recorded",
    (os.environ['HF_ORG'] + "/alpaca-hausa-cleaned"): "translated Alpaca; MT system not stated",
    (os.environ['HF_ORG'] + "/alpaca-igbo-cleaned"): "translated Alpaca; MT system not stated",
    (os.environ['HF_ORG'] + "/alpaca-yoruba-cleaned"): "translated Alpaca; MT system not stated",
    (os.environ['HF_ORG'] + "/alpaca_hausa_taco"): "translated Alpaca (TaCo); MT system not stated",
    (os.environ['HF_ORG'] + "/alpaca_igbo_taco"): "translated Alpaca (TaCo); MT system not stated",
    (os.environ['HF_ORG'] + "/alpaca_yoruba_taco"): "translated Alpaca (TaCo); MT system not stated",
    "saillab/alpaca-ewe-cleaned": "translated Alpaca; MT system not stated",
    "saillab/alpaca_ewe_taco": "translated Alpaca (TaCo); MT system not stated",
    (os.environ['HF_ORG'] + "/muri-it-language-split"): "MURI-IT reverse instructions; translation step not stated on the card",
    "African-Languages-Lab/multi-open": "translation dataset with a quality score; MT system not stated",
    "sentence-transformers/parallel-sentences-ccmatrix": "mined (CCMatrix) en-<African language> pairs; Malagasy (mlg) has no AfriCOMET-QE coverage so it is used unscored; "
                                                         "afr / swh passed the 200-pair QE sample, hau / ibo / som / xho were below the bar and are not built",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--bins", type=Path, default=None)
    ap.add_argument("--license-scan", type=Path, default=None)
    ap.add_argument("--provenance", type=Path, default=None)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    from sft_new.ext_policy import EXCLUDED_REPOS, HELD, HELD_PREFIXES

    a.dest.mkdir(parents=True, exist_ok=True)
    inv = json.loads((a.final / "inventory.json").read_text())
    ds_meta = json.loads((a.final / "datasets.json").read_text())
    man = json.loads((a.bins / "manifest.json").read_text()) if a.bins and (a.bins / "manifest.json").exists() else None
    audit = json.loads((a.out / "audit.json").read_text()) if (a.out / "audit.json").exists() else {}
    extra = json.loads((a.out / "audit_extra.json").read_text()) if (a.out / "audit_extra.json").exists() else {}
    prov = json.loads(a.provenance.read_text()) if a.provenance and a.provenance.exists() else {}

    # --- QE audit by dataset
    qe = collections.defaultdict(lambda: {"n": 0, "sum": 0.0, "low": 0.0, "groups": 0})
    for src in (audit, extra):
        for k, v in src.items():
            ds = k.split("|")[0].replace(" (pivot response)", "")
            q = qe[ds]
            q["n"] += v["n"]; q["sum"] += v["mean"] * v["n"]; q["low"] += v["share<0.46"] * v["n"]; q["groups"] += 1
    qe_sum = {ds: {"pairs_scored": q["n"], "mean_qe": round(q["sum"] / q["n"], 3), "share_below_0.46": round(q["low"] / q["n"], 3), "groups": q["groups"]}
              for ds, q in qe.items() if q["n"]}
    (a.dest / "qe_audit_summary.json").write_text(json.dumps(dict(sorted(qe_sum.items(), key=lambda kv: kv[1]["mean_qe"])), indent=1))

    # --- unverified provenance log
    rows_by = inv["by_source_dataset"]
    unv = {}
    for repo, why in UNVERIFIED.items():
        n = sum(v for k, v in rows_by.items() if k.split(":")[0].split("@")[0] == repo)
        if n:
            q = next((v for k, v in qe_sum.items() if k.split(":")[0] == repo), None)
            unv[repo] = {"rows_in_final": n, "why_unverified": why, "qe_sample": q, "decision": "used; logged only (owner 2026-10-10)"}
    (a.dest / "provenance_unverified.json").write_text(json.dumps(unv, indent=1, ensure_ascii=False))

    # --- licence scan + what happened to each repo
    scan = json.loads(a.license_scan.read_text()) if a.license_scan and a.license_scan.exists() else {"datasets": {}}
    ds = scan.get("datasets", {})
    used_repos = collections.Counter()
    for k, v in rows_by.items():
        used_repos[k.split(":")[0].split("@")[0]] += v
    outcome = {}
    for repo, rec in ds.items():
        status = "built_into_final" if used_repos.get(repo) else ("held" if any(repo in str(w) for w in HELD.values()) else "not_used")
        if repo in EXCLUDED_REPOS:
            status = "excluded: " + EXCLUDED_REPOS[repo]
        elif rec.get("decision") == "skip":
            status = "skipped (non-commercial licence)"
        outcome[repo] = {**rec, "rows_in_final": used_repos.get(repo, 0), "status": status}
    skipped = {r: {"licence": v.get("declared"), "gb": round((v.get("bytes") or 0) / 1e9, 2)} for r, v in outcome.items() if v["status"].startswith("skipped")}
    (a.dest / "license_scan.json").write_text(json.dumps({"summary": scan.get("summary"), "skipped_non_commercial_with_size": skipped,
                                                          "held_steps": inv.get("held_out_of_the_mix"), "datasets": outcome}, indent=1, ensure_ascii=False))

    # --- markdown
    L = ["# SFT build, October 2026", "",
         f"Rows after global dedupe: **{inv['rows_final']:,}** (from {inv['rows_in']:,}; {inv['duplicates_removed']:,} duplicates removed), "
         f"{inv['datasets']} dataset folders.", ""]
    if man:
        toks = sum(v["tokens"] for v in man["datasets"].values())
        L += [f"Tokenised with `{man['tokenizer']}` @ `{str(man.get('revision'))[:10]}`: **{toks / 1e9:.2f}B** training tokens "
              f"(+ {sum(v['val_tokens'] for v in man['datasets'].values()) / 1e6:.0f}M validation).", ""]
    L += ["## Rows per source language x task (top tasks)", "", "| language | rows | top tasks |", "|---|---|---|"]
    for lang, tasks in list(inv["by_source_language_task"].items())[:40]:
        top = ", ".join(f"{t} {n:,}" for t, n in list(tasks.items())[:5])
        L.append(f"| {lang} | {sum(tasks.values()):,} | {top} |")
    L += ["", "## Held out of the mix", "", "| step | rows | reason |", "|---|---|---|"]
    for name, h in sorted((inv.get("held_out_of_the_mix") or {}).items(), key=lambda kv: -kv[1]["rows"]):
        L.append(f"| {name} | {h['rows']:,} | {h['reason']} |")
    L += ["", "## Skipped for a non-commercial licence", "", "| repo | licence | size |", "|---|---|---|"]
    for r, v in sorted(skipped.items(), key=lambda kv: -kv[1]["gb"]):
        L.append(f"| {r} | {v['licence']} | {v['gb']} GB |")
    L += ["", "## AfriCOMET-QE samples (about 200 pairs per dataset, not full QE)", "", "| dataset | pairs | mean | below 0.46 |", "|---|---|---|---|"]
    for k, v in sorted(qe_sum.items(), key=lambda kv: kv[1]["mean_qe"])[:40]:
        L.append(f"| {k} | {v['pairs_scored']} | {v['mean_qe']} | {v['share_below_0.46']:.0%} |")
    L += ["", "## Used, machine-translation system not verifiable (logged only)", "", "| repo | rows | note |", "|---|---|---|"]
    for r, v in sorted(unv.items(), key=lambda kv: -kv[1]["rows_in_final"]):
        L.append(f"| {r} | {v['rows_in_final']:,} | {v['why_unverified']} |")
    (a.dest / "SFT_BUILD.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"report written to {a.dest}")


if __name__ == "__main__":
    main()
