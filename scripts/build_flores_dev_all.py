# SPDX-License-Identifier: AGPL-3.0-or-later
"""Collect the FLORES+ DEV pairs (English -> language) of every RL language into ONE jsonl, for `calibrate_shaping --bench-file`.

    python scripts/build_flores_dev_all.py --out /workspace/sft_build/calibration/flores_dev_all.jsonl

Dev only (the contamination rule of D-24: calibration never touches devtest). Row schema: iso, id, english, reference, url, benchmark, source.
Sources, in this order, for each language of `gemma_mt.languages.SUPPORTED_RL`:
  1. <ssa_cmp>/flores/<flores code>.jsonl   the dev files downloaded for the SSA-COMET study (id, text, url, ...); English = eng_Latn.jsonl
  2. bench_dev.jsonl rows with benchmark == flores_plus_dev (english, reference, id)
  3. the Hub (`openlanguagedata/flores_plus`, dev/<code>.jsonl, read token from HF_TOKEN); Twi has two varieties there
     (twi_Latn_akua1239 / twi_Latn_asan1239): the sentences alternate between them by id parity, so every English sentence appears once
  4. a language without FLORES+ gets MAFAND dev rows when bench_dev.jsonl has some (pcm); efi has neither and is reported as missing
Languages WITH FLORES+ rows also get their MAFAND dev rows (benchmark == mafand_dev): calibrate_shaping uses them only to widen the diacritic
inventory (Yoruba MAFAND writes the under-dot as U+0329, FLORES+ as U+0323), never for the QE anchors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ssa-flores", default="/workspace/sft_build/ssa_cmp/flores")
    ap.add_argument("--bench-dev", default="/workspace/sft_build/bench/bench_dev.jsonl")
    ap.add_argument("--hub-cache", default="/workspace/sft_build/hf")
    a = ap.parse_args()
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL

    en_rows = read_jsonl(os.path.join(a.ssa_flores, "eng_Latn.jsonl"))
    en = {int(r["id"]): r["text"] for r in en_rows}
    url = {int(r["id"]): r.get("url", "") for r in en_rows}
    bench: Dict[tuple, List[dict]] = defaultdict(list)
    if os.path.exists(a.bench_dev):
        for r in read_jsonl(a.bench_dev):
            bench[(r["benchmark"], r["iso"])].append(r)

    def hub(code: str) -> List[dict]:
        from huggingface_hub import hf_hub_download

        os.environ.setdefault("HF_HOME", a.hub_cache)
        p = hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        return read_jsonl(p)

    out: List[dict] = []
    report = {}
    for iso in SUPPORTED_RL:
        code = LANGUAGES[iso].flores
        rows: List[dict] = []
        src = None
        if code and os.path.exists(os.path.join(a.ssa_flores, f"{code}.jsonl")):
            raw = read_jsonl(os.path.join(a.ssa_flores, f"{code}.jsonl"))
            rows = [{"id": int(r["id"]), "text": r["text"]} for r in raw]
            src = "ssa_cmp_flores_dev"
        elif bench.get(("flores_plus_dev", iso)):
            rows = [{"id": int(str(r["id"]).split("-")[-1]), "text": r["reference"], "english": r["english"]} for r in bench[("flores_plus_dev", iso)]]
            bad = sum(1 for r in rows if en.get(r["id"], "").strip() != r["english"].strip())
            assert bad == 0, f"{iso}: {bad} bench_dev rows whose English differs from FLORES+ eng_Latn at that id"
            src = "bench_dev_flores_plus_dev"
        elif code == "twi_Latn":
            ak, asn = hub("twi_Latn_akua1239"), hub("twi_Latn_asan1239")
            by_id = {}
            for k, r in enumerate(ak):
                by_id.setdefault(int(r["id"]), {})[0] = r["text"]
            for r in asn:
                by_id.setdefault(int(r["id"]), {})[1] = r["text"]
            for i, d in sorted(by_id.items()):
                t = d.get(i % 2) or d.get(1 - i % 2)
                if t:
                    rows.append({"id": i, "text": t})
            src = "hub_flores_dev_twi_akua_asan_by_id_parity"
        if rows:
            for r in rows:
                if r["id"] in en and r["text"] and r["text"].strip():
                    out.append({"iso": iso, "id": r["id"], "english": en[r["id"]], "reference": r["text"].strip(), "url": url.get(r["id"], ""),
                                "benchmark": "flores_plus_dev", "source": src})
        if bench.get(("mafand_dev", iso)):
            for k, r in enumerate(bench[("mafand_dev", iso)]):
                out.append({"iso": iso, "id": k, "english": r["english"], "reference": r["reference"], "url": "", "benchmark": "mafand_dev",
                            "source": "bench_dev_mafand_dev"})
            src = (src + "+" if src else "") + "bench_dev_mafand_dev"
        report[iso] = {"flores_code": code, "source": src, "rows": sum(1 for r in out if r["iso"] == iso and r["benchmark"] == "flores_plus_dev"),
                      "mafand_rows": sum(1 for r in out if r["iso"] == iso and r["benchmark"] == "mafand_dev")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report))
    missing = [i for i, v in report.items() if not v["rows"] and not v["mafand_rows"]]
    print(f"wrote {len(out)} rows, {len(report) - len(missing)} languages; no reference data for: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
