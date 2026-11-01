#!/usr/bin/env python3
"""Build the H_sel evaluation file (FLORES+ DEV ids of H_sel, English -> every language that has a dev file) for gemma_mt.eval.eval_translation --bench-file.

    HF_TOKEN=... PYTHONPATH=<code> python hsel_eval_set.py --split heldout_split_cal_sel_v2.json --out /workspace/work/eval/hsel.jsonl [--isos amh yor ...]

H_sel is the selection half of the held-out dev ids (Amendment 3, article level, 296 ids): it is used to choose between arms (A2 ablation, NLLB language rule) and is NEVER
in any training or calibration role. The file holds benchmark text: it stays on the instance, in /workspace/work/eval, and is never pushed.
Columns written (one JSON object per line): english, reference, iso, id.
"""
import argparse, json, os, collections
from huggingface_hub import hf_hub_download, HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tmp", default="/tmp/hsel_dl"); ap.add_argument("--isos", nargs="*")
    a = ap.parse_args()
    from gemma_mt import languages as GL
    sel = set(json.load(open(a.split))["h_sel"])
    tok = os.environ["HF_TOKEN"]
    have = {s.rfilename[4:-6] for s in HfApi(token=tok).dataset_info("openlanguagedata/flores_plus").siblings if s.rfilename.startswith("dev/")}

    def load(code):
        p = hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=tok, local_dir=a.tmp)
        d = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if (r.get("text") or "").strip():
                    d[int(r["id"])] = r["text"].strip()
        os.remove(p)
        return d

    eng = load("eng_Latn")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for iso, L in GL.LANGUAGES.items():
            code = L.flores or L.flores_moz
            if not code or code not in have or (a.isos and iso not in a.isos):
                continue
            t = load(code)
            for i in sorted(sel):
                if i in t and i in eng:
                    f.write(json.dumps({"english": eng[i], "reference": t[i], "iso": iso, "id": i}, ensure_ascii=False) + "\n"); n[iso] += 1
    print(sum(n.values()), "rows,", len(n), "languages; fewest:", n.most_common()[-3:])


if __name__ == "__main__":
    main()
