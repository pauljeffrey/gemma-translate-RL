"""a1 decoding study, step 1 (venv_gemma): pick 20 languages x <=200 no_pair sentences, build prompt ids with the final model's tokenizer."""
import json, os, random, sys
sys.path.insert(0, "/workspace/sft_build/pylibs")
from gemma_mt import format as fmt
from gemma_mt.eval import benchmarks as B, contamination as C, runtime as RT
from gemma_mt.languages import LANGUAGES

OUT = "/workspace/jobs/a1/study"
os.makedirs(OUT, exist_ok=True)
LANGS = "swh hau yor ibo zul amh kin som wol sna lug bem tir kbp nus kab fon ach nbl efi".split()
N = 200
ROOT = "/workspace/sft_build/bench/test_sets"
CDIR = ROOT + "/_contamination"
rows_all = []
for b in ("flores_plus", "smol_sent"):
    isos = [i for i in LANGS if getattr(LANGUAGES[i], b if b != "flores_plus" else "flores")]
    if b == "smol_sent":
        isos = [i for i in LANGS if not LANGUAGES[i].flores]
    rows = B.load_benchmark(b, path=os.path.join(ROOT, b), isos=isos)
    C.apply_flags(rows, C.default_flag_files(CDIR, [b]))
    rows_all += rows
by = {}
for r in rows_all:
    by.setdefault(r["iso"], []).append(r)
ref = RT.resolve_model((os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), token=os.environ.get("HF_TOKEN"), revision="latest")
tok, stop, pad = RT.load_tokenizer_and_stops(ref.path)
sel = []
info = {}
for iso in LANGS:
    rs = [r for r in by[iso] if r.get("pair_seen") is False]
    rng = random.Random(1000 + sum(map(ord, iso)))
    rng.shuffle(rs)
    rs = sorted(rs[:N], key=lambda r: int(r["id"]) if str(r["id"]).isdigit() else str(r["id"]))
    info[iso] = {"benchmark": rs[0]["benchmark"], "n": len(rs), "n_no_pair_available": len([r for r in by[iso] if r.get("pair_seen") is False]),
                 "n_total": len(by[iso])}
    for r in rs:
        sel.append({"iso": iso, "benchmark": r["benchmark"], "id": str(r["id"]), "english": r["english"], "reference": r["reference"],
                    "prompt": fmt.build_prompt(r["english"], iso, tok), "contaminated": r.get("contaminated")})
json.dump({"model_path": ref.path, "sha": ref.sha, "stop_ids": list(stop.ids), "pad_id": pad, "rows": sel, "info": info},
          open(OUT + "/study_set.json", "w"), ensure_ascii=False)
print(json.dumps(info, indent=0)); print(len(sel), "rows; model", ref.sha, "stop", stop.ids, "pad", pad)
