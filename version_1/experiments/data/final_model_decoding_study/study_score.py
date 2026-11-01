"""a1 decoding study, step 3 (venv_gemma, CPU): decode every candidate, MBR, metrics per config x language, anchor recall, loops, bootstrap.
usage: study_score.py [groups.json]   (groups.json: {iso: "strong|medium|weak"}; default: defined from the greedy run of this study)"""
import glob, json, os, re, sys, time, unicodedata
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF, BLEU
from transformers import AutoTokenizer
from gemma_mt import format as fmt
from gemma_mt.rl.text import decode_completion
from gemma_mt.rl import guards, lid as L
from gemma_mt.languages import LANGUAGES

D = "/workspace/jobs/a1/study"
S = json.load(open(D + "/study_set.json"))
rows, stop = S["rows"], S["stop_ids"]
tok = AutoTokenizer.from_pretrained(S["model_path"])
N = len(rows)
isos = list(dict.fromkeys(r["iso"] for r in rows))
idx_of = {i: [k for k, r in enumerate(rows) if r["iso"] == i] for i in isos}
nfc = lambda s: unicodedata.normalize("NFC", s)

def decode_cfg(g):
    out = []
    for k, cs in enumerate(g["cands"]):
        row = []
        for ids in cs:
            d = decode_completion(tok, ids, stop)
            st = fmt.strip_iso_prefix(d.text, rows[k]["iso"])
            row.append({"hyp": st.text, "finished": bool(d.finished), "ntok": len(ids), "prefix": st.status})
        out.append(row)
    return out

_chrf = CHRF(word_order=2)
def _sent_chrf(a, b): return _chrf.sentence_score(a, [b]).score
def mbr_row(args):
    hyps, evid = args
    best, bs = 0, -1.0
    for i, h in enumerate(hyps):
        sc = [_sent_chrf(h, e) for j, e in enumerate(evid) if j != i]
        m = sum(sc) / max(len(sc), 1)
        if m > bs: bs, best = m, i
    return best

NUM = re.compile(r"\d+(?:[.,]\d+)*")
CAPW = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][\w'’-]*\b")
def anchors(src):
    nums = NUM.findall(src)
    names = [m for m in CAPW.findall(src) if m not in ("I",) and len(m) > 1]
    return nums, names
def has_num(tok_, text):
    return tok_ in text or re.sub(r"[.,]", "", tok_) in re.sub(r"(?<=\d)[.,](?=\d)", "", text)
def has_name(n, text): return n.casefold() in text.casefold()

def main():
    t_all = time.time()
    gens = {}
    for p in sorted(glob.glob(D + "/gen_*.json")):
        g = json.load(open(p)); gens[g["cfg"]] = g
    print("configs:", list(gens), flush=True)
    dec = {c: decode_cfg(g) for c, g in gens.items()}
    # --- MBR configs
    mbr_cfgs = {}
    with Pool(4) as pool:
        for src in ("s8T0.7", "s8T0.5"):
            if src not in dec: continue
            t0 = time.time()
            hy = [[c["hyp"] for c in row] for row in dec[src]]
            pick = pool.map(mbr_row, [(h, h) for h in hy], chunksize=32)
            mbr_cfgs["mbr8_" + src[2:]] = (src, pick, time.time() - t0, None)
            if "greedy" in dec:   # greedy joins the pool (candidates and evidence)
                hy9 = [h + [dec["greedy"][k][0]["hyp"]] for k, h in enumerate(hy)]
                pick9 = pool.map(mbr_row, [(h, h) for h in hy9], chunksize=32)
                mbr_cfgs["mbr9g_" + src[2:]] = (src, pick9, time.time() - t0, True)
    for name, (src, pick, dt, with_greedy) in mbr_cfgs.items():
        newdec, g0 = [], gens[src]
        for k in range(N):
            if with_greedy and pick[k] == len(dec[src][k]): newdec.append([dec["greedy"][k][0]])
            else: newdec.append([dec[src][k][pick[k]]])
        dec[name] = newdec
        gens[name] = {"cfg": name, "engine": "vllm+cpu", "seconds": g0["seconds"] + (gens["greedy"]["seconds"] if with_greedy else 0) + dt, "tokens": g0["tokens"] + (gens["greedy"]["tokens"] if with_greedy else 0),
                      "kw": {"mbr_from": src, "with_greedy": bool(with_greedy), "mbr_cpu_seconds_4procs": dt}, "n_prompts": N}
    # --- per config hypotheses (first candidate = the output; sampling configs: one sample)
    cfgs = [c for c in dec if c not in ("s8T0.7", "s8T0.5")]
    hyp = {c: [dec[c][k][0]["hyp"] for k in range(N)] for c in cfgs}
    fin = {c: np.array([dec[c][k][0]["finished"] for k in range(N)]) for c in cfgs}
    ntok = {c: np.array([dec[c][k][0]["ntok"] for k in range(N)]) for c in cfgs}
    refs = [nfc(r["reference"]) for r in rows]
    json.dump({c: hyp[c] for c in cfgs}, open(D + "/hyps.json", "w"), ensure_ascii=False)
    # --- LID
    lidbin = __import__("glob").glob("/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")[0]
    clf = L.GlotLIDClassifier(path=lidbin); known = set(clf.labels())
    acc = {}
    for i in isos:
        own = {x for x in LANGUAGES[i].glotlid if x in known}
        acc[i] = own | {x for x in L.siblings_of(tuple(own)) if x in known} if own else None
    # --- per-sentence stats
    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    sc, sb, per_row = {}, {}, {}
    rnum = [anchors(r["english"]) for r in rows]
    ref_keep = []
    for k, r in enumerate(rows):
        nums, names = rnum[k]
        ref_keep.append(([n for n in nums if has_num(n, refs[k])], [n for n in names if has_name(n, refs[k])]))
    for c in cfgs:
        H = [nfc(h) for h in hyp[c]]
        sc[c] = {i: np.array(chrf._extract_corpus_statistics([H[k] for k in idx_of[i]], [[refs[k] for k in idx_of[i]]]), dtype=float) for i in isos}
        sb[c] = {i: np.array(bleu._extract_corpus_statistics([H[k] for k in idx_of[i]], [[refs[k] for k in idx_of[i]]]), dtype=float) for i in isos}
        lab = clf(H)
        pr = []
        for k in range(N):
            h = H[k]
            nk, mk = ref_keep[k]
            pr.append({"fin": bool(fin[c][k]), "loop": bool((not fin[c][k]) or guards.is_repetitive(h) or guards.is_self_repetitive(h)),
                       "empty": not h.strip(), "copy": guards.is_copy_of_source(rows[k]["english"], h),
                       "num_kept": sum(has_num(n, h) for n in nk), "num_tot": len(nk), "name_kept": sum(has_name(n, h) for n in mk), "name_tot": len(mk),
                       "lid": (None if acc[rows[k]["iso"]] is None else (lab[k] in acc[rows[k]["iso"]])), "len_ratio": len(h) / max(len(refs[k]), 1), "ntok": int(ntok[c][k])})
        per_row[c] = pr
    # reference anchor ceiling (raw anchors, reference)
    raw_tot = {"num": sum(len(a[0]) for a in rnum), "name": sum(len(a[1]) for a in rnum)}
    ref_tot = {"num": sum(len(a[0]) for a in ref_keep), "name": sum(len(a[1]) for a in ref_keep)}
    # --- groups
    if len(sys.argv) > 1:
        grp = json.load(open(sys.argv[1]))
    else:
        grp = {}
        for i in isos:
            v = chrf._compute_score_from_stats(list(sc["greedy"][i].sum(0))).score
            grp[i] = "strong" if v >= 40 else ("medium" if v >= 25 else "weak")
    # --- point estimates
    res = {"configs": {}, "groups": grp, "isos": isos, "n_rows": N, "raw_anchor_totals": raw_tot, "ref_kept_anchor_totals": ref_tot,
           "info": S["info"], "model_sha": S["sha"]}
    for c in cfgs:
        per_lang = {}
        for i in isos:
            ix = idx_of[i]; pr = [per_row[c][k] for k in ix]
            lv = [p["lid"] for p in pr if p["lid"] is not None]
            per_lang[i] = {"chrf": chrf._compute_score_from_stats(list(sc[c][i].sum(0))).score, "bleu": bleu._compute_score_from_stats(list(sb[c][i].sum(0))).score,
                           "finished": float(np.mean([p["fin"] for p in pr])), "loop": float(np.mean([p["loop"] for p in pr])), "empty": float(np.mean([p["empty"] for p in pr])),
                           "copy": float(np.mean([p["copy"] for p in pr])), "num_kept": sum(p["num_kept"] for p in pr), "num_tot": sum(p["num_tot"] for p in pr),
                           "name_kept": sum(p["name_kept"] for p in pr), "name_tot": sum(p["name_tot"] for p in pr),
                           "lid": (float(np.mean(lv)) if lv else None), "len_ratio": float(np.mean([p["len_ratio"] for p in pr])), "mean_tokens": float(np.mean([p["ntok"] for p in pr]))}
        g = gens[c]
        res["configs"][c] = {"per_lang": per_lang, "seconds": g["seconds"], "tokens_generated": g["tokens"], "engine": g["engine"], "kw": g.get("kw"),
                             "tok_per_s": g["tokens"] / g["seconds"], "sent_per_s": N / g["seconds"], "batch_size": g.get("batch_size")}
    # --- bootstrap (paired over sentences within language)
    B = 1000
    rng = np.random.default_rng(0)
    samp = {i: rng.integers(0, len(idx_of[i]), size=(B, len(idx_of[i]))) for i in isos}
    bs = {m: np.zeros((B, len(cfgs), len(isos))) for m in ("chrf", "bleu")}
    t0 = time.time()
    for ci, c in enumerate(cfgs):
        for li, i in enumerate(isos):
            A, Bs = sc[c][i], sb[c][i]
            ca = np.cumsum  # noqa
            for b in range(B):
                bs["chrf"][b, ci, li] = chrf._compute_score_from_stats(list(A[samp[i][b]].sum(0))).score
                bs["bleu"][b, ci, li] = bleu._compute_score_from_stats(list(Bs[samp[i][b]].sum(0))).score
    print("bootstrap", time.time() - t0, flush=True)
    np.savez_compressed(D + "/bootstrap.npz", chrf=bs["chrf"], bleu=bs["bleu"], cfgs=np.array(cfgs), isos=np.array(isos))
    groups = {"all": isos, **{g: [i for i in isos if grp[i] == g] for g in ("strong", "medium", "weak")}}
    res["bootstrap"] = {}
    gi = {g: [isos.index(i) for i in v] for g, v in groups.items()}
    base = cfgs.index("greedy")
    for m in ("chrf", "bleu"):
        for g, ix in gi.items():
            if not ix: continue
            mac = bs[m][:, :, ix].mean(2)                # (B, cfg)
            for ci, c in enumerate(cfgs):
                lo, hi = np.percentile(mac[:, ci], [2.5, 97.5])
                dd = mac[:, ci] - mac[:, base]
                dlo, dhi = np.percentile(dd, [2.5, 97.5])
                res["bootstrap"].setdefault(m, {}).setdefault(g, {})[c] = {"point": float(np.mean([res["configs"][c]["per_lang"][i][m] for i in groups[g]])),
                    "lo": float(lo), "hi": float(hi), "d_vs_greedy": float(np.mean(dd)), "d_lo": float(dlo), "d_hi": float(dhi), "p_better": float(np.mean(dd > 0))}
    res["groups_members"] = groups
    # example outputs for the report: a handful of rows, all configs
    res["examples"] = [{"iso": rows[k]["iso"], "english": rows[k]["english"], "reference": rows[k]["reference"], "hyps": {c: hyp[c][k] for c in cfgs}} for k in range(0, N, N // 12)]
    json.dump(res, open(D + "/results.json", "w"), ensure_ascii=False)
    print("done", time.time() - t_all, flush=True)
main()
