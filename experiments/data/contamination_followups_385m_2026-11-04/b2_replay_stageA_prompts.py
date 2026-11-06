"""E-07(b) step 2: rebuild the stage-A prompt pool index (same file order / drop rules as gemma_mt.rl.data.load_rl_prompts) from the
metadata columns, validate it against the stage-A group_stats.jsonl (steps 41-400: exact prompt index, language, bin, domain), replay the
deterministic PromptSchedule (seed 1234, language weights of the run) for steps 1-40 (pool without the reference domain, no dynamic sampling,
64 prompts per step) and for steps 41-400 (pool with the reference domain, 71 prompts per step = ceil(64*1.1), the surplus is generated but not trained on),
and count how many drawn prompts are one of the 2,279 flagged windows (hit_texts.json) in the 7 unfiltered new-language files.
Only numpy / pyarrow needed.  Usage: python b2_replay_stageA_prompts.py"""
import os, sys, json, glob, re, math, collections
import numpy as np, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/path/to/repo"
ARC = f"{ROOT}/research_archive_2026-11-03"
META = f"{HERE}/pool_meta"
SEVEN = ["tll", "mfe", "guw", "lub", "crs", "zne", "lue"]
BINS = ("b64", "b256", "b512", "b1024", "b2048"); EDGES = {"b64": (0, 65), "b256": (65, 257), "b512": (257, 513), "b1024": (513, 1025), "b2048": (1025, 2049)}
def bin_of_tokens(n):
    n = 0 if n is None else int(n)
    for b in BINS:
        lo, hi = EDGES[b]
        if lo <= n < hi: return b
    return BINS[-1]
# --- PromptSchedule: the class source of the run's data.py, executed verbatim (numpy only; canonical_iso is the identity here)
src = open(f"{ARC}/clean_rerun_misc/tree/gemma_mt/gemma_mt/rl/data.py").read()
i0 = src.index("class PromptSchedule"); cls_src = src[i0:]
ns = {"np": np, "Dict": dict, "Sequence": list, "Optional": __import__("typing").Optional, "List": list, "Tuple": tuple, "MIXED": "mixed", "LENGTH_BINS": BINS, "canonical_iso": lambda x: x}
import typing; ns.update({k: getattr(typing, k) for k in ("Dict", "Sequence", "Optional", "List", "Tuple", "Any")})
exec(cls_src, ns); PromptSchedule = ns["PromptSchedule"]

LANGS81 = ['ach','afr','amh','bam','bas','bem','bin','crs','dua','dyu','efi','ewe','fon','guw','hau','her','ibo','ish','iso','kab','kam','kbp','kea','kik','kin','kmb','kon','kqn','kri','kua','lin','loz','lua','lub','lue','lug','lun','luo','mfe','mlg','mos','nbl','ndc','nso','nus','nya','nyn','oke','orm','pcm','run','sag','seh','shi','sid','sna','som','sot','ssw','swc','swh','tir','tiv','tll','tsc','tsn','tso','tum','twi','umb','urh','ven','vmw','wal','wes','wol','xho','yao','yor','zne','zul']
syml = [l.split(" -> ")[0].strip() for l in open(f"{ARC}/clean_rerun_misc/prompts_symlinks.txt") if " -> " in l]
general = sorted(os.path.basename(s)[:-8] for s in syml)

def load_meta(name):
    t = pq.read_table(f"{META}/{name}.parquet")
    d = {c: t.column(c).to_pylist() for c in t.column_names}
    return d

import hashlib
SPD = os.environ.get("POOL_ENGLISH_DIR", "/path/to/pool_english")
def split_fraction(key, seed):
    d = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") / 2.0 ** 64

def build_pool(with_ref):
    """list of (path-key, source) in the loader's order; returns dict of columns + per-file slices"""
    files = []
    for iso in general: files.append((f"clean_v3_385m/{iso}.parquet", ("general", iso)))
    for iso in SEVEN: files.append((f"clean_v3_385m/{iso}.parquet", ("own7", iso)))
    for f in glob.glob(f"{META}/news__*.parquet"): files.append((f"clean_v3_385m/news/{os.path.basename(f)[6:-8]}.parquet", ("news", os.path.basename(f)[6:-8])))
    if with_ref:
        for f in glob.glob(f"{META}/ref__*.parquet"): files.append((f"clean_v3_385m/ref/{os.path.basename(f)[5:-8]}.parquet", ("ref", os.path.basename(f)[5:-8])))
    files.sort(key=lambda x: x[0])
    iso_l, bin_l, dom_l, src_l, frac_l = [], [], [], [], []
    slices, dropped = {}, collections.Counter(); rowpos = {}
    n = 0
    for key, (kind, iso0) in files:
        if kind == "own7":
            t = pq.read_table(f"{ARC}/clean_rerun_misc/prompts/unfiltered_new7/{iso0}.parquet", columns=["iso", "src_afr_tokens", "length_bin", "source_dataset", "english"])
            d = {c: t.column(c).to_pylist() for c in t.column_names}
        else:
            d = load_meta(f"{kind}__{iso0}")
            d["english"] = pq.read_table(f"{SPD}/{kind}__{iso0}.parquet").column("english").to_pylist()
            assert len(d["english"]) == len(d["iso"])
        N = len(d["iso"]); start = n; kept = 0; rp = []
        lbl = d.get("length_bin") or [None] * N
        for r in range(N):
            iso = d["iso"][r]
            if iso not in LANGS81: dropped["not_rl_supported"] += 1; continue
            nt = d["src_afr_tokens"][r]
            if nt is not None and int(nt) > 256: dropped["source_too_long"] += 1; continue
            lb = lbl[r]
            lb = lb if lb in BINS else ""
            en = d["english"][r]
            if not isinstance(en, str) or not en.strip(): dropped["empty_english"] += 1; continue
            iso_l.append(iso); bin_l.append(lb or bin_of_tokens(nt)); frac_l.append(split_fraction(en.strip(), 1234))
            dm = d["domain"][r] if "domain" in d else ""
            dom_l.append(dm if isinstance(dm, str) and dm else "")
            src_l.append(d["source_dataset"][r] if "source_dataset" in d else "")
            kept += 1; rp.append(r)
        rowpos[(kind, iso0)] = rp
        slices[(kind, iso0)] = (start, start + kept, key)
        n += kept
    dom_l = [x or "general" for x in dom_l]
    return holdout(dict(iso=iso_l, bin=bin_l, dom=dom_l, src=src_l, frac=frac_l, slices=slices, rowpos=rowpos, dropped=dropped, n=n))

def holdout(P, n_eval=128, min_per_bin=16):
    """train_rl.split_eval_prompts: the n_eval prompts with the smallest hash of the English text are held out (+ min_per_bin top-up per bin); positions are re-indexed after removal."""
    frac = P["frac"]; order = sorted(range(len(frac)), key=lambda i: (frac[i], i)); ev = order[:min(n_eval, len(frac) // 10)]; evset = set(ev)
    have = collections.Counter(P["bin"][i] for i in ev)
    for b in sorted(set(P["bin"])):
        need = min_per_bin - have.get(b, 0)
        if need > 0:
            cand = [i for i in order if P["bin"][i] == b and i not in evset]
            # (copies of the same English text are held out together in the original; the top-up needs english, not stored: assert it is not needed)
            raise RuntimeError(f"bin {b} needs a top-up of {need}: implement with english texts")
    keep = [i for i in range(len(frac)) if i not in evset]
    newidx = {old: new for new, old in enumerate(keep)}
    P["evaluation_rows"] = sorted(ev); P["old_of_new"] = keep; P["new_of_old"] = newidx
    for k in ("iso", "bin", "dom", "src", "frac"): P[k] = [P[k][i] for i in keep]
    P["n"] = len(keep)
    return P

def cfg_schedule(pool, seed=1234, with_ref=True):
    lw = json.load(open(f"{ARC}/runs_385m/A/LANG_WEIGHTS.state.json"))["history"][0]["weights"]
    cfg = json.load(open(f"{ARC}/runs_385m/A/config.json"))["config"]
    lwc = cfg["language_weights"]
    # LANG_WEIGHTS.state.json weights at step 0 vs config language_weights: the file wins for the draw (file control on); they must agree
    assert {k: v for k, v in lw.items() if v != 1.0} == {k: float(v) for k, v in lwc.items() if v != 1.0}, "language weight mismatch"
    sched = PromptSchedule(pool["iso"], cfg["language_alpha"], cfg["seed"], length_bin=pool["bin"], curriculum={"b64": 400 + 200}, language_weights=lw,
                           domain=pool["dom"], domain_weights=({k: v for k, v in cfg["domain_weights"].items() if with_ref or k != "ref"}), domain_bin_weights=None, floor_steps=cfg["language_floor_steps"], floor_prompts=cfg["language_floor_prompts"])
    return sched, cfg

if __name__ == "__main__":
    flagged = set(json.load(open(f"{ARC}/clean_rerun_misc/hit_texts.json")))
    gs = [json.loads(l) for l in open(f"{ARC}/runs_385m/A/group_stats.jsonl")]
    out = {}
    # ---------- pool for steps 41-400 (with ref) and validation against group_stats
    P41 = build_pool(True); print("pool with ref:", P41["n"], dict(P41["dropped"]), "expected 4928554", flush=True)
    P1 = build_pool(False); print("pool without ref:", P1["n"], dict(P1["dropped"]), "expected 4917552", flush=True)
    out["pool_rows"] = {"with_ref": P41["n"], "without_ref": P1["n"], "expected_with_ref": 4928554, "expected_without_ref": 4917552}
    bad = 0
    for x in gs:
        p = x["prompt"]
        if not (P41["iso"][p] == x["iso"] and P41["bin"][p] == x["bin"] and P41["dom"][p] == x["domain"]): bad += 1
    out["group_stats_index_check"] = {"records": len(gs), "mismatches": bad}
    print("group_stats index check:", out["group_stats_index_check"], flush=True)
    # ---------- flagged membership of the 7 unfiltered files (by position inside the pool)
    own = {}
    for iso in SEVEN:
        t = pq.read_table(f"{ARC}/clean_rerun_misc/prompts/unfiltered_new7/{iso}.parquet", columns=["english", "iso", "src_afr_tokens", "length_bin"]).to_pylist()
        own[iso] = t
    def pool_flag_sets(P):
        fl = {}  # NEW global index (after the hold-out) -> (iso, english, bin)
        info = {}; ev_flagged = collections.Counter()
        for iso in SEVEN:
            s, e, key = P["slices"][("own7", iso)]
            rows = [r for r in own[iso] if r["iso"] in LANGS81 and (r["src_afr_tokens"] is None or int(r["src_afr_tokens"]) <= 256)]
            rows = [r for r in rows if isinstance(r["english"], str) and r["english"].strip()]
            assert len(rows) == e - s, (iso, len(rows), e - s)
            for k, r in enumerate(rows):
                if r["english"] in flagged or r["english"].strip() in flagged:
                    old = s + k
                    if old in P["new_of_old"]: fl[P["new_of_old"][old]] = (iso, r["english"].strip(), P["bin"][P["new_of_old"][old]])
                    else: ev_flagged[iso] += 1
            info[iso] = (s, e)
        P["ev_flagged"] = dict(ev_flagged)
        return fl, info
    fl41, info41 = pool_flag_sets(P41); fl1, info1 = pool_flag_sets(P1)
    print("flagged rows in the loaded pool:", len(fl41), len(fl1), flush=True)
    out["flagged_rows_in_pool"] = {"with_ref": len(fl41), "without_ref": len(fl1)}
    out["flagged_by_lang_in_pool"] = {iso: sum(1 for v in fl41.values() if v[0] == iso) for iso in SEVEN}
    out["flagged_by_lang_bin_in_pool"] = {iso: dict(collections.Counter(v[2] for v in fl41.values() if v[0] == iso)) for iso in SEVEN}
    out["rows_by_lang_bin_in_pool"] = {iso: dict(collections.Counter(b for i, b in zip(P41["iso"], P41["bin"]) if i == iso)) for iso in SEVEN}
    out["holdout_rows"] = {"with_ref": len(P41["evaluation_rows"]), "without_ref": len(P1["evaluation_rows"]), "flagged_rows_in_holdout_with_ref": P41["ev_flagged"], "flagged_rows_in_holdout_without_ref": P1["ev_flagged"]}
    # ---------- steps 41-400 from group_stats (drawn = all 71 per step; trained-on = the groups the update used is not logged, so 'drawn')
    drawn = collections.Counter(); hits = collections.Counter(); hit_prompts = collections.defaultdict(list)
    for x in gs:
        if x["iso"] in SEVEN:
            drawn[x["iso"]] += 1
            if x["prompt"] in fl41: hits[x["iso"]] += 1; hit_prompts[x["iso"]].append((x["step"], x["prompt"]))
    out["steps41_400"] = {"drawn": dict(drawn), "flagged_drawn": dict(hits), "draws_total_all_langs": len(gs)}
    # ---------- steps 1-40 replay (0-based loop step s draws record 's+1'): find the convention with the group_stats of 41 first
    sch41, cfg = cfg_schedule(P41)
    # convention check on steps 41-400 (the draw there uses the pool with ref and n_draw = 71)
    conv = {}
    for off in (0, 1):
        ok = tot = 0
        for step1 in (41, 42, 60, 100, 200, 400):
            idx = sch41.indices(step1 - 1 + off, 71)
            rec = [x["prompt"] for x in gs if x["step"] == step1]
            tot += 1; ok += int(idx == rec)
        conv[off] = (ok, tot)
    print("schedule convention check (matches of 6 steps) for loop-step = record-step -1 + off:", conv, flush=True)
    off = max(conv, key=lambda k: conv[k][0])
    out["schedule_replay_check_steps41_400"] = {"convention_offset": off, "exact_step_matches_of_6": conv[off][0]}
    # full check of all 360 steps
    full_ok = sum(int(sch41.indices(s - 1 + off, 71) == [x["prompt"] for x in gs if x["step"] == s]) for s in range(41, 401))
    out["schedule_replay_check_steps41_400"]["exact_steps_of_360"] = full_ok
    print("full replay check steps 41-400:", full_ok, "of 360", flush=True)
    sch1, _ = cfg_schedule(P1, with_ref=False)
    drawn1 = collections.Counter(); hits1 = collections.Counter(); hp1 = collections.defaultdict(list)
    all1 = collections.Counter()
    for s in range(1, 41):
        for p in sch1.indices(s - 1 + off, 64):
            all1[P1["iso"][p]] += 1
            if P1["iso"][p] in SEVEN:
                drawn1[P1["iso"][p]] += 1
                if p in fl1: hits1[P1["iso"][p]] += 1; hp1[P1["iso"][p]].append((s, p))
    out["steps1_40"] = {"drawn": dict(drawn1), "flagged_drawn": dict(hits1), "draws_total_all_langs": sum(all1.values())}
    # check against the run's own counters (archive seen:<iso> / 16) and the logged step samples
    import sqlite3
    cn = dict(sqlite3.connect(f"{ARC}/runs_385m/A/archive/archive.sqlite").execute("select name,value from counters").fetchall())
    out["archive_counter_seen_prompts"] = {iso: cn[f"seen:{iso}"] / 16 for iso in SEVEN}
    out["replay_vs_counter"] = {iso: [drawn1.get(iso, 0) + drawn.get(iso, 0), cn[f"seen:{iso}"] / 16] for iso in SEVEN}
    # compare against the logged sample text in rl_A_prev log (steps 1-40)
    logtxt = open(f"{ARC}/clean_rerun_misc/logs/rl_A_prev_1793691697.log").read().splitlines()
    cur = None; sample_ok = sample_tot = 0
    batches = {s: [(P1["iso"][p], p) for p in sch1.indices(s - 1 + off, 64)] for s in range(1, 41)}
    pre = {}
    for ln in logtxt:
        m = re.match(r"\[gemma_mt\] rl step (\d+)/400", ln)
        if m: cur = int(m.group(1)); continue
        m = re.match(r"\[gemma_mt\]   sample \[(\w+)\] '(.*?)' ->", ln)
        if m and cur and cur <= 40:
            iso, pref = m.group(1), m.group(2)
            # the logged english is a 60-char prefix of the prompt text; compare with the pool text for the replayed prompts of that language (only 7-lang texts are loaded; compare iso set and, for the own7, the text)
            sample_tot += 1
            sample_ok += int(any(i == iso for i, p in batches[cur]))
    out["log_sample_iso_in_replayed_batch"] = [sample_ok, sample_tot]
    # exact text check where possible: texts for own7 prompts
    txt_ok = txt_tot = 0
    cur = None
    for ln in logtxt:
        m = re.match(r"\[gemma_mt\] rl step (\d+)/400", ln)
        if m: cur = int(m.group(1)); continue
        m = re.match(r"\[gemma_mt\]   sample \[(\w+)\] '(.*?)' ->", ln)
        if m and cur and cur <= 40 and m.group(1) in SEVEN:
            iso, pref = m.group(1), m.group(2)
            s, e = info1[iso]
            texts = {p: own[iso][0] for p in []}
            rows = [r for r in own[iso] if r["iso"] in LANGS81 and (r["src_afr_tokens"] is None or int(r["src_afr_tokens"]) <= 256)]
            txt_tot += 1
            cand = [p for i, p in batches[cur] if i == iso]
            txt_ok += int(any(rows[p - s]["english"].strip().startswith(pref[:50]) for p in cand if s <= p < e))
    out["log_sample_text_check_own7"] = [txt_ok, txt_tot]
    # distinct-window repeat analysis over all 400 steps
    ent = collections.defaultdict(list)
    for iso, L in hp1.items():
        for s, p in L: ent[p].append(s)
    for iso, L in hit_prompts.items():
        for s, p in L: ent[p].append(s)
    # prompt ids differ between the two pools for the same text: map to text
    textof = {}
    for p, v in fl41.items(): textof[("w", p)] = v[1]
    ent_text = collections.defaultdict(list)
    for iso, L in hp1.items():
        for s, p in L: ent_text[fl1[p][1]].append(s)
    for iso, L in hit_prompts.items():
        for s, p in L: ent_text[fl41[p][1]].append(s)
    out["flagged_draws_total"] = sum(len(v) for v in ent_text.values())
    out["flagged_distinct_windows_drawn"] = len(ent_text)
    out["flagged_windows_drawn_more_than_once"] = sum(1 for v in ent_text.values() if len(v) > 1)
    out["flagged_draws_first40_vs_41plus"] = [sum(1 for v in ent_text.values() for s in v if s <= 40), sum(1 for v in ent_text.values() for s in v if s > 40)]
    # per language summary table + the list of drawn flagged windows (ids/steps/lengths, no text) for the data folder
    rows_out = []; pool_n = collections.Counter(P41["iso"])
    for iso in SEVEN:
        d1, d2 = drawn1.get(iso, 0), drawn.get(iso, 0)
        h1, h2 = hits1.get(iso, 0), hits.get(iso, 0)
        pool_fl = out["flagged_by_lang_in_pool"][iso]
        rows_out.append(dict(iso=iso, pool_rows=pool_n[iso], pool_flagged=pool_fl, pool_flagged_share=round(pool_fl / pool_n[iso], 5), drawn_1_40=d1, flagged_1_40=h1, drawn_41_400=d2, flagged_41_400=h2,
                             drawn_total=d1 + d2, flagged_total=h1 + h2, flagged_share_of_drawn=round((h1 + h2) / max(d1 + d2, 1), 5)))
    tot = dict(iso="TOTAL(7)", pool_rows=sum(r["pool_rows"] for r in rows_out), pool_flagged=sum(r["pool_flagged"] for r in rows_out), drawn_1_40=sum(r["drawn_1_40"] for r in rows_out), flagged_1_40=sum(r["flagged_1_40"] for r in rows_out),
               drawn_41_400=sum(r["drawn_41_400"] for r in rows_out), flagged_41_400=sum(r["flagged_41_400"] for r in rows_out))
    tot["pool_flagged_share"] = round(tot["pool_flagged"] / tot["pool_rows"], 5); tot["drawn_total"] = tot["drawn_1_40"] + tot["drawn_41_400"]; tot["flagged_total"] = tot["flagged_1_40"] + tot["flagged_41_400"]
    tot["flagged_share_of_drawn"] = round(tot["flagged_total"] / tot["drawn_total"], 5)
    rows_out.append(tot)
    out["per_language"] = rows_out
    # total stage-A draws (all languages) for the share of all prompts
    out["all_languages_draws"] = {"steps_1_40": sum(all1.values()), "steps_41_400": len(gs), "total": sum(all1.values()) + len(gs)}
    json.dump(out, open(f"{HERE}/b_stageA_flagged_draws.json", "w"), indent=1, default=str)
    import csv
    with open(f"{HERE}/b_stageA_flagged_draws_by_language.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) + [k for k in rows_out[-1] if k not in rows_out[0]]); w.writeheader(); [w.writerow(r) for r in rows_out]
    with open(f"{HERE}/b_stageA_flagged_draw_list.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "step", "pool", "global_index", "length_bin", "n_chars", "windows_draw_count"])
        for iso, L in hp1.items():
            for s, p in L: w.writerow([iso, s, "no_ref(1-40)", p, fl1[p][2], len(fl1[p][1]), len(ent_text[fl1[p][1]])])
        for iso, L in hit_prompts.items():
            for s, p in L: w.writerow([iso, s, "with_ref(41-400)", p, fl41[p][2], len(fl41[p][1]), len(ent_text[fl41[p][1]])])
    print(json.dumps({k: v for k, v in out.items() if k != "per_language"}, indent=1, default=str))
    for r in rows_out: print(r)
