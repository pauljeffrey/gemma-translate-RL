#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the compact per-language fluency models used by gemma_mt/rl/fluency.py (D-62).

    python scripts/build_fluency_models.py --train-old texts_train.tsv --train-human mono_train.tsv --flores flores_dev.tsv --mono-test mono_test.tsv \
        --best best_training_per_language.csv --siblings ft_per_language.csv --out artifacts/fluency_models

Per language: training set A (translation targets), B (human monolingual) or C (both) as chosen by FLORES (best_training_per_language.csv; B by default, A when there is no human text), the
character 5-gram model and the word-order model of scripts/fluency_lm.py are trained, PRUNED (grams with count < 3, contexts seen < 3 times, words < 2, bigrams < 2) and stored as sorted hashed
numpy arrays (one npz per language). The z-score statistics come from 300 held-out training sentences; the human-text statistics (mean, std, p05, p10 of comb_z) and the validity AUCs
(shuffle / strip / garble / foreign / sibling, human text vs damaged) from FLORES+ dev (45 languages) or, without FLORES, from the held-out web / Wikipedia test text. Each AUC is computed for the
UNPRUNED validated model and for the compact pruned one, so the cost of pruning is visible. A language is `usable` when shuffle, garble and foreign AUC of the compact model are all >= 0.8.
"""
import argparse, collections, csv, json, math, os, random, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import fluency_lm as F
from gemma_mt.rl import fluency as FL

G = {}
MIN_GRAM, MIN_CTX, MIN_UNI, MIN_BI = 3, 3, 2, 2
FIXED_SIBLINGS = {"zul": ["xho", "nbl", "nde"], "xho": ["zul", "nbl", "nde"], "nbl": ["zul", "xho", "nde"], "nde": ["zul", "xho", "nbl"], "kin": ["run"], "run": ["kin"],
                  "swh": ["swc"], "swc": ["swh"], "bam": ["dyu"], "dyu": ["bam"]}


def load(p):
    by = collections.defaultdict(list)
    for l in open(p, encoding="utf-8"):
        if "\t" in l:
            i, t = l.rstrip("\n").split("\t", 1); by[i].append(t)
    return by


# ----------------------------------------------------------------------------------------------------------------------- pruning (on the dict models of fluency_lm)
def prune_char(lm, min_gram=MIN_GRAM, min_ctx=MIN_CTX):
    for k in range(1, lm.n + 1):
        lm.cnt[k] = collections.Counter({g: c for g, c in lm.cnt[k].items() if c >= min_gram})
        keep = {h for h, c in lm.ctx[k].items() if c >= min_ctx or h == ""}
        lm.ctx[k] = collections.Counter({h: lm.ctx[k][h] for h in keep})
        lm.typ[k] = collections.Counter({h: lm.typ[k][h] for h in keep if h in lm.typ[k]})
    return lm


def prune_word(wm, min_uni=MIN_UNI, min_bi=MIN_BI):
    wm.uni = collections.Counter({w: c for w, c in wm.uni.items() if c >= min_uni})
    wm.bi = collections.Counter({p: c for p, c in wm.bi.items() if c >= min_bi})
    types = collections.Counter()
    for (a, b) in wm.bi: types[a] += 1
    wm.types = types
    return wm


# ----------------------------------------------------------------------------------------------------------------------- dict model -> hashed arrays
def str_hash(s):
    """hash of a character string as gemma_mt.rl.fluency.gram_hashes computes it: sum_j (ord(s[len-1-j]) + 1) * M[j] mod 2^64 (j = 0 is the LAST character)"""
    h = 0
    for j, c in enumerate(reversed(s)):
        h = (h + (ord(c) + 1) * int(FL._M[j])) & FL._MASK
    return h


def sorted_arrays(keys, vals, vdtype=np.uint32):
    k = np.array(keys, dtype=np.uint64); v = np.array(vals, dtype=vdtype)
    o = np.argsort(k, kind="stable"); k, v = k[o], v[o]
    assert len(k) == len(np.unique(k)), "hash collision inside one table"
    return k, v


def to_arrays(lm, wm):
    z = {"V": np.array([lm.V], dtype=np.int64), "N": np.array([wm.N], dtype=np.int64)}
    for k in range(1, lm.n + 1):
        z[f"gk{k}"], z[f"gc{k}"] = sorted_arrays([str_hash(g) for g in lm.cnt[k]], list(lm.cnt[k].values()))
        hs = list(lm.ctx[k].keys())
        z[f"ck{k}"], z[f"ct{k}"] = sorted_arrays([str_hash(h) for h in hs], [lm.ctx[k][h] for h in hs])
        _, z[f"cy{k}"] = sorted_arrays([str_hash(h) for h in hs], [lm.typ[k].get(h, 0) for h in hs])
    ws = list(wm.uni.keys())
    z["uk"], z["uc"] = sorted_arrays([FL.word_hash(w) for w in ws], [wm.uni[w] for w in ws])
    tw = [a for a in wm.types if wm.types[a] > 0]
    z["tk"], z["tv"] = sorted_arrays([FL.word_hash(a) for a in tw], [wm.types[a] for a in tw])
    ps = list(wm.bi.keys())
    z["bk"], z["bc"] = sorted_arrays([FL.bigram_hash(FL.word_hash(a), FL.word_hash(b)) for a, b in ps], [wm.bi[p] for p in ps])
    return z


# ----------------------------------------------------------------------------------------------------------------------- evaluation helpers
def damages(test, chars, pool_all, pool_sib, seed=7):
    rng = random.Random(seed); dmg = {"shuffle": [], "strip": [], "garble": [], "foreign": [], "sibling": []}
    for t in test:
        w = t.split()
        if len(w) >= 4:
            w2 = w[:]; rng.shuffle(w2)
            if w2 != w: dmg["shuffle"].append(" ".join(w2))
        s = F.strip_marks(t)
        if s != t: dmg["strip"].append(s)
        dmg["garble"].append("".join(rng.choice(chars) if rng.random() < 0.15 else c for c in t))
        dmg["foreign"].append(rng.choice(pool_all))
        if pool_sib: dmg["sibling"].append(rng.choice(pool_sib))
    return dmg


def aucs(score_fn, test, dmg):
    H = [score_fn(t) for t in test]; out = {}
    for k, v in dmg.items():
        if len(v) >= 20: out[k] = round(F.auc(H, [score_fn(x) for x in v]), 4)
    return out, np.array(H)


LEVELS = [(3, 3, 2, 2), (2, 2, 1, 1), (2, 1, 1, 1)]        # (min_gram, min_ctx, min_uni, min_bi): the most pruned level whose validity AUCs drop by <= 0.01 against the unpruned model is used
KEYS = ("shuffle", "garble", "foreign", "sibling")         # the AUCs the drop is judged on (strip is at chance level for most languages and not policed by this model)


def work(iso):
    t0 = time.time()
    old, hum = [F_nfc(t) for t in G["old"].get(iso, [])[:3000]], [F_nfc(t) for t in G["human"].get(iso, [])[:12000]]
    choice = G["best"].get(iso) or ("B" if len(hum) >= 500 else "A")
    tr = {"A": old, "B": hum, "C": old + [F_nfc(t) for t in G["human"].get(iso, [])[:9000]]}[choice]
    if len(tr) < 500:
        alt = "A" if len(old) >= 500 else ("B" if len(hum) >= 500 else None)
        if alt is None: return iso, None
        choice, tr = alt, {"A": old, "B": hum}[alt]
    hold, fit = tr[:300], tr[300:]
    # reference = the validated model (grams with count < 2 trimmed, word model whole)
    ref_lm = F.CharLM(5).fit(fit).trim(2); ref_wm = F.WordPMI().fit(fit)
    hc = np.array([ref_lm.score(t) for t in hold]); hw = np.array([x for x in (ref_wm.score(t) for t in hold) if x is not None])
    mu_c, sd_c = float(hc.mean()), float(hc.std() + 1e-6)
    mu_w, sd_w = (float(hw.mean()), float(hw.std() + 1e-6)) if len(hw) > 20 else (0.0, 1.0)
    flores = iso in G["flores"] and len(G["flores"][iso]) >= 200
    test = [F_nfc(t) for t in (G["flores"][iso] if flores else G["mono_test"].get(iso, []))][:400]
    if len(test) < 100: return iso, {"skip": "no test text", "choice": choice}
    sib_iso = (G["sib"].get(iso) or [None])[0]
    pool_sib = [F_nfc(t) for t in ((G["flores"].get(sib_iso) if sib_iso in G["flores"] else None) or G["mono_test"].get(sib_iso) or [])][:400] if sib_iso else []
    pool_all = [t for i, ts in G["flores"].items() if i != iso for t in ts[:5]] if G["flores"] else [t for i, ts in G["mono_test"].items() if i != iso for t in ts[:5]]
    dmg = damages(test, sorted(set("".join(fit[:300]))), pool_all, pool_sib)

    def ref_comb(t):
        c = (ref_lm.score(t) - mu_c) / sd_c; w = ref_wm.score(t)
        return c if w is None else (c + (w - mu_w) / sd_w) / math.sqrt(2.0)
    auc_ref, _ = aucs(ref_comb, test, dmg)
    chk = test[:40]
    for lvl in LEVELS:
        pr_lm = prune_char(F.CharLM(5).fit(fit), lvl[0], lvl[1]); pr_wm = prune_word(F.WordPMI().fit(fit), lvl[2], lvl[3])
        z = to_arrays(pr_lm, pr_wm)
        meta = {"order": 5, "char": {"mu": mu_c, "sd": sd_c}, "word": {"mu": mu_w, "sd": sd_w}}
        comp = FL.CompactLanguage(z, meta)
        dev = max(abs(comp.char_score(t) - pr_lm.score(t)) for t in chk)
        wdev = max([abs(comp.word_score(t) - pr_wm.score(t)) for t in chk if pr_wm.score(t) is not None] or [0.0])
        auc_cmp, Hc = aucs(lambda t: comp.score(t)["comb_z"], test, dmg)
        drop = round(max([auc_ref[k] - auc_cmp[k] for k in KEYS if k in auc_cmp and k in auc_ref] or [0.0]), 4)
        if drop <= 0.01: break
    usable = all(auc_cmp.get(k, 0) >= 0.8 for k in ("shuffle", "garble", "foreign"))
    meta.update({"file": f"{iso}.npz", "train": choice, "n_train": len(fit), "usable": bool(usable), "test_source": "flores_dev" if flores else "heldout_web", "n_test": len(test),
                 "comb": {"mean": float(Hc.mean()), "std": float(Hc.std()), "p05": float(np.percentile(Hc, 5)), "p10": float(np.percentile(Hc, 10))},
                 "auc": auc_cmp, "auc_unpruned": auc_ref, "prune_level": list(lvl), "prune_max_drop": drop,
                 "dev_char": dev, "dev_word": wdev, "seconds": round(time.time() - t0, 1)})
    spell = None
    if flores and iso in G["bible"] and len(G["bible"][iso]) >= 1000:
        sp_meta, sp_z = build_spelling(iso, test)
        meta["spelling" if sp_meta["enabled"] else "spelling_candidate"] = sp_meta
        spell = sp_z
    return iso, (meta, z, spell)


def char_arrays(lm):
    z = {"V": np.array([lm.V], dtype=np.int64)}
    for k in range(1, lm.n + 1):
        z[f"gk{k}"], z[f"gc{k}"] = sorted_arrays([str_hash(g) for g in lm.cnt[k]], list(lm.cnt[k].values()))
        hs = list(lm.ctx[k].keys())
        z[f"ck{k}"], z[f"ct{k}"] = sorted_arrays([str_hash(h) for h in hs], [lm.ctx[k][h] for h in hs])
        _, z[f"cy{k}"] = sorted_arrays([str_hash(h) for h in hs], [lm.typ[k].get(h, 0) for h in hs])
    return z


def build_spelling(iso, flores_test):
    """char-only model trained on BIBLE verses (consistent orthography). Enabled when its diacritic-stripping AUC on FLORES dev human text is >= 0.8."""
    bib = [F_nfc(t) for t in G["bible"][iso]]; random.Random(1).shuffle(bib)
    hold, fit = bib[:300], bib[300:12300]
    lm = prune_char(F.CharLM(5).fit(fit))
    hs = np.array([lm.score(t) for t in hold]); mu, sd = float(hs.mean()), float(hs.std() + 1e-6)
    comp = FL.CompactLanguage(char_arrays(lm), {"order": 5, "char": {"mu": mu, "sd": sd}})
    sz = lambda t: (comp.char_score(t) - mu) / sd
    dev = max(abs(comp.char_score(t) - lm.score(t)) for t in flores_test[:30])
    rng = random.Random(7); strip = [x for x in (F.strip_marks(t) for t in flores_test) if x not in set(flores_test)]
    stripped = [(t, F.strip_marks(t)) for t in flores_test if F.strip_marks(t) != t]
    H = np.array([sz(t) for t in flores_test])
    auc_strip = round(F.auc([sz(t) for t, _ in stripped], [sz(s) for _, s in stripped]), 4) if len(stripped) >= 20 else float("nan")      # paired set: only sentences that change
    chars = sorted(set("".join(fit[:300])))
    garble = ["".join(rng.choice(chars) if rng.random() < 0.15 else c for c in t) for t in flores_test]
    auc_garble = round(F.auc(list(H), [sz(x) for x in garble]), 4)
    hz = np.array([sz(t) for t in hold]); p05_b = float(np.percentile(hz, 5)); p05_f = float(np.percentile(H, 5))
    fpr_at_b = float((H < p05_b).mean())
    floor, src = (p05_b, "heldout_bible_p05") if fpr_at_b <= 0.10 else (p05_f, "flores_p05 (the Bible floor would penalise > 10% of FLORES human sentences)")
    enabled = bool(auc_strip == auc_strip and auc_strip >= 0.8)
    stripped_below = round(float(np.mean([sz(x) < floor for _, x in stripped])), 4) if stripped else float("nan")        # share of diacritic-stripped sentences the penalty would catch
    human_below = round(float((H < floor).mean()), 4)                                                                    # share of correct human sentences it would hit (about 5 % by construction)
    meta = {"enabled": enabled, "file": f"{iso}_spelling.npz", "mu": mu, "sd": sd, "floor": floor, "floor_source": src, "p05_bible_heldout": p05_b, "p05_flores": p05_f,
            "false_penalty_rate_flores_at_bible_floor": round(fpr_at_b, 4), "auc_strip_flores": auc_strip, "auc_garble_flores": auc_garble, "stripped_below_floor": stripped_below, "human_below_floor": human_below, "n_fit": len(fit), "n_strip_pairs": len(stripped), "dev_char": dev}
    return meta, (char_arrays(lm) if enabled else None)


def F_nfc(t):
    import unicodedata
    return unicodedata.normalize("NFC", t)


def main():
    ap = argparse.ArgumentParser()
    for a in ("train-old", "train-human", "flores", "mono-test", "best", "siblings", "out"): ap.add_argument("--" + a, required=True)
    ap.add_argument("--bible", default=None, help="iso<TAB>verse; Bible-trained SPELLING models for languages whose FLORES diacritic-strip AUC is >= 0.8")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    G["old"], G["human"], G["flores"], G["mono_test"] = load(a.train_old), load(a.train_human), load(a.flores), load(a.mono_test)
    G["bible"] = load(a.bible) if a.bible else {}
    G["best"] = {r["iso"]: r["best_training"] for r in csv.DictReader(open(a.best))}
    sib = {i: list(v) for i, v in FIXED_SIBLINGS.items()}
    for r in csv.DictReader(open(a.siblings)):
        n = float(r["n_test"]) or 1
        for tok in (r["top_confusions"] or "").split(" "):
            if ":" in tok:
                j, c = tok.split(":")
                if float(c) / n >= 0.05 and j != r["iso"] and j not in sib.setdefault(r["iso"], []): sib[r["iso"]].append(j)
    G["sib"] = sib
    isos = sorted(set(G["old"]) | set(G["human"]))
    with mp.get_context("fork").Pool(a.procs) as p:
        res = dict(p.imap_unordered(work, isos))
    langs = {}
    for iso, r in sorted(res.items()):
        if r is None or isinstance(r, dict): print("skipped", iso, r); continue
        meta, z, zs = r
        np.savez_compressed(os.path.join(a.out, meta["file"]), **z)
        if zs is not None: np.savez_compressed(os.path.join(a.out, meta["spelling"]["file"]), **zs)
        meta["siblings"] = [s for s in sib.get(iso, []) if s in res and isinstance(res[s], tuple)]
        meta["file_bytes"] = os.path.getsize(os.path.join(a.out, meta["file"])) + (os.path.getsize(os.path.join(a.out, meta["spelling"]["file"])) if "spelling" in meta else 0)
        langs[iso] = {k: v for k, v in meta.items()}
    man = {"version": 1, "order": 5, "created": time.strftime("%F"), "prune_levels": LEVELS, "languages": langs}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    # sibling-ratio validity: positives = own test text, negatives = the sibling's test text, both scored as the claimed language
    sc = FL.FluencyScorer(a.out); out = []
    for iso, m in langs.items():
        for s in m["siblings"][:1]:
            own = [F_nfc(t) for t in (G["flores"].get(iso) or G["mono_test"].get(iso, []))][:300]
            oth = [F_nfc(t) for t in (G["flores"].get(s) or G["mono_test"].get(s, []))][:300]
            if len(own) >= 100 and len(oth) >= 100:
                m["sibling_auc"] = round(F.auc([sc.sibling_ratio(iso, t) for t in own], [sc.sibling_ratio(iso, t) for t in oth]), 4); m["sibling_auc_vs"] = s; out.append((iso, s, m["sibling_auc"]))
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    tot = sum(m["file_bytes"] for m in langs.values())
    print(f"languages {len(langs)} | usable {sum(m['usable'] for m in langs.values())} | total {tot / 1e6:.1f} MB | max prune drop {max(m['prune_max_drop'] for m in langs.values()):.4f} | prune levels {dict(collections.Counter(tuple(m['prune_level']) for m in langs.values()))}"
          f" | max compact-vs-pruned dev char {max(m['dev_char'] for m in langs.values()):.2e} word {max(m['dev_word'] for m in langs.values()):.2e}")
    print("sibling-ratio AUC:", out)
    print("spelling models:", {i: (m["spelling"]["auc_strip_flores"], round(m["spelling"]["floor"], 2)) for i, m in langs.items() if "spelling" in m},
          "| candidates not enabled:", {i: m["spelling_candidate"]["auc_strip_flores"] for i, m in langs.items() if "spelling_candidate" in m})


if __name__ == "__main__":
    main()
