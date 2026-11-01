#!/usr/bin/env python3
"""All data tables of the paper as pandoc pipe tables in paper/tables/*.md (included by paper.md with {{table:NAME}}).
Run:  python paper/scripts/make_tables.py        (re-run after refreshing paper/data/final2_metrics.jsonl)"""
import json
import os
import re
import statistics as st
from collections import OrderedDict, defaultdict

from common import *  # noqa: F401,F403


def w(name, header, rows, align=None, note=None):
    align = align or (["l"] + ["r"] * (len(header) - 1))
    sep = "|" + "|".join({"l": ":--", "r": "--:", "c": ":-:"}[a] for a in align) + "|"
    out = ["| " + " | ".join(header) + " |", sep] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("table", name, len(rows))


def f1(x, d=1):
    return f"{x:.{d}f}"


def sgn(x, d=2):
    return f"{x:+.{d}f}"


WIDE = read_csv(os.path.join(RDATA, "final_model_per_language_wide.csv"))
BOOT = json.load(open(os.path.join(RDATA, "final_model_runs", "macro_boot.json")))
BNAME = OrderedDict([("flores_plus", "FLORES+ devtest"), ("ntrex", "NTREX-128"), ("mafand", "MAFAND-MT test"), ("smol_sent", "SmolSent"),
                     ("ibom_mt", "IbomMT"), ("flores_moz", "FLORES-MOZ"), ("salt", "SALT"), ("glot_story", "GlotStoryBook")])


def macro(bench, subset, pred=lambda r: True):
    rs = [r for r in WIDE if r["benchmark"] == bench and r["subset"] == subset and pred(r)]
    if not rs:
        return None
    return dict(n=len(rs), sft=st.mean(float(r["sft_chrf"]) for r in rs), rl=st.mean(float(r["rl_chrf"]) for r in rs),
                sft_sp=st.mean(float(r["sft_spbleu"]) for r in rs), rl_sp=st.mean(float(r["rl_spbleu"]) for r in rs),
                sft_fin=100 * st.mean(float(r["sft_finished"]) for r in rs), rl_fin=100 * st.mean(float(r["rl_finished"]) for r in rs),
                rows=int(float(rs[0]["n"])))


def t_main():
    rows = []
    for b, nm in BNAME.items():
        for sub in ("all", "no_pair", "clean"):
            m = macro(b, sub)
            if not m:
                continue
            bt = BOOT.get(f"{b}|{sub}|all")
            ci = lambda k: f"{sgn(bt[k]['d'])} [{sgn(bt[k]['lo'])}, {sgn(bt[k]['hi'])}]" if bt else "n/a"
            rows.append([(nm + ("$^\\dagger$" if b == "ibom_mt" else "")) if sub == "all" else "", sub.replace("_", "\\_"), m["n"], f1(m["sft"]), f1(m["rl"]), ci("chrf"), f1(m["sft_sp"]), f1(m["rl_sp"]), ci("spbleu"),
                         f"{m['sft_fin']:.1f} / {m['rl_fin']:.1f}"])
    w("t_main", ["benchmark", "subset", "langs", "chrF++ SFT", "chrF++ RL", "$\\Delta$ chrF++ [95% CI]", "spBLEU SFT", "spBLEU RL", "$\\Delta$ spBLEU [95% CI]", "finished % SFT / RL"], rows,
      ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r"])


def t_groups():
    rows = []
    for key, nm in [("all", "all FLORES+ languages"), ("qe_validated", "QE-validated"), ("blend", "QE-blend"), ("consensus_only", "consensus-only"), ("sft_only", "SFT-only (no RL)"),
                    ("rl77", "RL languages (all groups)")] + [(f"fam:{f}", f) for f in ["Bantu", "Atlantic-Congo (non-Bantu)", "Mande", "Afroasiatic", "Nilo-Saharan", "Creole / Pidgin", "Other (Indo-European, Austronesian)"]]:
        if key.startswith("fam:"):
            pred = lambda r, f=key[4:]: r["family"] == f
        elif key == "all":
            pred = lambda r: True
        elif key == "rl77":
            pred = lambda r: r["group"] != "sft_only"
        else:
            pred = lambda r, g=key: r["group"] == g
        m = macro("flores_plus", "no_pair", pred)
        bt = BOOT.get(f"flores_plus|no_pair|{key}")
        if not m or not bt:
            continue
        rows.append([nm, m["n"], f1(m["sft"]), f1(m["rl"]), f"{sgn(bt['chrf']['d'])} [{sgn(bt['chrf']['lo'])}, {sgn(bt['chrf']['hi'])}]",
                     f"{sgn(bt['spbleu']['d'])} [{sgn(bt['spbleu']['lo'])}, {sgn(bt['spbleu']['hi'])}]", f"{m['sft_fin']:.1f} / {m['rl_fin']:.1f}"])
    w("t_groups", ["subset (FLORES+ devtest, no\\_pair)", "langs", "chrF++ SFT", "chrF++ RL", "$\\Delta$ chrF++ [95% CI]", "$\\Delta$ spBLEU [95% CI]", "finished % SFT / RL"], rows)


def published(model, bench_prefix="FLORES-200 "):
    out = {}
    for r in read_csv(os.path.join(RDATA, "published_baselines.csv")):
        if r["model"] == model and r["direction"] == "en->xx" and r["benchmark"].startswith(bench_prefix):
            v = fnum(r["chrF++"])
            if v is not None:
                out[r["iso"]] = v
    return out


def t_published():
    systems = ["NLLB-200-1.3B", "NLLB-200-3.3B", "AfriqueLlama-8B", "AfriqueGemma-12B", "AfriqueQwen-14B"]
    P = {s: published(s) for s in systems}
    rows = []
    for sub in ("all", "no_pair"):
        ours = {r["iso"]: float(r["rl_chrf"]) for r in WIDE if r["benchmark"] == "flores_plus" and r["subset"] == sub}
        for s in systems:
            shared = [i for i in ours if i in P[s]]
            mo = st.mean(ours[i] for i in shared); ms = st.mean(P[s][i] for i in shared)
            better = sum(ours[i] > P[s][i] for i in shared)
            rows.append([sub.replace("_", "\\_") if s == systems[0] else "", s, len(shared), f1(mo), f1(ms), sgn(mo - ms, 1), f"{better}/{len(shared)}"])
    w("t_published", ["our subset", "published system (en$\\to$X, FLORES-200 devtest)", "shared langs", "ours (macro)", "system (macro)", "difference", "ours better on"], rows,
      ["l", "l", "r", "r", "r", "r", "r"])
    # the 21 languages covered by all five systems
    common = sorted(set.intersection(*[set(P[s]) for s in systems]) & {r["iso"] for r in WIDE if r["benchmark"] == "flores_plus"})
    rows = []
    for sub in ("all", "no_pair"):
        ours = {r["iso"]: float(r["rl_chrf"]) for r in WIDE if r["benchmark"] == "flores_plus" and r["subset"] == sub}
        rows.append([sub.replace("_", "\\_"), len(common), f1(st.mean(ours[i] for i in common))] + [f1(st.mean(P[s][i] for i in common)) for s in systems])
    w("t_published21", ["our subset", "langs", "ours"] + systems, rows, ["l"] + ["r"] * 7)
    # per-language table, ours (all) vs systems
    ours = {r["iso"]: (r["language"], float(r["rl_chrf"])) for r in WIDE if r["benchmark"] == "flores_plus" and r["subset"] == "all"}
    rows = []
    for i in sorted(ours, key=lambda k: -ours[k][1]):
        vals = [P[s].get(i) for s in systems]
        if all(v is None for v in vals):
            continue
        rows.append([f"{ours[i][0]} ({i})", f1(ours[i][1])] + [("%.1f" % v) if v is not None else "--" for v in vals])
    w("t_published_lang", ["language", "ours (all rows)"] + systems, rows, ["l"] + ["r"] * 6)


def t_phases():
    d = json.load(open(os.path.join(REPORTS, "rl_vs_sft_weak_signal_2026-10-21", "results.json")))
    ck = d["checkpoints"]
    nm = dict(sft="SFT-final", rl1="RL phase 1", b64="2b (step 488)", c256="2c b256", c512="2c b512", c1024="2c b1024", c2048="2c b2048", c2c="2c final (1,433)")
    rows = []
    for g, gn in [("control", "control: 10 QE-validated languages"), ("blend", "blend: 19 languages"), ("consensus", "consensus-only: 13 languages")]:
        m = d["groups"][g]["all"]["mean"]
        rows.append([gn] + [f1(m[c]) for c in ck])
    w("t_phases", ["group (mean chrF++, all rows)"] + [nm[c] for c in ck], rows)
    rows = []
    for g, gn in [("control", "control (10)"), ("blend", "blend (19)"), ("consensus", "consensus-only (13)")]:
        dv = d["groups"][g]["all"]["delta_vs_first"]
        rows.append([gn] + [f"{sgn(dv[c]['mean'], 1)} [{sgn(dv[c]['ci'][0], 1)}, {sgn(dv[c]['ci'][1], 1)}]" for c in ck[1:]])
    w("t_phases_delta", ["group ($\\Delta$ vs SFT)"] + [nm[c] for c in ck[1:]], rows)


def t_longdoc():
    core, human = None, None
    T = md_tables(os.path.join(REPORTS, "longdoc_eval_final_model_2026-10-21.md"))
    core, human = {}, {}
    for t in T:
        h = t["heading"]
        if h.startswith("core bin b") and "macro over 8 languages" in h:
            core[h.split()[2]] = {r[0]: r for r in t["rows"]}
        if h.startswith("human bin") and "macro over" in h:
            human[h.split()[2]] = {r[0]: r for r in t["rows"]}
    cfgs = [("greedy", "final, whole document, greedy"), ("freq005", "final, whole doc., freq. penalty 0.05"), ("t07", "final, whole doc., T 0.7"), ("bon4", "final, whole doc., best-of-4 (T 0.7)"),
            ("greedy_win", "final, 256-token windows"), ("greedy_sent", "final, sentence by sentence"), ("sentfix", "final, sentence by sentence + repair (sentfix)"),
            ("sft:greedy", "SFT, whole document, greedy"), ("sft:greedy_sent", "SFT, sentence by sentence")]
    rows = []
    for c, lab in cfgs:
        rows.append([lab] + [core[b][c][1].split("[")[0].strip() for b in BINS])
    w("t_longdoc_chrf", ["configuration (chrF++ vs NLLB-3.3B pseudo-reference, macro of 8 languages)"] + BINS, rows)
    rows = []
    for c, lab in cfgs:
        rows.append([lab] + [f"{core[b][c][4]} / {core[b][c][5]}" for b in BINS])
    w("t_longdoc_fin", ["configuration (finished / loop, % of documents)"] + BINS, rows)
    hb = ["sm", "st512", "st1024", "st2048"]
    rows = []
    for c, lab in [("greedy", "final, whole document, greedy"), ("t07", "final, whole doc., T 0.7"), ("bon4", "final, whole doc., best-of-4"), ("greedy_win", "final, 256-token windows"),
                   ("sentfix", "final, sentence by sentence + repair"), ("sft:greedy", "SFT, whole document, greedy"), ("sft:greedy_sent", "SFT, sentence by sentence")]:
        rows.append([lab] + [human[b][c][1] for b in hb])
    w("t_longdoc_human", ["configuration (chrF++ [95% CI] vs human SmolDoc references)", "short docs (6 lang.)", "~512 tok. (6)", "~1,024 tok. (6)", "~2,048 tok. (4)"], rows)
    # reward and ok rate
    rw = [t for t in T if t["header"][:1] == ["system"] and t["header"][-1] == "all bins"]
    rows = []
    for r in rw[0]["rows"]:
        if r[0] in ("rl2c:greedy", "rl2c:sentfix", "rl2c:greedy_win", "rl2c:t10", "sft:greedy", "sft:greedy_sent"):
            rows.append([r[0].replace("rl2c:", "final: ").replace("sft:", "SFT: ")] + r[1:])
    for r in rw[1]["rows"]:
        rows.append([r[0].replace("yard_", "yardstick: ")] + r[1:])
    w("t_longdoc_reward", ["system (mean total RL reward, B8)"] + BINS + ["all"], rows)


def t_decoding():
    T = md_tables(os.path.join(REPORTS, "final_model_decoding_study_2026-10-21.md"))
    t = [x for x in T if x["header"][:2] == ["config", "chrF++"]][0]
    rows = []
    for r in t["rows"]:
        if "cross-check" in r[0]:
            continue
        rows.append([r[0].replace("_", "\\_"), r[1], r[2], r[3], r[5], r[7], r[8], r[12]])
    w("t_decoding", ["configuration", "chrF++", "$\\Delta$ chrF++ [95% CI]", "spBLEU", "AfriCOMET (18 lang.)", "loop %", "LID %", "cost vs same-engine greedy"], rows,
      ["l", "r", "r", "r", "r", "r", "r", "r"])


def t_sft_stage():
    T = md_tables(os.path.join(REPORTS, "final_run_integration_2026-10-21.md"))
    t1 = [t for t in T if t["header"][:2] == ["model", "sent. chrF++"] and "weak-dev blend (19)" in t["header"]][0]
    rows = [[r[0], r[1], r[2], r[3], r[4], r[5].replace("**", ""), r[6], r[7]] for r in t1["rows"]]
    w("t_sft_stage_v1", ["model (mix\\_v1)", "sent. chrF++", "paper chrF", "paper finished", "general dev $\\Delta$", "weak-blend $\\Delta$ [95% CI]", "weak-consensus $\\Delta$", "control $\\Delta$"], rows)
    t2 = [t for t in T if t["header"][:2] == ["model", "sent. chrF++"] and "guardrails" in t["header"]][0]
    rows = [[r[0].replace("**", "")] + [c.replace("**", "") for c in r[1:]] for r in t2["rows"]]
    w("t_sft_stage_v2", ["mixture (alpha 0.3 / 0.5 / 0.7 / 1.0)", "sent. chrF++", "paper chrF", "paper finished %", "general $\\Delta$", "weak-blend $\\Delta$", "weak-consensus $\\Delta$", "guardrails"], rows)
    T3 = md_tables(os.path.join(REPORTS, "final_run2_integration_2026-10-22.md"))
    t3 = T3[0]
    rows = [[c.replace("**", "") for c in r] for r in t3["rows"]]
    w("t_sft_stage_c3", t3["header"], rows)


def t_gemma31b():
    T = md_tables(os.path.join(REPORTS, "gemma31b_weak_language_probe_2026-10-22.md"))
    t = [x for x in T if x["header"][:3] == ["lang", "name", "n"]][0]
    rows = [[c.replace("**", "") for c in r] for r in t["rows"]]
    w("t_gemma31b", ["iso", "language", "n", "our model", "Gemma-4-31B few-shot-8", "$\\Delta$", "95% CI", "verdict"], rows, ["l", "l", "r", "r", "r", "r", "r", "l"])


def t_nllb():
    T = md_tables(os.path.join(REPORTS, "nllb600m_teacher_2026-10-22.md"))
    t = [x for x in T if x["header"][:3] == ["benchmark (English -> X)", "langs", "ours chrF++"]][0]
    rows = [[c.replace("`", "") for c in r] for r in t["rows"]]
    w("t_nllb", ["benchmark (en$\\to$X)", "langs", "ours chrF++", "NLLB-3.3B", "$\\Delta$ [95% CI]", "ours spBLEU", "NLLB-3.3B", "$\\Delta$ [95% CI]"], rows)


def t_anchors():
    rows = [
        ["entity (numbers), 3 weak languages", "100% (other source's translation); 59% (second half dropped)", "0.7% mean, 2.0% max", "anchors\\_2026-10-21.md l.55-67, l.45-53"],
        ["repetition", "100% (clause x6, whole translation twice)", "1.8% mean, 6.7% max (sentences)", "same"],
        ["boilerplate buffer", "100% (stock sentence appended)", "0.4% mean, 3.0% max", "same"],
        ["substring copy (v2)", "100% exact / appended / half copies; 78% of a copied 25% tail; 96% of a 50% tail", "0.5% mean, 1.3% max (sentences), 0.0% (documents)", "anchors\\_v2\\_2026-10-22.md sec. 3"],
        ["stop-word profile (v2)", "AUC 0.99 vs English copy, 0.97 vs stuffed bags (sentences)", "1.2% sentences, 0.6% documents", "anchors\\_v2 sec. 4"],
        ["padding, v2 (length bug)", "bag/repeat attacks caught", "5 / 14 / 31 / 81 / 100% of human documents of <100 / 100-200 / 200-400 / 400-800 / >800 words", "anchors\\_v3 sec. 1"],
        ["padding, v3 (segmental type/token)", "--", "0.0% median out-of-domain FP (max 76.4%, sid)", "anchors\\_v3 sec. 3"],
    ]
    w("t_anchors", ["detector", "detection of corrupted human references", "false positives on held-out human text", "source"], rows, ["l", "l", "l", "l"])


def t_final2():
    rows = read_jsonl(os.path.join(DATA, "final2_metrics.jsonl"))
    ev = [r for r in rows if "bench/chrf_macro" in r]
    aux = {r["step"]: r for r in rows if "bench_aux/afrisci_dev_doc/chrf_macro" in r}
    out = []
    for r in ev:
        a = aux.get(r["step"], {})
        out.append([int(r["step"]), f1(r["bench/chrf_macro"], 2), f1(a.get("bench_aux/afrisci_dev_doc/chrf_macro", float("nan")), 2), f"{a.get('bench_aux/afrisci_dev_doc/finished_rate', float('nan')):.3f}",
                    f1(a.get("bench_aux/general_dev/chrf_macro", float("nan")), 2), f1(a.get("bench_aux/weak_dev/chrf_macro", float("nan")), 2), (f"{r['eval/reward']:.3f}" if 'eval/reward' in r else "--")])
    w("t_final2", ["step", "AfriScience dev sentence chrF++", "paper-level chrF++", "paper-level finished", "general dev (19 lang.)", "weak dev", "eval reward"], out)
    return rows


def t_master():
    prim = {}
    for r in WIDE:
        if r["subset"] == "no_pair" and r["primary"] == "1":
            prim[r["iso"]] = r
    langs = OrderedDict()
    for r in read_csv(os.path.join(RDATA, "final_model_per_language.csv")):
        langs.setdefault(r["iso"], (r["language"], r["group"], r["family"]))
    rows = []
    for g in ["qe_validated", "blend", "consensus_only", "sft_only"]:
        items = [(i, langs[i]) for i in langs if langs[i][1] == g]
        items.sort(key=lambda kv: -float(prim[kv[0]]["rl_chrf"]) if kv[0] in prim else 1e9)
        for i, (nm, grp, fam) in items:
            p = prim.get(i)
            if p:
                star = "*" if float(p["d_chrf_lo"]) > 0 or float(p["d_chrf_hi"]) < 0 else ""
                rows.append([f"`{i}`", nm, GROUP_LABEL[grp], BNAME.get(p["benchmark"], p["benchmark"]), int(float(p["n"])), f1(float(p["sft_chrf"])), f1(float(p["rl_chrf"])),
                             f"{sgn(float(p['d_chrf']), 1)} [{sgn(float(p['d_chrf_lo']), 1)}, {sgn(float(p['d_chrf_hi']), 1)}]{star}", f1(float(p["sft_spbleu"])), f1(float(p["rl_spbleu"])),
                             f"{100*float(p['sft_finished']):.0f} / {100*float(p['rl_finished']):.0f}"])
            else:
                rows.append([f"`{i}`", nm, GROUP_LABEL[grp], "none", "--", "--", "--", "--", "--", "--", "--"])
    w("t_master", ["iso", "language", "group", "primary benchmark", "n", "chrF++ SFT", "chrF++ RL", "$\\Delta$ [95% CI]", "spBLEU SFT", "spBLEU RL", "finished % SFT / RL"], rows,
      ["l", "l", "l", "l", "r", "r", "r", "r", "r", "r", "r"])


def t_groups_def():
    # counts: reports/final_model_per_language_2026-10-21.md l.19 (qe_validated 42, blend 12, consensus_only 23, sft_only 8; 77 RL = first three)
    cnt = {"qe_validated": 42, "blend": 12, "consensus_only": 23, "sft_only": 8}
    desc = {"qe_validated": ("AfriCOMET-QE validated against human references (AUC human vs mismatched >= 0.70)", "QE (ensemble) + consensus where weak"),
            "blend": ("named on the AfriCOMET card or a variety of a validated language, no reference-based validation", "0.5 x QE + 0.5 x consensus"),
            "consensus_only": ("not covered by AfriCOMET", "consensus chrF++ in the group + constraint penalties"),
            "sft_only": ("SFT targets that were not RL-trained (some without any SFT data)", "none")}
    rows = [[GROUP_LABEL[g], cnt[g], desc[g][0], desc[g][1]] for g in ["qe_validated", "blend", "consensus_only", "sft_only"]]
    rows.append(["total", sum(cnt.values()), "85 language-table rows, of which 77 are RL-trained (first three groups)", ""])
    w("t_groups_def", ["group", "languages", "definition", "content reward"], rows, ["l", "r", "l", "l"])
    T = md_tables(os.path.join(REPORTS, "final_model_per_language_2026-10-21.md"))
    t = [x for x in T if x["header"][:3] == ["iso", "language", "group"] and len(x["header"]) == 5][0]
    rows = [[r[0], r[1], r[2], r[3]] for r in t["rows"]]
    w("t_nobench", ["iso", "language", "group", "RL-trained"], rows, ["l", "l", "l", "l"])


def t_hparams():
    c2 = json.load(open(os.path.join(DATA, "rl2c_config.json"))); c2 = c2.get("config", c2)
    f2 = json.load(open(os.path.join(DATA, "final2_config.json"))); f2 = f2.get("config", f2)
    keys = ["steps", "prompts_per_step", "group_size", "greedy_in_group", "greedy_stats", "greedy_adv_scale", "temperature", "top_p", "top_k", "learning_rate", "warmup_steps", "weight_decay", "beta1", "beta2", "grad_clip",
            "advantage_norm", "std_eps", "kl_coef", "kl_estimator", "kl_guard_threshold", "kl_guard_window", "kl_guard_factor", "kl_guard_min_scale", "num_updates_per_batch", "loss_agg", "max_new_tokens", "max_src_tokens",
            "language_alpha", "language_floor_steps", "language_floor_prompts", "select_by", "select_min_gain", "select_min_finished_rate", "seed"]
    rows = [[f"`{k}`", str(c2.get(k, "--")), str(f2.get(k, "--"))] for k in keys]
    rows.append(["`curriculum`", "b256 391, b512 446, b1024 149, b2048 298, mixed 149 steps", "mixed from step 0 (70% science / 30% general)"])
    rows.append(["`max_new_tokens_by_bin`", json.dumps(c2.get("max_new_tokens_by_bin")), json.dumps(f2.get("max_new_tokens_by_bin"))])
    rows.append(["`prompts_per_step_by_bin`", json.dumps(c2.get("prompts_per_step_by_bin")), json.dumps(f2.get("prompts_per_step_by_bin"))])
    rows.append(["`start model`", "2b step-488 (b64-stage) model", "science SFT stage A2W, alpha 0.7"])
    w("t_hparams", ["key", "phase 2c", "final2"], rows, ["l", "l", "l"])


def rl_completions():
    """Estimated sampled completions and generated tokens of every RL phase of the lineage final_1433 (from logged steps, curricula and
    prompts per step of the yaml files in gemma_mt/configs; phase 2 log starts at step 51, steps 1-50 are not in the copy we have)."""
    def phase(csvf, cur, ppb, G, maxstep=None):
        c = t = 0
        for r in read_csv(os.path.join(DATA, csvf)):
            s_ = int(r["step"])
            if maxstep and s_ > maxstep:
                continue
            acc, b = 0, "mixed"
            for name, n in cur:
                if s_ <= acc + n:
                    b = name; break
                acc += n
            k = ppb[b] * G
            c += k; t += k * float(r["completion_len"])
        return c, t
    P1 = {"b64": 64, "b256": 48, "b512": 42, "b1024": 42, "mixed": 42}
    P2 = {"b64": 128, "b256": 96, "b512": 84, "b1024": 84, "mixed": 84}
    out = OrderedDict()
    out["phase 1 (300 steps, G=8)"] = phase("rl1_steps.csv", [("b64", 40), ("b256", 60), ("b512", 60), ("b1024", 60)], P1, 8)
    out["phase 2 (logged steps 51-378, G=8)"] = phase("rl2_steps.csv", [("b64", 98), ("b256", 130), ("b512", 75), ("b1024", 45)], P2, 8)
    out["phase 2b, b64 stage only (steps 1-488, G=16)"] = phase("rl2b_steps.csv", [("b64", 488)], P1, 16, maxstep=488)
    rows = read_jsonl(os.path.join(DATA, "rl2c_metrics.jsonl"))
    cur = [("b256", 391), ("b512", 446), ("b1024", 149), ("b2048", 298)]
    Pc = {"b256": 48, "b512": 42, "b1024": 42, "b2048": 21, "mixed": 42}
    seen, c, t = set(), 0, 0
    for r in rows:
        if "kl" not in r or r["step"] in seen:
            continue
        seen.add(r["step"])
        acc, b = 0, "mixed"
        for name, n in cur:
            if r["step"] <= acc + n:
                b = name; break
            acc += n
        k = Pc[b] * 16
        c += k; t += k * r["completion_len"]
    out["phase 2c (1,433 steps, G=16)"] = (c, t)
    return out


def flores_macro_old(system, subset, isos=None):
    rows = [r for r in read_csv(os.path.join(REPORTS, "gemma_sft_benchmarks_2026-10-18", "full_flores", "results.csv"))
            if r["system"] == system and r["benchmark"] == "flores_plus" and r["subset"] == subset]
    return {r["iso"]: float(r["chrf"]) for r in rows if (isos is None or r["iso"] in isos)}


def flores_macro_320(system, subset):
    f = "bench_metrics_sft320.csv" if system.startswith("sft") else "bench_metrics_rl320.csv"
    rows = [r for r in read_csv(os.path.join(RDATA, "final_model_runs", f)) if r["benchmark"] == "flores_plus" and r["subset"] == subset]
    return {r["iso"]: float(r["chrf"]) for r in rows}


def t_rq():
    comp = rl_completions()
    isos = None
    sets = {}
    for sub in ("all", "clean"):
        base = flores_macro_old("base", sub); s20 = flores_macro_old("sft", sub)
        s27 = flores_macro_320("sft320", sub); rl = flores_macro_320("rl320", sub)
        common = set(base) & set(s20) & set(s27) & set(rl)
        sets[sub] = {k: st.mean(v[i] for i in common) for k, v in dict(base=base, s20=s20, s27=s27, rl=rl).items()}
        sets[sub]["n"] = len(common)
    tot_c = sum(v[0] for v in comp.values()); tot_t = sum(v[1] for v in comp.values())
    SFT_TOK = 27638 * 131072 / 1e9; SFT20 = 20500 * 131072 / 1e9; LATE = SFT_TOK - SFT20
    a, c = sets["all"], sets["clean"]
    rows = [
        ["SFT, base model to step 20,500", f"{SFT20:.2f} B training tokens (teacher forcing)", f1(a["s20"] - a["base"], 2), f1(c["s20"] - c["base"], 2), f1((a["s20"] - a["base"]) / SFT20, 2), f1((c["s20"] - c["base"]) / SFT20, 2)],
        ["SFT, step 20,500 to 27,638 (last 26%)", f"{LATE:.2f} B training tokens", f1(a["s27"] - a["s20"], 2), f1(c["s27"] - c["s20"], 2), f1((a["s27"] - a["s20"]) / LATE, 2), f1((c["s27"] - c["s20"]) / LATE, 2)],
        ["RL, whole lineage to step 1,433", f"{tot_c/1e6:.2f} M sampled completions, {tot_t/1e9:.2f} B generated tokens", f1(a["rl"] - a["s27"], 2), f1(c["rl"] - c["s27"], 2), f1((a["rl"] - a["s27"]) / (tot_t / 1e9), 2), f1((c["rl"] - c["s27"]) / (tot_t / 1e9), 2)],
    ]
    w("t_rq", ["step", "data / compute used", "$\Delta$ chrF++ (all rows)", "$\Delta$ chrF++ (clean rows)", "per billion tokens (all)", "per billion tokens (clean)"], rows, ["l", "l", "r", "r", "r", "r"])
    rows = [[k, f"{v[0]:,}", f"{v[1]/1e9:.3f}"] for k, v in comp.items()] + [["total to final_1433", f"{tot_c:,}", f"{tot_t/1e9:.3f}"]]
    w("t_rl_budget", ["RL phase", "sampled completions (estimate)", "generated tokens, billions (estimate)"], rows, ["l", "r", "r"])
    json.dump(dict(sets=sets, completions={k: list(v) for k, v in comp.items()}, total_completions=tot_c, total_tokens=tot_t, sft_tokens_b=SFT_TOK, late_b=LATE),
              open(os.path.join(DATA, "rq_budget.json"), "w"), indent=1)


def t_geez():
    rows = []
    for iso, nm in (("amh", "Amharic"), ("tir", "Tigrinya")):
        for b in ("flores_plus", "ntrex", "mafand", "smol_sent"):
            r = [x for x in WIDE if x["iso"] == iso and x["benchmark"] == b and x["subset"] == "no_pair"]
            if not r:
                continue
            r = r[0]
            rows.append([nm, BNAME[b], int(float(r["n"])), f1(float(r["sft_chrf"])), f1(float(r["rl_chrf"])), f"{sgn(float(r['d_chrf']), 1)} [{sgn(float(r['d_chrf_lo']), 1)}, {sgn(float(r['d_chrf_hi']), 1)}]",
                         f"{100*float(r['sft_finished']):.1f} / {100*float(r['rl_finished']):.1f}"])
    w("t_geez", ["language", "benchmark (`no\\_pair`)", "n", "chrF++ SFT", "chrF++ RL", "$\\Delta$ [95% CI]", "finished % SFT / RL"], rows, ["l", "l", "r", "r", "r", "r", "r"])


def t_tokcost():
    """Tokens needed for the same FLORES+ devtest text, relative to the English text (total tokens, not per word)."""
    d = json.load(open(os.path.join(REPO, "docs", "report", "tokenizer_comparison_flores_plus.json")))
    toks = [("gemma3", "Gemma 3 (ours)"), ("qwen3", "Qwen3"), ("gpt-oss", "gpt-oss"), ("old-52k", "earlier 52k vocabulary"), ("sabiyarn-0.5B", "SabiYarn-0.5B (65k, African-trained)")]
    rows = []
    for iso, nm in (("amh", "Amharic"), ("tir", "Tigrinya"), ("yor", "Yoruba"), ("zul", "Zulu")):
        r = [nm]
        for k, _ in toks:
            tot = d[iso][k] * d[iso]["words"]; eng = d["eng"][k] * d["eng"]["words"]
            r.append(f"{tot / eng:.2f}")
        rows.append(r)
    w("t_tokcost", ["language"] + [n for _, n in toks], rows)


def t_rq_units():
    """RL versus SFT per unit of data. Dev-set rows come from the logged evaluations (paper/data/*.csv, rl2c_metrics.jsonl)."""
    b = json.load(open(os.path.join(DATA, "rq_budget.json")))
    a, c = b["sets"]["all"], b["sets"]["clean"]
    tot_c, tot_t = b["total_completions"] / 1e6, b["total_tokens"] / 1e6
    sft_all, sft_late = b["sft_tokens_b"] * 1e3, b["late_b"] * 1e3
    sft20 = sft_all - sft_late
    ev = [r for r in read_jsonl(os.path.join(DATA, "rl2c_metrics.jsonl")) if "bench/chrf_macro" in r]
    d0, d1 = ev[0]["bench/chrf_macro"], ev[-1]["bench/chrf_macro"]
    comp = b["completions"]
    first = sum(v[0] for k, v in comp.items() if not k.startswith("phase 2c")) / 1e6
    first_tok = sum(v[1] for k, v in comp.items() if not k.startswith("phase 2c")) / 1e6
    c2c, c2c_tok = comp["phase 2c (1,433 steps, G=16)"][0] / 1e6, comp["phase 2c (1,433 steps, G=16)"][1] / 1e6
    sft_dev = 39.65  # reports/final_model_runs: dev macro of SFT-final (see Section 6.3)
    rows = [
        ["SFT, base model to step 20,500", f"{sft20:,.0f} M training tokens", f"{a['s20'] - a['base']:+.2f}", f"{1000 * (a['s20'] - a['base']) / sft20:.2f}", "n/a"],
        ["SFT, step 20,500 to 27,638", f"{sft_late:,.0f} M training tokens", f"{a['s27'] - a['s20']:+.2f}", f"{1000 * (a['s27'] - a['s20']) / sft_late:.2f}", "n/a"],
        ["RL, all phases to step 1,433", f"{tot_t:,.0f} M generated tokens; {tot_c:.2f} M completions", f"{a['rl'] - a['s27']:+.2f}", f"{1000 * (a['rl'] - a['s27']) / tot_t:.2f}", f"{(a['rl'] - a['s27']) / tot_c:.2f}"],
        ["RL, all phases, clean rows", "same", f"{c['rl'] - c['s27']:+.2f}", f"{1000 * (c['rl'] - c['s27']) / tot_t:.2f}", f"{(c['rl'] - c['s27']) / tot_c:.2f}"],
        ["RL, phases 1, 2, 2b (dev set, 760 sent.)", f"{first_tok:,.0f} M generated tokens; {first:.2f} M completions", f"{d0 - sft_dev:+.2f}", f"{1000 * (d0 - sft_dev) / first_tok:.2f}", f"{(d0 - sft_dev) / first:.2f}"],
        ["RL, phase 2c (dev set, 760 sent.)", f"{c2c_tok:,.0f} M generated tokens; {c2c:.2f} M completions", f"{d1 - d0:+.2f}", f"{1000 * (d1 - d0) / c2c_tok:.2f}", f"{(d1 - d0) / c2c:.2f}"],
    ]
    w("t_rq_units", ["stage", "data used", "$\\Delta$ macro chrF++", "per billion tokens", "per million completions"], rows, ["l", "l", "r", "r", "r"])


def t_geez_data():
    """RL versus extra parallel data for Amharic and Tigrinya (numbers from reports/final_run2_integration_2026-10-22.md section 1,
    reports/final_run_integration_2026-10-21.md section 4 and reports/weak_language_data_2026-10-22.md section 4; see CLAIMS_LEDGER)."""
    def d(iso, bench):
        r = [x for x in WIDE if x["iso"] == iso and x["benchmark"] == bench and x["subset"] == "no_pair"][0]
        return float(r["d_chrf"])
    rows = [
        ["Amharic", f"{d('amh', 'flores_plus'):+.1f} (FLORES+), {d('amh', 'ntrex'):+.1f} (NTREX), {d('amh', 'mafand'):+.1f} (MAFAND), {d('amh', 'smol_sent'):+.1f} (SmolSent)", "+0.02 (weak dev, 60 sent.)", "1.5 M + 0.37 M SA + 0.14 M code-switch (B_v2); 4.47 M NLLB teacher tokens over 27 languages (C3)", "paper-level quality fell with B_v2_ibom (20.5 to 18.8 / 17.6)"],
        ["Tigrinya", f"{d('tir', 'flores_plus'):+.1f} (FLORES+), {d('tir', 'ntrex'):+.1f} (NTREX), {d('tir', 'smol_sent'):+.1f} (SmolSent)", "+0.55 (weak dev, 60 sent.)", "2.0 M (B_v2), plus C3 teacher data", "B_v2_ibom against RL model: +0.70 / +0.74 at $\\alpha$ 0.5 / 0.7"],
    ]
    w("t_geez_data", ["language", "RL on top of SFT: $\\Delta$ chrF++, `no\\_pair` rows", "extra SFT data: C3 against A2W", "parallel tokens added to the SFT stage", "other"], rows, ["l", "l", "l", "l", "l"])


def _ab2():
    """AB2 result files: the compact ablation_results.json (point estimates, dev curves) and the full ab2_results.json (intervals, confirmation sets)."""
    ab = json.load(open(os.path.join(DATA, "ablation_results.json")))["arms"]
    full = json.load(open(os.path.join(DATA, "ab2_results", "ab2_results.json")))["results"]["units"]
    return ab, full


def _ab2ci(v, nd=2, sign=False):
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return f.format(v[0]) + " [" + f.format(v[1]) + ", " + f.format(v[2]) + "]"


def t_ablation():
    """Controlled ablation, design AB2 (result: Section 8.4). Reads paper/data/ablation_results.json (AB2's results/ablation_results.json; schema: {"arms": {"S","W","W_S1","W_S1P"}} with, per arm,
    start_flores, end_flores, gain_flores (language macro chrF++ of FLORES+ devtest `no_pair` over yor kin orm lin), finished ("a / b"), completions, gain_per_m, curve) and the 95% paired-bootstrap
    intervals of AB2's ab2_results.json (results/units/FLORES+ no_pair/all). S = arm F (RL from SFT-final), W = RL from the weak SFT, W_S1 / W_S1P = continued-SFT controls from the weak SFT."""
    ab, full = _ab2()
    p = full["FLORES+ no_pair"]["all"]
    arms_cols = [("S", "F_RL", "arm F: RL from SFT-final"), ("W", "W_RL", "arm W: RL from weak SFT"), ("W_S1", "W_S1", "W + continued SFT, equal GPU time (S1)"), ("W_S1P", "W_S1P", "W + continued SFT, equal number of sentences (S1P)")]
    cols = ["quantity (macro of yor, kin, orm, lin; FLORES+ devtest `no\\_pair`; 95% interval)"] + [c for _, _, c in arms_cols]
    ck = {"S": "gain_F = F_RL - F0", "W": "gain_W = W_RL - W0", "W_S1": "gain_S1 = W_S1 - W0", "W_S1P": "gain_S1P = W_S1P - W0"}
    rows = []
    rows.append(["start model (F0 for arm F, W0 for the others), chrF++"] + [_ab2ci(p["macro"]["chrf"]["F0" if a == "S" else "W0"]) for a, _, _ in arms_cols])
    rows.append(["end model, chrF++"] + [_ab2ci(p["macro"]["chrf"][sysn]) for _, sysn, _ in arms_cols])
    rows.append(["gain (end minus start), chrF++"] + [_ab2ci(p["contrasts"]["chrf"][ck[a]], sign=True) for a, _, _ in arms_cols])
    rows.append(["difference of the gains, gain_W minus gain_F (DiD)", "--", _ab2ci(p["contrasts"]["chrf"]["DiD = gain_W - gain_F"], sign=True), "--", "--"])
    rows.append(["W_RL minus the control, chrF++ (RL minus continued SFT)", "--", "--", _ab2ci(p["contrasts"]["chrf"]["RL - S1 = W_RL - W_S1"], sign=True), _ab2ci(p["contrasts"]["chrf"]["RL - S1P = W_RL - W_S1P"], sign=True)])
    rows.append(["finished within 768 tokens, before / after (%)"] + [" / ".join(f"{100 * float(x):.1f}" for x in ab[a]["finished"].split(" / ")) for a, _, _ in arms_cols])
    rows.append(["gain of the finished rate (points)"] + [_ab2ci(p["contrasts"]["fin"][ck[a]], 1, True) for a, _, _ in arms_cols])
    rows.append(["gain, AfriCOMET with reference (x100)"] + [_ab2ci(p["contrasts"]["afri_ref"][ck[a]], 2, True) for a, _, _ in arms_cols])
    rows.append(["gain, SSA-COMET with reference (x100)"] + [_ab2ci(p["contrasts"]["ssa_ref"][ck[a]], 2, True) for a, _, _ in arms_cols])
    rows.append(["sampled completions / training"] + [str(ab["S"]["completions"]), str(ab["W"]["completions"]), "none (SFT, 367 steps, about 48 M tokens)", "none (SFT, 3,136 samples)"])
    rows.append(["gain per million sampled completions (chrF++)"] + [str(ab["S"]["gain_per_m"]), str(ab["W"]["gain_per_m"]), "--", "--"])
    w("t_ablation", cols, rows, ["l", "r", "r", "r", "r"])


def t_ablation_sets():
    """Contrasts of the ablation on every evaluation set (chrF++, language macro; AB2 ab2_results.json): the primary set, the sentences finished by every system, and the three confirmation sets."""
    ab, full = _ab2()
    sets = [("FLORES+ `no_pair` (primary; 4 languages)", "FLORES+ no_pair", "all"), ("FLORES+ `no_pair`, only sentences finished by every system", "FLORES+ no_pair", "cf"),
            ("NTREX `no_pair` (yor, kin, orm)", "NTREX no_pair", "all"), ("NTREX, only finished by every system", "NTREX no_pair", "cf"),
            ("MAFAND-MT `no_pair` (yor, kin)", "MAFAND no_pair", "all"), ("MAFAND-MT, only finished by every system", "MAFAND no_pair", "cf"),
            ("SmolSent `no_pair` (4 languages)", "SMOL no_pair", "all"), ("SmolSent, only finished by every system", "SMOL no_pair", "cf")]
    keys = [("gain_F", "gain_F = F_RL - F0"), ("gain_W", "gain_W = W_RL - W0"), ("DiD", "DiD = gain_W - gain_F"), ("RL - S1", "RL - S1 = W_RL - W_S1"), ("RL - S1P", "RL - S1P = W_RL - W_S1P")]
    rows = []
    for lab, u, part in sets:
        c = full[u][part]["contrasts"]["chrf"]
        rows.append([lab] + [_ab2ci(c[k], 2, True) for _, k in keys] + [_ab2ci(full[u][part]["ratios"]["chrf"]["survival = gain_W / gain_F"], 2)])
    w("t_ablation_sets", ["set (chrF++, language macro)"] + [a for a, _ in keys] + ["survival gain_W / gain_F"], rows, ["l", "r", "r", "r", "r", "r", "r"])


def t_ablation_metrics():
    """Contrasts of the ablation on the primary set by metric, all sentences and only sentences finished by every system (AB2 ab2_results.json). AfriCOMET-QE is the RL reward and SSA-COMET-QE is reported only."""
    ab, full = _ab2()
    ms = [("chrF++", "chrf", 2), ("spBLEU", "spbleu", 2), ("AfriCOMET, reference (x100)", "afri_ref", 2), ("SSA-COMET, reference (x100)", "ssa_ref", 2), ("finished rate (points)", "fin", 1), ("GlotLID-ok rate (points)", "lid", 1), ("AfriCOMET-QE (x100; the RL reward, not evidence)", "afri_qe", 2)]
    keys = [("gain_F", "gain_F = F_RL - F0"), ("gain_W", "gain_W = W_RL - W0"), ("DiD", "DiD = gain_W - gain_F"), ("RL - S1", "RL - S1 = W_RL - W_S1")]
    rows = []
    for part, tag in (("all", "all sentences"), ("cf", "finished by every system")):
        for lab, m, nd in ms:
            if part == "cf" and m == "fin":
                continue
            c = full["FLORES+ no_pair"][part]["contrasts"][m]
            rows.append([f"{lab}, {tag}"] + [_ab2ci(c[k], nd, True) for _, k in keys])
    w("t_ablation_metrics", ["metric, subset (FLORES+ `no_pair`, language macro)"] + [a for a, _ in keys], rows, ["l", "r", "r", "r", "r"])


def t_ablation_lang():
    """Per-language chrF++ contrasts of the ablation on FLORES+ `no_pair` (AB2 ab2_results.json contrasts_pl)."""
    ab, full = _ab2()
    keys = [("gain_F", "gain_F = F_RL - F0"), ("gain_W", "gain_W = W_RL - W0"), ("DiD", "DiD = gain_W - gain_F"), ("RL - S1", "RL - S1 = W_RL - W_S1")]
    names = {"yor": "Yoruba", "kin": "Kinyarwanda", "orm": "Oromo", "lin": "Lingala"}
    rows = []
    for part, tag in (("all", "all"), ("cf", "finished by every system")):
        n = full["FLORES+ no_pair"][part]["n"]
        for l in ("yor", "kin", "orm", "lin"):
            c = full["FLORES+ no_pair"][part]["contrasts_pl"]["chrf"][l]
            rows.append([f"{names[l]}, {tag} (n = {n[l]})"] + [_ab2ci(c[k], 2, True) for _, k in keys])
    w("t_ablation_lang", ["language, subset (chrF++)"] + [a for a, _ in keys], rows, ["l", "r", "r", "r", "r"])


# ------------------------------------------------------------------------------------------------ all-models comparison (E2 tables, P3 views)
def _read_md_table(name):
    lines = [l for l in open(os.path.join(TAB, name + ".md"), encoding="utf-8").read().splitlines() if l.startswith("|")]
    hdr = [c.strip() for c in lines[0].strip("|").split("|")]
    body = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
    return hdr, body


def _strip_ci(x):
    return re.sub(r"\s*\[[^\]]*\]", "", x)


def t_cmp_main():
    """Compact main table of the all-models comparison (point estimates; intervals are in t_model_comparison_main / computed_*)."""
    hdr, body = _read_md_table("t_model_comparison_main")
    want = ["Model", "Params", "chrF++ NLLB-43", "AfriCOMET (ours) 42 lang.", "chrF++ TPsy-19", "AfriCOMET (ours) TPsy-19", "chrF++ Afrique-21", "SSA-COMET Afrique-21, ours-scored x100"]
    idx = [hdr.index(c) for c in want]
    keep_pub = ("NLLB-200-3.3B", "NLLB-200-MoE-54.5B", "AfriqueGemma-4B", "AfriqueLlama-8B", "AfriqueGemma-12B", "AfriqueQwen-14B", "AfriqueQwen3.5-4B-50Langs")
    rows = []
    for r in body:
        name = r[0]
        computed = "computed" in r[-1]
        if computed and ("2B" in name and "TranslatePsy" in name):
            continue
        if not computed and name not in keep_pub:
            continue
        cells = [_strip_ci(r[i]) for i in idx]
        if not computed:
            cells[0] = name + " (published, whole devtest)"
        rows.append(cells)
    hd = ["Model", "params", "chrF++ NLLB-43", "AfriCOMET 42 lang.", "chrF++ TPsy-19", "AfriCOMET TPsy-19", "chrF++ Afrique-21", "SSA-COMET Afrique-21 (ours-scored)"]
    w("t_cmp_main", hd, rows, ["l"] + ["r"] * 7)


def t_cmp_wins():
    """Per-language wins of RL-final against each comparison system (no_pair rows, point estimates; 'clear' = 95% intervals do not overlap)."""
    import csv
    _pl = os.path.join(RDATA, "model_comparison_computed", "per_language_with_final2.csv")
    if not os.path.exists(_pl):
        _pl = os.path.join(RDATA, "model_comparison_computed", "per_language.csv")
    d = [r for r in csv.DictReader(open(_pl)) if r["subset"] == "no_pair"]
    by = defaultdict(dict)
    for r in d:
        by[r["model"]][r["iso"]] = r
    rl = by["rl"]
    num = lambda r, k: float(r[k]) if r.get(k, "") != "" else None
    rows = []
    for m, lab in (("nllb600m", "NLLB-200-distilled-600M"), ("nllb3b3", "NLLB-200-3.3B"), ("tpsy08", "TranslatePsy-AfriSLM-0.8B"), ("sft", "SFT-final (own)"), ("final2", "final run, final2 step 401 (own)")):
        if m not in by:
            continue
        o = by[m]
        com = [i for i in rl if i in o and num(o[i], "chrf") is not None]
        win = sum(num(rl[i], "chrf") > num(o[i], "chrf") for i in com)
        cw = sum(num(rl[i], "chrf_lo") > num(o[i], "chrf_hi") for i in com)
        cl = sum(num(o[i], "chrf_lo") > num(rl[i], "chrf_hi") for i in com)
        ca = [i for i in com if num(rl[i], "africomet") is not None and num(o[i], "africomet") is not None]
        wa = sum(num(rl[i], "africomet") > num(o[i], "africomet") for i in ca)
        rows.append([lab, len(com), f"{win}", f"{cw} / {cl}", len(ca), f"{wa}"])
    w("t_cmp_wins", ["comparison system", "languages (chrF++)", "RL-final higher (chrF++)", "clear win / clear loss (chrF++)", "languages (AfriCOMET)", "RL-final higher (AfriCOMET)"], rows, ["l"] + ["r"] * 5)


def _perlang_view(src, dst, keep):
    hdr, body = _read_md_table(src)
    idx = [hdr.index(c) for c in keep]
    names = {"SY RL-final": "RL-final", "SY SFT-final": "SFT-final", "NLLB-200-distilled-600M": "NLLB-600M", "NLLB-200-3.3B": "NLLB-3.3B", "TranslatePsy-AfriSLM-0.8B": "TPsy-0.8B", "NLLB-3.3B pub.": "NLLB-3.3B published"}
    w(dst, [names.get(hdr[i], hdr[i]) for i in idx], [[r[i] for i in idx] for r in body], ["l"] + ["r"] * (len(idx) - 1))


def t_cmp_perlang():
    _perlang_view("t_model_comparison_perlang_chrf", "t_cmp_perlang_chrf", ["Language", "n", "SY RL-final", "SY final run", "SY SFT-final", "NLLB-200-distilled-600M", "NLLB-200-3.3B", "TranslatePsy-AfriSLM-0.8B", "NLLB-3.3B pub."])
    _perlang_view("t_model_comparison_perlang_africomet", "t_cmp_perlang_africomet", ["Language", "n", "SY RL-final", "SY final run", "SY SFT-final", "NLLB-200-distilled-600M", "NLLB-200-3.3B", "TranslatePsy-AfriSLM-0.8B"])


def t_nllb_widen():
    """R17: NLLB-3.3B pseudo-reference signal widened to 24 languages (prepared for a next RL round; reports/nllb_signal_widening_2026-10-23.md)."""
    d = {k: v["nllb"] for k, v in json.load(open(os.path.join(RDATA, "nllb_reward_calibration_v2.json"))).items()}
    use = sorted([k for k, v in d.items() if v.get("use")], key=lambda k: -d[k]["teacher_vs_student"]["delta"])
    rows = []
    for k in use:
        p = d[k]["pseudoref"]; ci = p["policy_spearman_A_minus_current_ci"]
        rows.append([k, f"{d[k]['teacher_vs_student']['delta']:+.1f}", f"{p['policy_spearman_A']:.2f}", f"{p['policy_spearman_current_reward']:.2f}",
                     f"{p['policy_spearman_A_minus_current']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", f"{p['hack_auc_padded_A']:.2f}", f"{d[k]['weight']:.3f}", f"{d[k]['weight_C3']:.3f}"])
    w("t_nllb_widen", ["language", "teacher minus student (chrF++, FLORES+)", "Spearman with human chrF++: pseudo-ref signal", "current reward", "difference [95% CI]", "AUC human vs padded (signal)", "weight (no NLLB data in SFT mix)", "weight (C3 mix, double-pull discount)"], rows, ["l"] + ["r"] * 7)
    import csv
    da = list(csv.DictReader(open(os.path.join(RDATA, "nllb_widening_ssamte_human_da.csv"))))
    bl = {r["iso"]: r for r in csv.DictReader(open(os.path.join(RDATA, "nllb_widening_blend_da.csv")))}
    rows = []
    for r in da:
        b = bl[r["iso"]]
        rows.append([r["iso"], r["n_judgements"], f"{float(r['sp_A']):.2f}", f"{float(r['sp_africomet']):.2f}", f"{float(r['sp_ssa_mtl']):.2f}", f"{float(r['sp_QEens']):.2f}",
                     f"[{float(r['d_A_minus_ssa_lo']):+.2f}, {float(r['d_A_minus_ssa_hi']):+.2f}]",
                     f"{float(b['sp_w0.25']) - float(b['sp_w0.0']):+.3f} [{float(b['d_sp_w0.25_lo']):+.3f}, {float(b['d_sp_w0.25_hi']):+.3f}]"])
    w("t_nllb_widen_da", ["language", "judgements", "Spearman: pseudo-ref signal", "AfriCOMET-QE", "SSA-COMET-MTL", "QE ensemble", "signal minus SSA-COMET-MTL [95% CI]", "change of the QE ensemble when blended at w = 0.25 [95% CI]"], rows, ["l"] + ["r"] * 7)


def t_ab1_design():
    """Design of the controlled ablation as realised by agent AB2 (box folder /workspace/jobs/ab2; numbers copied from DECISION_RULE.txt, configs/rl_ab2_{full,weak}.yaml, ab2_sft_*.yaml, scripts/ab2_cost.py)."""
    rows = [
        ["languages, prompts", "yor, kin, orm, lin (the validated-QE languages with an AfriCOMET model that are not Ge'ez); b64 windows only; the RL prompt pool is drawn from the RL prompts v3 and decontaminated against every evaluation set (exact sentence or shared word 8-gram); uniform draws over the four languages", "49 steps x 64 prompts = 3,136 prompt draws (784 per language)", "`scripts/ab2_make_prompts.py`, `ab2_drawn_prompts.py`"],
        ["weak start (W0)", "SFT re-created from google/gemma-3-270m-it: the first 3,500 steps of SFT stage A (same packs, AdamW, language_alpha 0.5, seed 1234), learning rate annealed 5e-5 to 5e-6 over the shortened horizon", "3,500 steps x 131,072 tokens = 0.459 B tokens = 12.7% of the 27,638 steps / 3.62 B tokens of SFT-final", "`configs/ab2_sft_weak.yaml`"],
        ["strong start (F0)", "SFT-final (step 27,638)", "3.62 B tokens", "`models/sft_final`"],
        ["RL arms F and W", "phase-2c recipe: group 16, standardised advantages, lr 2e-6, $k_{3c}$ KL 0.05 with KL guard, warm-up 5 steps, QE + consensus reward, fresh optimiser, seed 1234, no science text, anchors, ensemble or long-document reward; identical prompts and seeds; configs differ only in `model_name`, `out_dir`, `run_name`", "49 steps x 64 prompts x group 16 = 50,176 completions per arm", "`configs/rl_ab2_full.yaml`, `rl_ab2_weak.yaml`"],
        ["control S1 (equal GPU time)", "W0 + continued SFT on the packs of the real stage-A schedule from step 3,501 on (92-93% of the draws are packs W never drew; all 77 languages as in the real schedule); fresh AdamW, warm-up 20, lr 5e-6 constant for 80% of the steps, then linear to 5e-7; N chosen so that the job's wall-clock equals one RL arm", "N = 432 (central) to 487 (high) steps in the cost model, fixed from the measured arm-F job (`configs/ab2_sft_s1.yaml` holds a placeholder of 420)", "`configs/ab2_sft_s1.yaml`, `scripts/ab2_set_n.py`"],
        ["control S1P (equal number of sentences)", "W0 + continued SFT on 784 short samples per chosen language from packs W never drew (same lr recipe, one epoch)", "3,136 samples = the number of RL prompt draws", "`configs/ab2_sft_s1pair.yaml`"],
        ["primary endpoint", "FLORES+ devtest `no_pair` rows, language-macro chrF++ over the four languages (n = 657 / 1,012 / 988 / 1,012 sentences for yor / kin / orm / lin), greedy, 768 new tokens, paired sentence bootstrap (B = 1,000, seed 0, 95% percentile intervals)", "systems F0, F_RL, W0, W_RL, W_S1, W_S1P", "`DECISION_RULE.txt` section 3"],
        ["confirmation", "NTREX (yor, kin, orm), MAFAND (yor, kin), SMOL-sent (all four), the sentences that every system finishes; spBLEU, SSA-COMET and AfriCOMET with reference; AfriCOMET-QE (the reward model) and SSA-COMET-QE are reported but enter no verdict", "--", "`DECISION_RULE.txt` section 3"],
        ["minimum detectable effect (80% power, two-sided 5%)", "1.16 chrF++ for the difference of the two RL gains (language macro), about 0.82 for RL against a control on the same sentences, about 2.3 for the difference of the gains in a single language", "from the paired-bootstrap standard errors of the main run's SFT-final to RL-final comparison (0.56 / 0.61 / 0.53 / 0.64 per language); one run per arm, so the run-to-run variance of RL is NOT included", "`ab2_analyze.py --power-from data/main_run_power_inputs.csv`"],
        ["preconditions (else verdict A is inconclusive)", "P1: F0 - W0 >= 3.0 chrF++; P2: the interval of the gain of arm F is above 0; P3: W0 finishes at least 60% of its FLORES+ `no_pair` translations within 768 tokens", "--", "`DECISION_RULE.txt` section 5"],
        ["budget", "4.60 GPU-hours = 5.48 USD (central) to 6.51 h = 7.75 USD (high) at 1.19 USD/h; owner's cap 9.00 USD including a 15% margin, i.e. a plan of at most 7.83 USD (6.58 h); optional extension of both RL arms to 98 steps: +1.02 (central) to +1.46 USD (high)", "weak SFT 2.74-3.53 h, arm F 0.40-0.59 h, arm W 0.42-0.66 h, S1 0.40-0.59 h, S1P 0.08-0.13 h, evaluations 0.30-0.51 h, hand-offs 0.25-0.50 h", "`scripts/ab2_cost.py`"],
    ]
    w("t_ab1_design", ["element", "as realised in the AB2 files", "size", "source"], rows, ["l", "l", "l", "l"])


def t_cmp_main_full():
    """Unabridged Table M1 without the source column (published rows are marked in the model name)."""
    hdr, body = _read_md_table("t_model_comparison_main")
    si = hdr.index("Source")
    rows = []
    for r in body:
        c = list(r)
        if "published" in c[si]:
            c[0] = c[0] + " (published)"
        rows.append(c[:si])
    hd = [h.replace("(ours-scored x100)", "").replace(", ours-scored x100", " (ours, x100)").replace(", paper's", " (paper's)") for h in hdr[:si]]
    w("t_cmp_main_full", hd, rows, ["l"] + ["r"] * (si - 1))


# ------------------------------------------------------------------------------------------------ E1 pairwise benchmark comparison
PW = os.path.join(RDATA, "pairwise_benchmark_comparison")
PW_METRICS = [("chrf", "chrF++ (all rows)"), ("chrf_sub", "chrF++ (rows scored by the neural metrics)"), ("afri_ref", "AfriCOMET, reference-based"), ("ssa_ref", "SSA-COMET-MTL, reference-based"),
              ("ens_ref", "majority of the three reference-based metrics"), ("afri_qe", "AfriCOMET-QE (the RL reward model)"), ("ssa_qe", "SSA-COMET-MTL, reference-free")]


def _pw_macro(sub, name="main"):
    import csv
    return {(r["subset"], r["b"], r["scope"], r["metric"]): r for r in csv.DictReader(open(os.path.join(PW, name, "macro.csv")))}


def _pct(x, d=1):
    return f"{100 * float(x):.{d}f}"


def t_pw():
    import csv
    M = _pw_macro(None)
    # T1: headline, no_pair and clean
    rows = []
    for key, lab in PW_METRICS:
        r = M.get(("no_pair", "ALL", "all_langs", key)); c = M.get(("clean", "ALL", "all_langs", key))
        if r is None:
            continue
        dd = (f"{float(r['mean_a']):.3g}", f"{float(r['mean_b']):.3g}", f"{float(r['delta']):+.3g} [{float(r['delta_lo']):+.3g}, {float(r['delta_hi']):+.3g}]") if key != "ens_ref" else ("", "", "")
        if key in ("afri_ref", "ssa_ref", "afri_qe", "ssa_qe"):
            dd = (f"{float(r['mean_a']):.3f}", f"{float(r['mean_b']):.3f}", f"{float(r['delta']):+.3f} [{float(r['delta_lo']):+.3f}, {float(r['delta_hi']):+.3f}]")
        elif key != "ens_ref":
            dd = (f"{float(r['mean_a']):.2f}", f"{float(r['mean_b']):.2f}", f"{float(r['delta']):+.2f} [{float(r['delta_lo']):+.2f}, {float(r['delta_hi']):+.2f}]")
        rows.append([lab, r["n_langs"], dd[0], dd[1], dd[2], f"{_pct(r['macro_win'])} / {_pct(r['macro_tie'])} / {_pct(r['macro_loss'])}",
                     f"{_pct(r['micro_pref'])} [{_pct(r['micro_pref_lo'])}, {_pct(r['micro_pref_hi'])}]", f"{r['langs_b_better_sig']} / {r['langs_b_worse_sig']}" if key != "ens_ref" else "",
                     f"{_pct(c['micro_pref'])} [{_pct(c['micro_pref_lo'])}, {_pct(c['micro_pref_hi'])}]"])
    w("t_pw_headline", ["metric", "languages", "SFT-final", "RL-final", "difference [95% CI]", "RL wins / ties / loses % (macro)", "RL preferred % of non-tied sentences [95% CI]", "languages RL clearly better / worse", "`clean` subset: RL preferred % [95% CI]"], rows, ["l"] + ["r"] * 8)
    # T2: by benchmark
    rows = []
    names = OrderedDict([("flores_plus", "FLORES+ devtest"), ("ntrex", "NTREX-128"), ("mafand", "MAFAND-MT"), ("smol_sent", "SmolSent"), ("ibom_mt", "IbomMT"), ("flores_moz", "FLORES-MOZ"), ("salt", "SALT"), ("glot_story", "GlotStoryBook")])
    for b, lab in names.items():
        row = [lab]
        first = M.get(("no_pair", b, "all_langs", "chrf"))
        row.append(first["n_langs"] if first else "")
        for key in ("chrf", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe"):
            r = M.get(("no_pair", b, "all_langs", key))
            if r is None:
                row.append("")
            else:
                d = 2 if key == "chrf" else 3
                row.append(f"{float(r['delta']):+.{d}f}; {_pct(r['micro_pref'], 0)}%")
        rows.append(row)
    w("t_pw_bench", ["benchmark (no_pair rows)", "languages", "chrF++: delta; RL preferred", "AfriCOMET-ref", "SSA-COMET-ref", "AfriCOMET-QE", "SSA-COMET QE"], rows, ["l"] + ["r"] * 6)
    # T3: by group / family / tier
    rows = []
    for sc in ["all_langs", "group:qe_validated", "group:blend", "group:consensus_only", "group:sft_only", "family:Afroasiatic", "family:Atlantic-Congo (non-Bantu)", "family:Bantu", "family:Creole / Pidgin", "family:Mande", "family:Nilo-Saharan",
               "script:Ge'ez (amh, tir)", "tier:strong", "tier:medium", "tier:weak"]:
        c = M.get(("no_pair", "ALL", sc, "chrf"))
        if c is None:
            continue
        row = [sc.replace("_", " "), c["n_langs"], f"{float(c['delta']):+.2f}; {_pct(c['micro_pref'], 0)}%"]
        for key in ("afri_ref", "ssa_ref"):
            r = M.get(("no_pair", "ALL", sc, key))
            row.append(f"{float(r['delta']):+.3f}; {_pct(r['micro_pref'], 0)}%" if r else "no metric")
        rows.append(row)
    w("t_pw_groups", ["scope (no_pair rows, all benchmarks)", "languages", "chrF++: delta; RL preferred", "AfriCOMET-ref", "SSA-COMET-ref"], rows, ["l", "r", "r", "r", "r"])
    # T13: without loops
    rows = []
    for r in csv.DictReader(open(os.path.join(PW, "main", "robustness_no_loop.csv"))):
        if r["metric"] not in ("chrf", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe"):
            continue
        rows.append([r["subset"], dict(PW_METRICS)[r["metric"]] if r["metric"] in dict(PW_METRICS) else r["metric"], r["n"], f"{float(r['delta_macro_lang']):+.3g}", f"{_pct(r['win'])} / {_pct(r['tie'])} / {_pct(r['loss'])}", _pct(r["pref"])])
    w("t_pw_noloop", ["sentences", "metric", "n", "macro delta", "RL wins / ties / loses %", "RL preferred %"], rows, ["l", "l", "r", "r", "r", "r"])
    # T5: agreement
    lab = {"chrf_sub": "chrF++", "afri_ref": "AfriC-ref", "ssa_ref": "SSA-ref", "afri_qe": "AfriC-QE", "ssa_qe": "SSA-QE"}
    rows = []
    for r in csv.DictReader(open(os.path.join(PW, "main", "agreement.csv"))):
        if r["subset"] != "no_pair" or r["scope"] != "all_langs" or r["metric_1"] not in lab or r["metric_2"] not in lab:
            continue
        rows.append([f"{lab[r['metric_1']]} vs {lab[r['metric_2']]}", r["n_both_nontie"], _pct(r["agree_nontie"]), f"{float(r['kappa_3class']):.2f}", f"{float(r['spearman_delta']):.2f}"])
    w("t_pw_agree", ["pair of metrics", "sentences both call a winner", "same winner %", "Cohen kappa (3 classes)", "Spearman of the deltas"], rows, ["l", "r", "r", "r", "r"])
    # T11 / T12: long input
    D = _pw_macro(None, "longdoc_docs"); S = _pw_macro(None, "longdoc_segments")
    order = ["core_b64", "core_b256", "core_b512", "core_b1024", "core_b2048", "core_b4096", "human_sm", "human_st512", "human_st1024", "human_st2048"]
    rows = []
    for b in order:
        c = D.get(("all", b, "all_langs", "chrf")); f = D.get(("all", b, "all_langs", "fin")); l = D.get(("all", b, "all_langs", "loop"))
        if not c:
            continue
        rows.append([b.replace("core_", "NLLB-ref ").replace("human_sm", "human, short").replace("human_st", "human, stitched ~"), c["n_langs"], c["n_sent"], f"{float(c['delta']):+.2f}", f"{_pct(c['micro_pref'], 0)}",
                     f"{_pct(f['mean_a'], 0)} / {_pct(f['mean_b'], 0)}", f"{_pct(l['mean_a'], 0)} / {_pct(l['mean_b'], 0)}"])
    w("t_pw_docs", ["documents (clean, sentence-by-sentence decoding)", "languages", "documents", "chrF++ delta", "RL preferred % (chrF++)", "finished %: SFT / RL", "with a loop %: SFT / RL"], rows, ["l"] + ["r"] * 6)
    rows = []
    for b in ["core_b64", "core_b256", "core_b512", "core_b1024", "human_sm", "human_st512", "human_st1024"]:
        c = S.get(("all", b, "all_langs", "chrf"))
        if not c:
            continue
        row = [b.replace("core_", "NLLB-ref ").replace("human_sm", "human, short").replace("human_st", "human, stitched ~"), c["n_sent"], f"{float(c['delta']):+.2f}; {_pct(c['micro_pref'], 0)}%"]
        for key in ("afri_ref", "ssa_ref"):
            r = S.get(("all", b, "all_langs", key))
            row.append(f"{float(r['delta']):+.3f}; {_pct(r['micro_pref'], 0)}%" if r else "")
        rows.append(row)
    w("t_pw_segs", ["aligned sentences inside the documents", "sentences (chrF++)", "chrF++: delta; RL preferred", "AfriCOMET-ref", "SSA-COMET-ref"], rows, ["l", "r", "r", "r", "r"])
    # A: per language (no_pair, all benchmarks pooled)
    U = {}
    for r in csv.DictReader(open(os.path.join(PW, "main", "unit_stats.csv"))):
        if r["subset"] == "no_pair" and r["b"] == "ALL" and r["metric"] in ("chrf", "afri_ref", "ssa_ref", "fin"):
            U[(r["iso"], r["metric"])] = r
    isos = sorted({k[0] for k in U}, key=lambda i: -float(U[(i, "chrf")]["delta"]))
    rows = []
    for i in isos:
        c = U[(i, "chrf")]
        row = [i, c["group"].replace("_", " "), c["n"], f"{float(c['delta']):+.2f}" + ("*" if float(c["delta_lo"]) > 0 else (" (worse)" if float(c["delta_hi"]) < 0 else "")), _pct(c["pref"], 0)]
        for key in ("afri_ref", "ssa_ref"):
            r = U.get((i, key))
            row.append((f"{float(r['delta']):+.3f}" + ("*" if float(r["delta_lo"]) > 0 else "") + f"; {_pct(r['pref'], 0)}%") if r else "n/a")
        f = U.get((i, "fin"))
        row.append(f"{_pct(f['mean_a'], 0)} / {_pct(f['mean_b'], 0)}" if f else "")
        rows.append(row)
    w("t_pw_perlang", ["language", "reward group", "sentences", "chrF++ delta", "RL preferred % (chrF++)", "AfriCOMET-ref: delta; preferred", "SSA-COMET-ref: delta; preferred", "finished %: SFT / RL"], rows, ["l", "l", "r", "r", "r", "r", "r", "r"])


# ------------------------------------------------------------------------------------------------ P5: evaluation of the final checkpoint (final2, step 401)
F2D = os.path.join(DATA, "final2_eval")


def _rows(name):
    import csv
    with open(os.path.join(F2D, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ci(d, lo, hi, nd=2):
    return f"{float(d):+.{nd}f} [{float(lo):+.{nd}f}, {float(hi):+.{nd}f}]"


def t_final2_eval():
    """final2 (step 401) on FLORES+ / NTREX-128 / MAFAND-MT / SmolSent (chrF++, spBLEU; paired with RL-final and SFT-final), sentence-level win/tie/loss, AfriScience-MT."""
    mac = _rows("macro_bench.csv")
    par = _rows("paired_bench.csv")
    bn = OrderedDict([("flores_plus", "FLORES+ devtest"), ("ntrex", "NTREX-128"), ("mafand", "MAFAND-MT test"), ("smol_sent", "SmolSent")])
    M = {(r["benchmark"], r["subset"], r["system"], r["metric"]): r for r in mac}
    P = {(r["benchmark"], r["subset"], r["a"], r["metric"]): r for r in par}
    rows = []
    for b, nm in bn.items():
        for sub in ("no_pair", "clean"):
            g = lambda s, m: M[(b, sub, s, m)]
            pc, ps = P[(b, sub, "rl", "chrf")], P[(b, sub, "rl", "spbleu")]
            rows.append([(nm + ("$^\\dagger$" if b in ("ntrex", "smol_sent") else "")) if sub == "no_pair" else "", sub.replace("_", "\\_"), g("rl", "chrf")["k"], f1(float(g("sft", "chrf")["value"])), f1(float(g("rl", "chrf")["value"])),
                         f1(float(g("final2", "chrf")["value"])), _ci(pc["delta"], pc["lo"], pc["hi"]), _ci(ps["delta"], ps["lo"], ps["hi"]),
                         f"{pc['langs_up']} / {pc['langs_clearly_better']} / {pc['langs_clearly_worse']}", f"{float(g('rl', 'chrf')['finished']):.1f} / {float(g('final2', 'chrf')['finished']):.1f}"])
    w("t_final2_bench", ["benchmark", "subset", "langs", "chrF++ SFT-final", "chrF++ RL-final", "chrF++ final2", "$\\Delta$ chrF++ final2 $-$ RL-final [95% CI]", "$\\Delta$ spBLEU final2 $-$ RL-final [95% CI]",
                         "languages higher / clearly higher / clearly lower (chrF++)", "finished % RL / final2"], rows, ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r"])
    # sentence-level win / tie / loss (chrF++, threshold 1.0), pooled over languages
    wt = {(r["benchmark"], r["subset"]): r for r in _rows("sentence_wtl_final2_vs_rl.csv")}
    rows = []
    for b, nm in bn.items():
        r = wt[(b, "no_pair")]
        rows.append([nm, f"{int(r['n']):,}", r["mean_rl"], r["mean_final2"], _ci(r["delta"], r["delta_lo"], r["delta_hi"]), f"{100 * float(r['win']):.1f} / {100 * float(r['tie']):.1f} / {100 * float(r['loss']):.1f}",
                     f"{100 * float(r['pref']):.1f} [{100 * float(r['pref_lo']):.1f}, {100 * float(r['pref_hi']):.1f}]"])
    w("t_final2_wtl", ["benchmark (`no_pair` rows)", "sentence pairs", "mean sentence chrF++ RL-final", "final2", "$\\Delta$ [95% CI]", "final2 wins / ties / loses %", "final2 preferred, % of non-tied [95% CI]"], rows,
      ["l", "r", "r", "r", "r", "r", "r"])
    # per language (appendix)
    pl = _rows("per_language_bench.csv")
    by = defaultdict(dict)
    for r in pl:
        if r["subset"] == "no_pair":
            by[r["iso"]][r["benchmark"]] = r
    sys_path_added = False
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_model_comparison import LANG
    def cell(r):
        if r is None:
            return "--"
        lo, hi, d = float(r["d_final2_rl_chrf_lo"]), float(r["d_final2_rl_chrf_hi"]), float(r["d_final2_rl_chrf"])
        return f"{d:+.2f}" + ("\\*" if lo > 0 or hi < 0 else "")
    rows = []
    for iso in sorted(by, key=lambda i: LANG.get(i, i)):
        d = by[iso]
        n = d["flores_plus"]["n"] if "flores_plus" in d else next(iter(d.values()))["n"]
        rows.append([f"{LANG.get(iso, iso)} ({iso})"] + [cell(d.get(b)) for b in bn] + [f"{float(d['flores_plus']['rl_chrf']):.1f} / {float(d['flores_plus']['final2_chrf']):.1f}" if "flores_plus" in d else "--"])
    w("t_final2_perlang", ["language", "FLORES+", "NTREX-128", "MAFAND-MT", "SmolSent", "FLORES+ chrF++ RL-final / final2"], rows, ["l", "r", "r", "r", "r", "r"])

    # AfriScience-MT
    am = _rows("afs_macro.csv")
    ap = _rows("afs_paired.csv")
    AM = {(r["split"], r["level"], r["system"], r["metric"]): r for r in am}
    AP = {(r["split"], r["level"], r["a"], r["metric"]): r for r in ap}
    def v(sp, lv, sy, m, nd=2, pct=False):
        r = AM[(sp, lv, sy, m)]
        x = float(r["value"]) * (100 if pct else 1)
        return f"{x:.{nd}f}"
    cols = [("dev", "sentence", "chrfpp"), ("dev", "document", "chrf"), ("dev", "document", "fin"), ("test", "sentence", "chrfpp"), ("test", "document", "chrf"), ("test", "document", "chrfpp"), ("test", "document", "fin")]
    rows = []
    for sy, nm in (("rl_r7", "RL-final (phase 2c, step 1,433)"), ("a2w", "science-stage start A2W ($\\alpha=0.7$)"), ("final2", "final run, final2 step 401")):
        rows.append([nm] + [v(sp, lv, sy, m, 1 if m == "fin" else 2, m == "fin") for sp, lv, m in cols])
    for a, nm in (("rl_r7", "final2 $-$ RL-final"), ("a2w", "final2 $-$ A2W")):
        r = []
        for sp, lv, m in cols:
            x = AP[(sp, lv, a, m)]
            sc = 100 if m == "fin" else 1
            r.append(_ci(float(x["delta"]) * sc, float(x["lo"]) * sc, float(x["hi"]) * sc, 1 if m == "fin" else 2))
        rows.append([nm + " [95% CI]"] + r)
    w("t_final2_afriscience", ["AfriScience-MT (6 languages)", "dev sentence chrF++", "dev paper chrF", "dev paper finished %", "test sentence chrF++", "test paper chrF", "test paper chrF++", "test paper finished %"], rows,
      ["l"] + ["r"] * 7)
    # published numbers (test, chrF without ++) next to ours
    txt = open(os.path.join(F2D, "afriscience_eval_final2_step401.md"), encoding="utf-8").read()
    def parse(section):
        out = {}
        blk = txt.split(section)[1].split("###")[0]
        for l in blk.splitlines():
            if l.startswith("| published chrF:") or l.startswith("| chrf |"):
                c = [x.strip() for x in l.strip("|").split("|")]
                out[c[0].replace("published chrF: ", "").replace("doc: ", "")] = c[1:]
        return out
    sent_t = parse("test - sentence level")
    doc_t = parse("test - document level")
    rows = []
    for nm, vals in sent_t.items():
        if nm == "chrf":
            continue
        rows.append([nm, "published (sentence)"] + vals)
    ours = {"RL-final (1,433)": ("rl_r7",), "science-stage start A2W": ("a2w",), "final2 step 401": ("final2",)}
    ro = []
    for nm, (sy,) in ours.items():
        r = AM[("test", "sentence", sy, "chrf")]
        ro.append([nm, "ours (sentence)"] + [f"{float(r[l]):.1f}" for l in ("amh", "hau", "lug", "nso", "yor", "zul")] + [f"{float(r['value']):.1f}"])
    for nm, vals in doc_t.items():
        if nm == "chrf":
            continue
        rows.append([nm, "published (paper)"] + vals)
    for nm, (sy,) in ours.items():
        r = AM[("test", "document", sy, "chrf")]
        ro.append([nm, "ours (paper)"] + [f"{float(r[l]):.1f}" for l in ("amh", "hau", "lug", "nso", "yor", "zul")] + [f"{float(r['value']):.1f}"])
    allr = [x for x in ro if "sentence" in x[1]] + [x for x in rows if "sentence" in x[1]] + [x for x in ro if "paper" in x[1]] + [x for x in rows if "paper" in x[1]]
    w("t_final2_afriscience_pub", ["system", "level", "amh", "hau", "lug", "nso", "yor", "zul", "macro"], allr, ["l", "l"] + ["r"] * 7)


def t_final2_pairwise():
    """G1: final2 (step 401) against RL-final, sentence by sentence, with the learned reference-based metrics (AfriCOMET-ref, SSA-COMET-ref) and chrF++; `no_pair` rows; paired sentence bootstrap."""
    rows_ = [r for r in _rows("g1_pairwise_final2_summary.csv") if r["baseline"] == "rl"]
    S = {(r["scope"], r["metric"]): r for r in rows_}
    scopes = OrderedDict([("ntrex", "NTREX-128"), ("mafand", "MAFAND-MT test"), ("smol_sent", "SmolSent"), ("ntrex+mafand+smol_sent", "the three pooled"), ("flores_plus", "FLORES+ devtest")])
    mets = OrderedDict([("chrf", "chrF++ (sentence)"), ("afri_ref", "AfriCOMET-ref"), ("ssa_ref", "SSA-COMET-ref")])
    rows = []
    for sc, nm in scopes.items():
        for m, mn in mets.items():
            r = S.get((sc, m))
            if r is None:
                continue
            nd = 2 if m == "chrf" else 4
            ci = f"{float(r['delta']):+.{nd}f} [{float(r['delta_lo']):+.{nd}f}, {float(r['delta_hi']):+.{nd}f}]"
            rows.append([nm if m == "chrf" else "", mn, r["langs"], f"{int(r['n']):,}", f"{float(r['mean_base']):.{nd}f}", f"{float(r['mean_final2']):.{nd}f}", ci,
                         f"{100 * float(r['macro_win']):.1f} / {100 * float(r['macro_tie']):.1f} / {100 * float(r['macro_loss']):.1f}",
                         f"{100 * float(r['pref']):.1f} [{100 * float(r['pref_lo']):.1f}, {100 * float(r['pref_hi']):.1f}]",
                         f"{r['langs_up']} / {r['langs_clearly_better']} / {r['langs_clearly_worse']}"])
    w("t_final2_pw", ["benchmark (`no_pair` rows)", "metric", "languages", "sentences", "RL-final (macro mean)", "final2", "$\\Delta$ final2 $-$ RL-final [95% CI]", "final2 wins / ties / loses % (macro)",
                      "final2 preferred, % of non-tied [95% CI]", "languages higher / clearly higher / clearly lower"], rows, ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r"])


def t_final2_pw_perlang():
    """G1: per language (NTREX-128, MAFAND-MT, SmolSent pooled, `no_pair`): final2 minus RL-final in sentence chrF++, AfriCOMET-ref and SSA-COMET-ref."""
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_model_comparison import LANG
    R = [r for r in _rows("g1_pairwise_final2_per_language.csv") if r["baseline"] == "rl" and r["scope"] == "ntrex+mafand+smol_sent"]
    by = defaultdict(dict)
    for r in R:
        by[r["iso"]][r["metric"]] = r

    def cell(r, nd):
        if r is None:
            return "--"
        d, lo, hi = float(r["delta"]), float(r["lo"]), float(r["hi"])
        return f"{d:+.{nd}f}" + ("\\*" if lo > 0 or hi < 0 else "")
    rows = []
    for iso in sorted(by, key=lambda i: LANG.get(i, i)):
        d = by[iso]
        rows.append([f"{LANG.get(iso, iso)} ({iso})", f"{int(d['chrf']['n']):,}", cell(d.get("chrf"), 2), cell(d.get("afri_ref"), 4), cell(d.get("ssa_ref"), 4),
                     f"{100 * float(d['afri_ref']['win']):.0f} / {100 * float(d['afri_ref']['loss']):.0f}" if "afri_ref" in d else "--"])
    w("t_final2_pw_perlang", ["language", "sentences (chrF++)", "$\\Delta$ sentence chrF++", "$\\Delta$ AfriCOMET-ref", "$\\Delta$ SSA-COMET-ref", "AfriCOMET-ref: final2 wins / loses %"], rows, ["l", "r", "r", "r", "r", "r"])


def t_judge_pilot():
    """P11 (2026-10-25): LLM-as-judge validation pilot (reports/data/judge_pilot_2026-10-25/). Writes the long-paper tables and the short-paper appendix table."""
    d = os.path.join(RDATA, "judge_pilot_2026-10-25")
    cor = [r for r in read_csv(os.path.join(d, "correlations.csv")) if r["kind"] == "spearman"]
    pdf = [r for r in read_csv(os.path.join(d, "paired_differences.csv")) if r["kind"] == "spearman"]
    cost = {r["judge"]: r for r in read_csv(os.path.join(d, "cost_and_failures.csv"))}
    PRO, FLASH, SSA = "gemini-3.1-pro-preview", "gemini-3.8-flash", "SSA-COMET-MTL(ref)"
    AFR, AFRM = "AfriCOMET-1.1-STL(ref)", "AfriCOMET-1.0-MTL(ref)"
    get = {(r["scorer"], r["scope"]): r for r in cor}
    pairs = [("eng-amh", "Amharic", "SSA-MTE"), ("eng-yor", "Yoruba", "SSA-MTE"), ("eng-kin", "Kinyarwanda", "SSA-MTE"), ("eng-hau", "Hausa", "SSA-MTE"),
             ("eng-swh", "Swahili", "AfriMTE"), ("eng-ibo", "Igbo", "AfriMTE"), ("eng-som", "Somali", "AfriMTE"), ("eng-xho", "Xhosa", "AfriMTE")]
    tot = [("POOLED", "pooled", "both"), ("SSA4_POOLED", "pooled SSA-MTE pairs", "SSA-MTE"), ("AFRI4_POOLED", "pooled AfriMTE pairs", "AfriMTE"),
           ("POOLED_DEMEANED", "pooled, language-demeaned", "both")]
    ci = lambda r: f"{float(r['r']):.3f} [{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}]"
    pt = lambda r: f"{float(r['r']):.3f}"
    rows = []
    for k, nm, src in [(a, f"{b} ({a[4:]})", c) for a, b, c in pairs] + tot:
        n = get[(PRO, k)]["n"]
        rows.append([nm, src, n, ci(get[(PRO, k)]), ci(get[(FLASH, k)]), pt(get[(SSA, k)]), pt(get[(AFR, k)]), pt(get[(AFRM, k)])])
    w("t_judge_pilot", ["pair", "set", "segments", "Gemini 3.1 Pro: Spearman [95% CI]", "Gemini 3.8 Flash: Spearman [95% CI]", "SSA-COMET-MTL (ref)", "AfriCOMET-STL-1.1 (ref)", "AfriCOMET-MTL-1.0 (ref)"],
      rows, ["l", "l", "r", "r", "r", "r", "r", "r"])
    pg = {(r["scope"], r["contrast"]): r for r in pdf}
    c1, c2 = f"{PRO} minus {FLASH}", f"{PRO} minus {SSA}"
    dc = lambda r: f"{float(r['diff']):+.3f} [{float(r['ci_lo']):+.2f}, {float(r['ci_hi']):+.2f}] (SE {float(r['boot_se']):.3f})"
    rows = []
    for k, nm, src in [(a, f"{b} ({a[4:]})", c) for a, b, c in pairs] + tot:
        rows.append([nm, pg[(k, c1)]["n"], dc(pg[(k, c1)]), dc(pg[(k, c2)])])
    w("t_judge_pilot_pairs", ["pair", "segments", "Pro minus Flash: difference of Spearman [95% CI] (bootstrap SE)", "Pro minus SSA-COMET-MTL (ref): difference [95% CI] (bootstrap SE)"], rows, ["l", "r", "r", "r"])
    rows = []
    for j, nm in ((PRO, "Gemini 3.1 Pro (gemini-3.1-pro-preview)"), (FLASH, "Gemini 3.8 Flash (gemini-3.8-flash)")):
        c = cost[j]
        rows.append([nm, c["n"], c["parse_failures"], f"{int(c['input_tokens']):,}", f"{int(c['output_tokens_incl_thinking']):,}", f"{float(c['mean_thought_tokens']):.0f}",
                     f"{float(c['cost_usd']):.2f}", f"{float(c['cost_usd']) / int(c['n']):.5f}"])
    w("t_judge_pilot_cost", ["judge", "segments", "parse failures", "input tokens", "output tokens (incl. thinking)", "mean thinking tokens per segment", "actual cost (USD)", "USD per segment"], rows, ["l"] + ["r"] * 7)
    # the short-paper judge tables (tab_judge_pilot.tex, tab_judge_run2.tex, tab_judge_run3.tex) are written by make_judge.py (P13, 2026-10-25)


if __name__ == "__main__":
    for fn in [t_main, t_groups, t_published, t_phases, t_longdoc, t_decoding, t_sft_stage, t_gemma31b, t_nllb, t_anchors, t_final2, t_master, t_groups_def, t_hparams, t_rq, t_geez, t_tokcost, t_rq_units, t_geez_data, t_ablation, t_ablation_sets, t_ablation_metrics, t_ablation_lang, t_cmp_main, t_cmp_wins, t_cmp_perlang, t_nllb_widen, t_ab1_design, t_cmp_main_full, t_pw, t_final2_eval, t_final2_pairwise, t_final2_pw_perlang, t_judge_pilot]:
        try:
            fn()
        except Exception as e:  # keep going so one broken source does not block the others
            import traceback; traceback.print_exc()
            print("FAILED", fn.__name__, e)
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_longctx.py"), "tables"], check=False)   # LC1 tables t_longctx_*.md
    subprocess.run([_sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_cpu.py"), "tables"], check=False)   # P7 tables t_tokcost_lang, t_cpu_*.md
    subprocess.run([_sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_tables_hardening.py")], check=False)   # P12 tables t_hardening_*.md (Section 7.9)
    subprocess.run([_sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_judge.py"), "tables"], check=False)   # P13 tables t_judge_j2_*.md (Section 6.12) and short-paper judge tables
    subprocess.run([_sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_noGeez.py")], check=False)   # P14 tables t_noGeez_flores, t_noGeez_judge (Sections 6.8, 6.12)
