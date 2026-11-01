#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render the TABLES of reports/generation_config_study_2026-10-22.md from reports/data/generation_config_study/json/*.json (written by the box scripts kept in
reports/data/generation_config_study/scripts/) and substitute them into the prose template next to this file's data (`report_template.md`, markers {{name}}).
Stdlib only. usage: build_generation_config_report.py [template] [out]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "reports" / "data" / "generation_config_study"
J = D / "json"


def load(name, default=None):
    p = J / name
    return json.load(open(p)) if p.exists() else default


G, P, Q, SEEDS, REP = load("grid.json"), load("pools.json", {}), load("qe.json", {}), load("seeds.json", {}), load("rep_by_group.json", {})
R5 = json.load(open(ROOT / "reports" / "data" / "final_model_decoding_study" / "results.json"))
SFT = load("sft.json", {})
RL, DOC, UNIT = load("rl_summary.json"), load("doc_tables.json"), load("unit_repair.json")
ISOS = list(G["greedy"]["per_lang_chrf"])
GRP = {"strong": "swh hau zul kin som".split(), "medium": "ibo sna lug bem nbl".split(), "weak": "yor amh wol tir kbp nus kab fon ach efi".split()}
TEMPS = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "1.0"]
SAMP = ["", "p0.8", "p0.9", "p0.95", "k10", "k20", "k50", "mp0.02", "mp0.05", "mp0.1"]
SAMP_LABEL = {"": "none", "p0.8": "top-p 0.8", "p0.9": "top-p 0.9", "p0.95": "top-p 0.95", "k10": "top-k 10", "k20": "top-k 20", "k50": "top-k 50", "mp0.02": "min-p 0.02", "mp0.05": "min-p 0.05", "mp0.1": "min-p 0.1"}


def star(v, key="all", m="chrf"):
    return "*" if v[f"lo_{m}_{key}"] > 0 or v[f"hi_{m}_{key}"] < 0 else ""


def fmt(v, key="all", m="chrf", dec=2):
    return f"{v[f'd_{m}_{key}']:+.{dec}f}{star(v, key, m)}"


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def name_of(t, s):
    return "T" + t + ("_" + s if s else "")


def t_heat(key="all", m="chrf"):
    rows = []
    for t in TEMPS:
        rows.append([f"T {t}"] + [(fmt(G[name_of(t, s)], key, m) if name_of(t, s) in G else "-") for s in SAMP])
    return md(rows, ["temperature"] + [SAMP_LABEL[s] for s in SAMP])


def t_group_best(top=5):
    out = []
    excl = lambda n: n == "greedy" or n.endswith(("_s1", "_s2", "_s3", "_s4")) or "_n16" in n
    for grp in ("all", "strong", "medium", "weak"):
        rows = []
        for n, v in sorted(((n, v) for n, v in G.items() if not excl(n)), key=lambda kv: -kv[1][f"d_chrf_{grp}"])[:top]:
            rows.append([f"`{n}`", f"{v[f'chrf_{grp}']:.2f}", fmt(v, grp), f"[{v[f'lo_chrf_{grp}']:+.2f}, {v[f'hi_chrf_{grp}']:+.2f}]", fmt(v, grp, "bleu")])
        out.append(f"**{grp}** (greedy {G['greedy'][f'chrf_{grp}']:.2f} chrF++, {G['greedy'][f'bleu_{grp}']:.2f} spBLEU)\n\n" + md(rows, ["config", "chrF++", "delta chrF++", "95% CI", "delta spBLEU"]))
    return "\n\n".join(out)


CANON = ["greedy", "T0.2_mp0.05", "T0.3", "T0.5", "T0.7", "T0.7_mp0.05", "T0.8_p0.9", "T1.0", "T1.0_p0.95", "T1.0_k64_p0.95", "T1.0_p0.9", "T1.0_mp0.05", "T1.0_mp0.1", "hf_T1.0_typ0.9", "hf_T1.0_eps0.02"]


def t_canon():
    rows = []
    for n in CANON:
        if n not in G: continue
        v = G[n]
        rows.append([f"`{n}`", f"{v['chrf_all']:.2f}", fmt(v), fmt(v, "all", "bleu"), fmt(v, "strong"), fmt(v, "medium"), fmt(v, "weak"), f"{v['loop']*100:.2f}", f"{v['rep3']*100:.1f}", f"{v['num_kept']*100:.1f}", f"{v['name_kept']*100:.1f}", f"{v['lid']*100:.1f}"])
    return md(rows, ["config", "chrF++", "delta chrF++", "delta spBLEU", "delta strong", "delta medium", "delta weak", "loop %", "repeated 3-gram %", "numbers kept %", "names kept %", "LID %"])


def t_hf():
    rows = []
    for n, v in sorted(((n, v) for n, v in G.items() if n.startswith("hf_")), key=lambda kv: -kv[1]["d_chrf_all"]):
        rows.append([f"`{n[3:]}`", f"{v['chrf_all']:.2f}", fmt(v), f"[{v['lo_chrf_all']:+.2f}, {v['hi_chrf_all']:+.2f}]", fmt(v, "strong"), fmt(v, "medium"), fmt(v, "weak")])
    return md(rows, ["HF sampling config (T, then cut-off)", "chrF++", "delta chrF++", "95% CI", "strong", "medium", "weak"])


def t_seeds():
    rows = [[f"`{c}`", ", ".join(f"{x:.2f}" for x in v["seeds_chrf"]), f"{v['mean']:.2f}", f"{v['sd']:.2f}", f"{v['mean_d']:+.2f}", f"{v['spbleu_mean']:.2f} +- {v['spbleu_sd']:.2f}"] for c, v in SEEDS.items()]
    return md(rows, ["config", "chrF++ of the 5 seeds", "mean", "sd", "mean delta vs greedy", "spBLEU mean +- sd"])


def t_rep():
    rows = []
    for n, r in REP.items():
        key = n if n in G else None
        d = ""
        if n in G: d = fmt(G[n])
        elif n in P: d = fmt(P[n])
        elif n.startswith("R5:"):
            c = R5["bootstrap"]["chrf"]["all"].get(n[3:])
            d = f"{c['d_vs_greedy']:+.2f}" if c else ""
        elif "__" in n:
            k = n.replace("__", ":")
            if k in P: d = fmt(P[k])
        rows.append([f"`{n.replace('__', ':')}`", d, f"{r['all']*100:.1f}", f"{r['strong']*100:.1f}", f"{r['medium']*100:.1f}", f"{r['weak']*100:.1f}"])
    return md(rows, ["config", "delta chrF++", "repeated 3-gram, all %", "strong %", "medium %", "weak %"])


def t_pools(methods=("mbr",), ns=(4, 8, 16)):
    rows = []
    for pool in ["T0.3_n16", "T0.5_n16", "T0.5_mp0.05_n16", "T0.7_n16", "T0.7_mp0.05_n16", "T0.7_p0.9_n16", "T1.0_n16", "T1.0_p0.95_n16", "T1.0_p0.9_n16", "T1.0_mp0.1_n16", "T1.0_mp0.05_n16"]:
        r = [f"`{pool[:-4]}`"]
        for m in methods:
            for n in ns:
                v = P.get(f"{m}{n}:{pool}")
                r.append(fmt(v) if v else "-")
        rows.append(r)
    return md(rows, ["sampler of the pool"] + [f"{m} n={n}" for m in methods for n in ns])


def t_pool_groups():
    rows = []
    for pool in ["T0.5_n16", "T0.7_n16", "T0.7_mp0.05_n16", "T1.0_mp0.05_n16", "T1.0_mp0.1_n16"]:
        for n in (8, 16):
            v = P.get(f"mbr{n}:{pool}")
            if v: rows.append([f"`{pool[:-4]}` n={n}", f"{v['chrf_all']:.2f}", fmt(v), fmt(v, "all", "bleu"), fmt(v, "strong"), fmt(v, "medium"), fmt(v, "weak"), f"{v['tok_x_greedy']:.0f}x"])
    return md(rows, ["MBR (chrF++ consensus)", "chrF++", "delta chrF++", "delta spBLEU", "strong", "medium", "weak", "tokens vs greedy"])


def t_pool_best_group():
    rows = []
    for n in (8, 16):
        for grp in ("all", "strong", "medium", "weak"):
            c = sorted(((v[f"d_chrf_{grp}"], k, v) for k, v in P.items() if v.get("method") == "mbr" and v["n"] == n), key=lambda x: -x[0])[:3]
            rows.append([f"n = {n}", grp] + [f"`{k.split(':')[1][:-4]}` {d:+.2f} [{v[f'lo_chrf_{grp}']:+.2f}, {v[f'hi_chrf_{grp}']:+.2f}]" for d, k, v in c])
    return md(rows, ["pool size", "group", "best", "2nd", "3rd"])


def t_oracle():
    rows = []
    for pool in ["T0.7_n16", "T1.0_mp0.05_n16", "T1.0_n16"]:
        rows.append([f"`{pool[:-4]}`"] + [fmt(P[f"oracle{n}:{pool}"]) if f"oracle{n}:{pool}" in P else "-" for n in (4, 8, 16)])
    return md(rows, ["pool", "oracle n=4", "oracle n=8", "oracle n=16"])


def t_selection():
    rows = []
    for k in ["mbr16:T0.7_n16", "mbrg16:T0.7_n16", "mbrqe16__T0.7_n16", "qe16__T0.7_n16", "mbr16:T1.0_mp0.05_n16", "mbrg16:T1.0_mp0.05_n16", "mbrqe16__T1.0_mp0.05_n16", "qe16__T1.0_mp0.05_n16", "mbr8:T0.5_n16", "mbrg8:T0.5_n16", "mbrqe8__T0.5_n16", "qe8__T0.5_n16"]:
        v = P.get(k) or P.get(k.replace("__", ":"))
        if not v: continue
        qk = k.replace(":", "__")
        q = Q.get(qk, {})
        rows.append([f"`{k}`", f"{v['chrf_all']:.2f}", fmt(v), fmt(v, "all", "bleu"), f"{v['num_kept']*100:.1f}", f"{q['ref']['macro']:.4f} ({q['ref']['d']:+.4f})" if q else "-", f"{q['qe']['macro']:.4f} ({q['qe']['d']:+.4f})" if q else "-"])
    return md(rows, ["selection (pool, n)", "chrF++", "delta chrF++", "delta spBLEU", "numbers kept %", "AfriCOMET ref (delta)", "AfriCOMET-QE (delta)"])


def t_africomet():
    rows = []
    for n, q in sorted(Q.items(), key=lambda kv: -kv[1]["ref"]["macro"]):
        v = G.get(n) or P.get(n) or P.get(n.replace("__", ":"))
        rows.append([f"`{n.replace('__', ':')}`", f"{q['ref']['macro']:.4f}", f"{q['ref']['d']:+.4f} [{q['ref']['d_lo']:+.4f}, {q['ref']['d_hi']:+.4f}]", f"{q['qe']['macro']:.4f}", f"{q['qe']['d']:+.4f}", (fmt(v) if v else "")])
    return md(rows, ["config", "AfriCOMET ref", "delta [95% CI]", "AfriCOMET-QE", "delta QE", "delta chrF++"])


PL_CONFIGS = ["T0.3", "T0.2_mp0.05", "T0.5", "T0.7", "T1.0", "T1.0_p0.9", "T1.0_mp0.05", "mbr8:T0.7_n16", "mbr16:T0.7_n16", "mbr16:T0.5_n16", "mbr16:T1.0_mp0.05_n16", "mbrg16:T0.7_n16"]


def t_perlang():
    base = G["greedy"]["per_lang_chrf"]
    rows = [["greedy (chrF++)"] + [f"{base[i]:.1f}" for i in ISOS]]
    for c in PL_CONFIGS:
        v = G.get(c) or P.get(c)
        if v: rows.append([f"`{c}`"] + [f"{v['per_lang_chrf'][i]-base[i]:+.1f}" for i in ISOS])
    for b in ("beam4", "beam8"):
        pl = R5["configs"][b]["per_lang"]; rows.append([f"R5 `{b}` (HF)"] + [f"{pl[i]['chrf']-base[i]:+.1f}" for i in ISOS])
    grp = {i: g[0].upper() for g, ls in GRP.items() for i in ls}
    return md(rows, ["config"] + [f"{i} ({grp[i]})" for i in ISOS])


def t_rl():
    if not RL: return "_(RL measurement not available)_"
    rows = []
    for n, c in RL["configs"].items():
        m = c.get("reward_by_mode", {}); st = c["status"]
        bad = sum(v for k, v in st.items() if k != "ok") * 100
        rows.append([f"`{n}`", f"{m.get('qe', float('nan')):.3f}", f"{m.get('blend', float('nan')):.3f}", f"{m.get('consensus', float('nan')):.3f}", f"{c['reward_macro']:.3f}", f"{c['reward_by_bin'].get('b64', float('nan')):.3f}", f"{c['reward_by_bin'].get('b256', float('nan')):.3f}",
                     f"{bad:.2f}", "-" if c["group_std"] is None else f"{c['group_std']:.3f}", "-" if c["group_best"] is None else f"{c['group_best']:.3f}", "-" if c["flat_groups"] is None else f"{c['flat_groups']*100:.1f}"])
    return md(rows, ["rollout setting", "reward, 13 QE-mode languages", "reward, 5 blend languages", "reward, 2 consensus languages", "reward, all (macro)", "b64", "b256", "failed % (wrong language, degenerate, ...)", "within-group std", "mean best of group", "flat groups % (std < 0.05)"])


def t_rl_bias():
    if not RL or "bias_own" not in RL: return "_(bias measurement not available)_"
    fl = RL.get("t1_flags", {})
    rows = []
    for n, b in RL["bias_own"].items():
        f = fl.get(n, {})
        rows.append([f"`{n}`", b["n"], f"{max(b['kl_tok'], 0):.4f}", f"{max(b['kl_seq'], 0):.2f}",
                     f"{f['share_seq_with_cut_token']*100:.1f}" if f and f["share_seq_with_cut_token"] > 0 else "-", f"{f['share_tokens_cut']*100:.2f}" if f and f["share_seq_with_cut_token"] > 0 else "-",
                     f"{f['reward_inside']:.3f}" if f and f.get("reward_with_cut_token") is not None else "-", f"{f['reward_with_cut_token']:.3f}" if f and f.get("reward_with_cut_token") is not None else "-"])
    return md(rows, ["sampler q", "sequences", "KL(q, p) per token (nats)", "KL(q, p) per sequence", "% of T=1 sequences containing a token q can never emit", "% of T=1 tokens outside q", "reward of T=1 sequences inside q", "reward of T=1 sequences with such a token"])


def t_sft():
    rows = []
    order = ["sft_greedy", "sft_T0.3", "sft_T0.5", "sft_T0.7", "sft_T0.7_mp0.05", "sft_T1.0_mp0.1", "sft_T1.0_mp0.05", "sft_T1.0_p0.9", "sft_T1.0", "mbr8:sft_T0.7_n16", "mbr16:sft_T0.7_n16"]
    for n in order:
        v = SFT.get(n)
        if not v: continue
        d = (lambda k: "-" if n == "sft_greedy" else fmt(v, k))
        rows.append([f"`{n.replace('sft_', '')}`", f"{v['chrf_all']:.2f}", d("all"), d("strong"), d("medium"), d("weak"), f"{v['bleu_all']:.2f}", f"{v['finished']*100:.1f}", f"{v['loop']*100:.1f}", f"{v['rep3']*100:.1f}", f"{v['num_kept']*100:.1f}"])
    return md(rows, ["SFT-final, sentence level", "chrF++", "delta chrF++", "strong", "medium", "weak", "spBLEU", "finished %", "loop %", "repeated 3-gram %", "numbers kept %"])


def t_doc(kind):
    if not DOC or kind not in DOC: return "_(not available)_"
    return DOC[kind]


TABLES = {"heat_chrf": lambda: t_heat(), "heat_bleu": lambda: t_heat(m="bleu"), "heat_weak": lambda: t_heat("weak"), "heat_strong": lambda: t_heat("strong"), "heat_medium": lambda: t_heat("medium"),
          "group_best": t_group_best, "canon": t_canon, "hf": t_hf, "seeds": t_seeds, "rep": t_rep, "pools": t_pools, "pools_p": lambda: t_pools(("mbr",), (4, 8, 16)), "pool_groups": t_pool_groups, "oracle": t_oracle, "pool_best_group": t_pool_best_group,
          "selection": t_selection, "africomet": t_africomet, "perlang": t_perlang, "rl": t_rl, "sft": t_sft, "rl_bias": t_rl_bias}


def main():
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "reports" / "generation_config_study_2026-10-22.md"
    parts = ["template_head.md", "template_part_setup.md", "template_part_sentence.md", "template_part_doc.md", "template_part_rl.md", "template_part_tail.md"]
    text = "\n".join((D / f).read_text() for f in parts if (D / f).exists()) if len(sys.argv) <= 1 else Path(sys.argv[1]).read_text()
    for key, f in (("answer", "text_answer.md"), ("decision_table", "text_decision.md"), ("minTokens_sentence", "text_mt_sentence.md"), ("doc_minTokens", "text_mt.md"), ("sft_block", "text_sft.md")):
        text = text.replace("{{" + key + "}}", (D / f).read_text().strip() if (D / f).exists() else "_(not available)_")
    for k, f in TABLES.items():
        text = text.replace("{{" + k + "}}", f())
    for k in ("doc_core", "doc_human", "doc_units", "doc_repair", "doc_whole"):
        text = text.replace("{{" + k + "}}", t_doc(k))
    out.write_text(text)
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
