#!/usr/bin/env python3
"""Tables and figures of Section 3.3 (tokenisation cost by language) and Section 6.11 (running the model on a CPU).

    python paper/scripts/make_cpu.py tables      # -> paper/tables/t_tokcost_lang.md, t_cpu_*.md
    python paper/scripts/make_cpu.py figs        # -> paper/figures/fig17_token_cost.{png,pdf}, fig18_cpu_tradeoff.{png,pdf}   (needs matplotlib)

Every number is read from a file (claims ledger Part J):
    reports/data/cpu_speed_2026-10-23/{quality,quality_paragraph,headline_speed_4threads,speed,lean_rss,per_language_speed,
                                       tokens_per_sentence,data_parallel_4x1thread,solo_vs_parallel_streams}.csv
    reports/data/weak_language_data_2026-10-22/tokenizer_flores_devtest.csv       (Gemma tokenizer of final_1433, FLORES+ devtest, 1,012 sentences)
    reports/data/final_model_per_language.csv                                      (RL-final FLORES+ chrF++, subset all)
    paper/data/tokenizer_amharic_probe_2026-10-23.csv                              (one-off probe; text only)
The speed measurements were taken on a shared Xeon Gold 6126 pinned to 4 cores; see reports/cpu_speed_2026-10-23.md.
"""
import math
import os
import statistics as st
import sys

from common import *  # noqa: F401,F403

CPU = os.path.join(RDATA, "cpu_speed_2026-10-23")
WL = os.path.join(RDATA, "weak_language_data_2026-10-22")
SIX = ["swh", "hau", "kin", "yor", "amh", "tir"]
NAME = dict(swh="Swahili", hau="Hausa", kin="Kinyarwanda", yor="Yoruba", amh="Amharic", tir="Tigrinya")


def f1(x, d=1):
    return f"{float(x):.{d}f}"


def sg(x, d=2):
    return f"{float(x):+.{d}f}"


def w(name, header, rows, align=None):
    align = align or (["l"] + ["r"] * (len(header) - 1))
    sep = "|" + "|".join({"l": ":--", "r": "--:", "c": ":-:"}[a] for a in align) + "|"
    out = ["| " + " | ".join(header) + " |", sep] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("table", name, len(rows))


def rk(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(s):
        r[i] = float(k)
    return r


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return sum((p - ma) * (q - mb) for p, q in zip(a, b)) / math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))


def load():
    D = {}
    D["tok"] = {r["lang"]: r for r in read_csv(os.path.join(WL, "tokenizer_flores_devtest.csv"))}
    D["tps"] = {r["lang"]: r for r in read_csv(os.path.join(CPU, "tokens_per_sentence.csv"))}
    D["pls"] = {(r["engine"], r["lang"]): r for r in read_csv(os.path.join(CPU, "per_language_speed.csv"))}
    D["q"] = {r["variant"]: r for r in read_csv(os.path.join(CPU, "quality.csv"))}
    D["qp"] = {r["variant"]: r for r in read_csv(os.path.join(CPU, "quality_paragraph.csv"))}
    D["hs"] = read_csv(os.path.join(CPU, "headline_speed_4threads.csv"))
    D["sp"] = {r["variant"] + "|" + r["batch"]: r for r in read_csv(os.path.join(CPU, "speed.csv"))}
    D["lean"] = {r["file"]: r for r in read_csv(os.path.join(CPU, "lean_rss.csv"))}
    D["dp"] = read_csv(os.path.join(CPU, "data_parallel_4x1thread.csv"))
    D["svp"] = read_csv(os.path.join(CPU, "solo_vs_parallel_streams.csv"))
    D["rlf"] = {r["iso"]: float(r["chrf"]) for r in read_csv(os.path.join(RDATA, "final_model_per_language.csv"))
                if r["benchmark"] == "flores_plus" and r["subset"] == "all" and r["checkpoint"] == "rl_final_step1433"}
    return D


def tokcost_numbers(D):
    """Per-language numbers used by the table, the figure and the ledger."""
    out = {}
    swh_ref = float(D["tps"]["swh"]["ref_tok_mean"]); swh_gen = float(D["tps"]["swh"]["gen_tok_mean_ours"])
    swh_sps = float(D["pls"][("llama.cpp Q8_0", "swh")]["sent_per_s_best_rep"])
    for l in SIX:
        t = D["tps"][l]; p = D["pls"][("llama.cpp Q8_0", l)]
        out[l] = dict(
            cpt_flores=float(D["tok"][l]["chars_per_token"]), vs_en=float(D["tok"][l]["tokens_vs_english"]),
            cpt_subset=float(t["ref_chars_mean"]) / float(t["ref_tok_mean"]),
            ref_tok=float(t["ref_tok_mean"]), gen_tok=float(t["gen_tok_mean_ours"]), n=int(t["n"]),
            ref_rel=float(t["ref_tok_mean"]) / swh_ref, gen_rel=float(t["gen_tok_mean_ours"]) / swh_gen,
            sps=float(p["sent_per_s_best_rep"]), time_rel=swh_sps / float(p["sent_per_s_best_rep"]),
            h1m=float(p["h_per_1M_best_rep"]), chrf_gpu=float(D["q"]["vLLM bf16 GPU (greedy)"][l]), chrf_q8=float(D["q"]["q_Q8_0_b32"][l]),
            chrf_rl=D["rlf"][l])
    common = [l for l in D["tok"] if l != "eng" and l in D["rlf"]]
    x = [float(D["tok"][l]["chars_per_token"]) for l in common]; y = [D["rlf"][l] for l in common]
    out["_corr"] = dict(n=len(common), langs=common, spearman=pearson(rk(x), rk(y)))
    return out


def tables():
    D = load()
    N = tokcost_numbers(D)
    # ---- Section 3.3: tokenisation cost by language
    rows = []
    for l in SIX:
        n = N[l]
        rows.append([NAME[l], f1(n["cpt_flores"], 2), f1(n["vs_en"], 2), f1(n["ref_tok"]), f1(n["gen_tok"]), f1(n["gen_rel"], 2), f1(n["sps"], 2), f1(n["time_rel"], 2),
                     f1(n["chrf_gpu"]), f1(n["chrf_rl"])])
    w("t_tokcost_lang", ["language", "characters per token (FLORES+, 1,012 sentences)", "tokens vs English", "reference tokens per sentence", "tokens we generate per sentence",
                         "generated tokens vs Swahili", "Q8\\_0 sentences/s (batch 32)", "time per sentence vs Swahili", "chrF++ final run (speed subset)", "chrF++ RL-final (FLORES+)"], rows)

    # ---- Section 6.11: quantisation versus quality
    q = D["q"]
    order = [("vLLM bf16 GPU (greedy)", "GPU (vLLM, bf16), reference", "GPU"), ("q_fp32_b32", "CTranslate2 fp32", "CPU"), ("q_Q8_0_b32", "GGUF Q8\\_0 (llama.cpp)", "CPU"),
             ("q_Q6_K_b32", "GGUF Q6\\_K", "CPU"), ("q_hyb_b32", "CTranslate2 hybrid (attention fp32, rest int8)", "CPU"), ("q_Q5_K_M_b32", "GGUF Q5\\_K\\_M", "CPU"),
             ("q_Q4_K_M_b32", "GGUF Q4\\_K\\_M", "CPU"), ("q_int8_b32", "CTranslate2 int8", "CPU")]
    rows = []
    for k, nm, _ in order:
        r = q[k]
        if k.startswith("vLLM"):
            rows.append([nm, f1(r["macro"], 2), "", f1(r["yor"]), f1(r["amh"]), ""])
        else:
            d, ci = float(r["d_vs_gpu"]), r["d_vs_gpu_ci"].strip("[]").split(",")
            rows.append([nm, f1(r["macro"], 2), f"{sg(d)} [{sg(ci[0])}, {sg(ci[1])}]", f1(r["yor"]), f1(r["amh"]), f1(100 * float(r["exact_match_gpu"]), 1)])
    w("t_cpu_quant", ["variant", "macro chrF++", "$\\Delta$ vs GPU [95% CI]", "Yoruba", "Amharic", "outputs identical to the GPU output (%)"], rows)

    # ---- speed (headline file; solo and shared-box rows are labelled)
    rows = []
    sysname = {"ours Q8_0 llama.cpp": "Q8\\_0 (llama.cpp)", "ours CT2 int8": "CTranslate2 int8", "ours Q4_K_M llama.cpp": "Q4\\_K\\_M (llama.cpp)",
               "ours CT2 hybrid (attn fp32, ffn/emb int8)": "CTranslate2 hybrid", "ours CT2 fp32": "CTranslate2 fp32"}
    by = {}
    for r in D["hs"]:
        if r["system"] in sysname:
            by.setdefault(r["system"], {})[r["batch"]] = r
    for s in ["ours Q8_0 llama.cpp", "ours CT2 int8", "ours Q4_K_M llama.cpp", "ours CT2 hybrid (attn fp32, ffn/emb int8)", "ours CT2 fp32"]:
        b = by[s]
        solo = "solo" if b["1"]["timing_conditions"].startswith("solo") else "shared box"
        cell = lambda k: f"{float(b[k]['sent_per_s_mean_of_3']):.2f} ({float(b[k]['min']):.2f}-{float(b[k]['max']):.2f})"
        rss = lambda k: f"{float(b[k]['peak_rss_mb(incl python/torch for ct2+nllb)']) / 1000:.1f}"
        rows.append([sysname[s], solo, f1(b["1"]["macro_chrF++_on_Q(1170 sent)"], 2), cell("1"), cell("8"), cell("32"), f1(b["32"]["hours_per_1M"], 0), f1(b["32"]["decode_tok_per_s"], 0),
                     f1(b["1"]["first_token_ms_p50(b1)"], 0), f"{rss('1')} / {rss('32')}"])
    w("t_cpu_speed", ["variant", "timing", "macro chrF++", "sentences/s, batch 1 (min-max of 3)", "batch 8", "batch 32", "hours per 1 M sentences (batch 32)", "decode tokens/s (batch 32)",
                      "first token, batch 1 (ms)", "peak RAM batch 1 / 32 (GB, incl. Python for CT2)"], rows,
      ["l", "l"] + ["r"] * 8)

    # ---- thread scaling and process parallelism
    sp = D["sp"]
    g = lambda v, b: float(sp[f"{v}|{b}"]["sent_per_s"])
    rows = []
    for nm, vs, bt in [("Q8\\_0, batch 1", ("z_Q8_0_t1_U", "z_Q8_0_t2_U", "z_Q8_0_t4_U"), "1"), ("Q8\\_0, batch 8", ("z_Q8_0_t1_T8", "z_Q8_0_t2_T8", "z_Q8_0_t4_T8"), "8"),
                       ("CTranslate2 int8, batch 8", ("z_int8_t1_T", "z_int8_t2_T", "z_int8_t4_T"), "8")]:
        a = [g(v, bt) for v in vs]
        rows.append([nm, f1(a[0], 2), f1(a[1], 2), f1(a[2], 2), f"{a[1] / a[0]:.2f}x / {a[2] / a[0]:.2f}x"])
    dp = {}
    for r in D["dp"]:
        dp.setdefault(r["variant"], []).append(float(r["aggregate_sent_per_s(120 sentences / slowest process)"]))
    for k, a in dp.items():
        nm = "Q8\\_0, 4 processes x 1 thread, batch 8 each" if k.startswith("Q8") else "CTranslate2 int8, 4 processes x 1 thread, batch 8 each"
        rows.append([nm + " (aggregate)", "", "", f1(st.mean(a), 2), f"range {min(a):.2f}-{max(a):.2f}"])
    w("t_cpu_threads", ["configuration", "1 thread", "2 threads", "4 threads", "speed-up over 1 thread (2 / 4 threads)"], rows)

    # ---- NLLB reference
    ours = q["q_Q8_0_b32"]
    hs = {(r["system"], r["batch"]): r for r in D["hs"]}
    nl = [("NLLB-200-distilled-600M, greedy", "nllb600_g_Q", ("NLLB-600M CT2 int8 greedy", "32")), ("NLLB-200-distilled-600M, beam 4", "nllb600_b4_Q", ("NLLB-600M CT2 int8 beam4", "32")),
          ("NLLB-200 3.3B, greedy", "nllb3p3b_g_Q", ("NLLB-3.3B CT2 int8 greedy", "32")), ("NLLB-200 3.3B, beam 4", "nllb3p3b_b4_Q", ("NLLB-3.3B CT2 int8 beam4", "32"))]
    L4 = ["swh", "hau", "yor", "kin"]
    m4 = lambda r: st.mean(float(r[l]) for l in L4)
    rows = []
    for nm, k, h in nl:
        r = q[k]; s = hs[h]
        rows.append([nm + " (reference)", f1(r["macro"], 2), f1(m4(r), 1), f1(r["amh"]), f1(r["tir"]), f1(s["sent_per_s_mean_of_3"], 2), f1(s["hours_per_1M"], 0), f1(float(s["peak_rss_mb(incl python/torch for ct2+nllb)"]) / 1000, 1), f1(r["gen_tok_mean"], 1)])
    o = hs[("ours Q8_0 llama.cpp", "32")]
    rows.append(["ours, GGUF Q8\\_0 greedy", f1(ours["macro"], 2), f1(m4(ours), 1), f1(ours["amh"]), f1(ours["tir"]), f1(o["sent_per_s_mean_of_3"], 2), f1(o["hours_per_1M"], 0), f1(float(o["peak_rss_mb(incl python/torch for ct2+nllb)"]) / 1000, 1), f1(ours["gen_tok_mean"], 1)])
    oi = hs[("ours CT2 int8", "32")]; qi = q["q_int8_b32"]
    rows.append(["ours, CTranslate2 int8 greedy", f1(qi["macro"], 2), f1(m4(qi), 1), f1(qi["amh"]), f1(qi["tir"]), f1(oi["sent_per_s_mean_of_3"], 2), f1(oi["hours_per_1M"], 0), f1(float(oi["peak_rss_mb(incl python/torch for ct2+nllb)"]) / 1000, 1), f1(qi["gen_tok_mean"], 1)])
    w("t_cpu_nllb", ["system (same 4 cores, batch 32)", "macro chrF++ (6 languages)", "swh hau yor kin mean", "Amharic", "Tigrinya", "sentences/s", "hours per 1 M", "peak RAM (GB)", "output tokens per sentence"], rows)

    # ---- paragraphs
    qp = D["qp"]
    spp = {r["variant"]: r for r in read_csv(os.path.join(CPU, "speed.csv"))}
    rows = []
    for k, nm in [("p_fp32_b8_r1", "CTranslate2 fp32"), ("p_Q8_0_b8_r1", "GGUF Q8\\_0"), ("p_hyb_b8_r1", "CTranslate2 hybrid"), ("p_Q4_K_M_b8_r1", "GGUF Q4\\_K\\_M"), ("p_int8_b8_r1", "CTranslate2 int8"),
                  ("p_n600_b4_r1", "NLLB-200-distilled-600M, beam 4 (reference)"), ("p_n3p3_b4_r1", "NLLB-200 3.3B, beam 4 (reference)")]:
        r = qp[k]
        rows.append([nm, f1(r["chrf_ref"]), f"{int(float(r['unfinished']))} / 20", f1(r["gen_tok_mean"], 0)])
    w("t_cpu_paragraph", ["system (20 stitched SmolDoc paragraphs, 475-516 prompt tokens)", "chrF++ vs human reference", "hit the 1,024-token cap", "output tokens per paragraph"], rows)
    return D, N


# --------------------------------------------------------------------------------------------------------------- figures
def figs():
    plt = setup_matplotlib()
    D = load()
    N = tokcost_numbers(D)
    # Figure 17: tokenisation cost
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(wspace=0.32, width_ratios=[1.15, 1.0]))
    ax = axs[0]
    common = N["_corr"]["langs"]
    for l in common:
        x = float(D["tok"][l]["chars_per_token"]); y = D["rlf"][l]
        six = l in SIX
        ax.scatter([x], [y], s=26 if six else 16, color=OI["blue"] if six else OI["grey"], zorder=3, alpha=1.0 if six else 0.75)
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6.3 if six else 5.6, color=OI["blue"] if six else "#555555")
    ax.set_xlabel("characters per Gemma token (FLORES+ devtest, 1,012 sentences)")
    ax.set_ylabel("RL-final chrF++, FLORES+ (all rows)")
    ax.set_title(f"(a) Spearman {N['_corr']['spearman']:.2f} over {N['_corr']['n']} languages", fontsize=7.2)
    ax.set_ylim(5, 65)
    ax = axs[1]
    xs = np_arange(len(SIX)); wd = 0.38
    gen = [N[l]["gen_tok"] for l in SIX]; sps = [N[l]["sps"] for l in SIX]
    ax.bar([i - wd / 2 for i in xs], gen, wd, color=OI["orange"], label="tokens generated per sentence (left axis)")
    for i, v in zip(xs, gen):
        ax.text(i - wd / 2, v + 1, f"{v:.0f}", ha="center", fontsize=6)
    ax.set_ylabel("tokens generated per sentence"); ax.set_ylim(0, 100)
    ax.set_xticks(list(xs)); ax.set_xticklabels(SIX)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.bar([i + wd / 2 for i in xs], sps, wd, color=OI["green"], label="Q8_0 sentences/s, batch 32 (right axis)")
    for i, v in zip(xs, sps):
        ax2.text(i + wd / 2, v + 0.04, f"{v:.2f}", ha="center", fontsize=6)
    ax2.set_ylim(0, 4.0); ax2.set_ylabel("sentences per second"); ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6, ncol=1)
    ax.set_title("(b) output length and CPU speed, 4 threads", fontsize=7.2)
    save(fig, "fig17_token_cost")
    plt.close(fig)

    # Figure 18: quality-speed trade-off on the CPU
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    hs = {(r["system"], r["batch"]): r for r in D["hs"]}
    pts = [("ours Q8_0 llama.cpp", "Q8_0", OI["blue"], "o"), ("ours CT2 int8", "CT2 int8", OI["blue"], "s"), ("ours Q4_K_M llama.cpp", "Q4_K_M", OI["blue"], "^"),
           ("ours CT2 hybrid (attn fp32, ffn/emb int8)", "CT2 hybrid", OI["blue"], "D"), ("ours CT2 fp32", "CT2 fp32", OI["blue"], "v"),
           ("NLLB-600M CT2 int8 greedy", "NLLB-600M greedy", OI["verm"], "o"), ("NLLB-600M CT2 int8 beam4", "NLLB-600M beam 4", OI["verm"], "s"),
           ("NLLB-3.3B CT2 int8 greedy", "NLLB-3.3B greedy", OI["verm"], "^"), ("NLLB-3.3B CT2 int8 beam4", "NLLB-3.3B beam 4", OI["verm"], "D")]
    offs = {"Q8_0": (6, 4), "CT2 fp32": (-34, 6), "CT2 hybrid": (6, -9), "Q4_K_M": (6, -9), "CT2 int8": (-8, -13)}
    for k, lab, c, m in pts:
        r = hs[(k, "32")]
        solo = r["timing_conditions"].startswith("solo")
        x = float(r["sent_per_s_mean_of_3"]); y = float(r["macro_chrF++_on_Q(1170 sent)"])
        ax.scatter([x], [y], s=34, marker=m, facecolor=c if solo else "white", edgecolor=c, linewidth=1.2, zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=offs.get(lab, (5, -2)), fontsize=6)
    ax.set_xlabel("sentences per second, batch 32, 4 threads (open marker: other jobs were running)")
    ax.set_ylabel("macro chrF++, 1,170 FLORES+ sentences")
    ax.set_xlim(0, 5.0); ax.set_ylim(32, 42.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=OI["blue"], label="ours (270M)"), Line2D([], [], marker="o", ls="", color=OI["verm"], label="NLLB-200 (reference only)")], loc="upper right", fontsize=6.5)
    ax.set_title("Quality against speed on 4 CPU cores", fontsize=8)
    save(fig, "fig18_cpu_tradeoff")
    plt.close(fig)


def np_arange(n):
    return list(range(n))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tables"
    if mode == "tables":
        tables()
    elif mode == "figs":
        figs()
    else:
        tables(); figs()
