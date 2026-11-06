#!/usr/bin/env python3
"""Step 3: NLLB-mined / NLLB-derived share of the 385M's stage-B SFT data, per language (rows and Gemma tokens) and overall.

Inputs: composition_rows_by_lang_source.csv + token_sample.parquet (s01), the stage-B schedule (databox_backup/work/sched_g/B/schedule.json: packs and draws per language, 4,096-token packs,
efficiency 0.9995 from the shard log), the instruction-stream table of reports/intake2_instr_packs_2026-11-01.md (tokens per language), the Aya-aligned and Tumbuka instruction shards
(instr/), and reports/data/nllb_in_stageB_2026-11-01.csv (the earlier count, used as a cross-check of the row counts).

Definitions
  nllb_mined      source_dataset 'nllb-mined:OPUS-NLLB-v1' (OPUS NLLB v1, 43 languages) + 'extra-nllb_mined:michsethowusu/english-fulah_sentence-pairs' (ful)
  nllb_seed       'extra-nllb_seed' (tzm, human translated NLLB-Seed; not mined)
  aya_aligned     instr/aya_collection_aligned: African side of the NLLB-3.3B translated blocks of the Aya collection, in the instruction packs
  tumbuka_mt      instr/tumbuka_instruction + tumbuka_language: machine translation, system not stated, 'possibly NLLB-derived' (the report's own words)
  NLLB-derived (strict)  = nllb_mined + nllb_seed + aya_aligned ;  NLLB-derived (incl. possible) adds tumbuka_mt
Tokens: Gemma-3 tokenizer, content tokens (English + target) of the pairs x a global window-overhead factor f (prompt, tags, chat wrapper) calibrated so that the translation stream of the
languages WITHOUT instruction data and WITHOUT NLLB matches their pack tokens (printed). Instruction/Aya/Tumbuka tokens: sampled tokens per character x characters x the pack-level scale.
"""
import json, os, re
import numpy as np, pandas as pd
import duckdb
from tokenizers import Tokenizer

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, "..", "..", ".."))
ARC = os.path.join(ROOT, "research_archive_2026-11-03", "databox_backup")
TOK = os.path.join(ROOT, "gemma_mt", "artifacts", "tokenizer_gemma3_tags", "tokenizer.json")
PACK = 4096 * 0.9995

comp = pd.read_csv(os.path.join(H, "composition_rows_by_lang_source.csv"))
smp = pd.read_parquet(os.path.join(H, "token_sample.parquet"))
tk = Tokenizer.from_file(TOK)
tk.no_padding(); tk.no_truncation()
rows = []
for iso, g in smp.groupby("iso"):
    for grp, h in g.groupby("grp"):
        h = h.head(4000)
        en = tk.encode_batch(h.english.tolist(), add_special_tokens=False)
        tg = tk.encode_batch(h.target.tolist(), add_special_tokens=False)
        ntok = sum(len(a.ids) for a in en) + sum(len(a.ids) for a in tg)
        nch = int(h.english.str.len().sum() + h.target.str.len().sum())
        rows.append((iso, grp, len(h), ntok, nch))
tp = pd.DataFrame(rows, columns=["iso", "grp", "n_sample", "tok", "chars"])
tp["tpc"] = tp.tok / tp.chars
tpc_iso = {r.iso: r.tpc for r in tp.groupby("iso").apply(lambda x: pd.Series({"tpc": x.tok.sum() / x.chars.sum()}), include_groups=False).reset_index().itertuples()}
tp.to_csv(os.path.join(H, "tokens_per_char_sample.csv"), index=False)
tpc = {(r.iso, r.grp): r.tpc for r in tp.itertuples() if r.n_sample >= 200}

# ---- translation content tokens per (iso, source_group)
comp["tpc"] = [tpc.get((i, g), tpc_iso.get(i, np.nan)) for i, g in zip(comp.iso, comp.source_group)]
glob_tpc = tp.tok.sum() / tp.chars.sum()
comp["tpc"] = comp.tpc.fillna(glob_tpc)
comp["content_tokens"] = comp.chars * comp.tpc
agg = comp.pivot_table(index="iso", columns="source_group", values=["rows", "content_tokens"], aggfunc="sum", fill_value=0)
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index()
for c in ["rows_nllb_mined", "rows_nllb_seed", "rows_other", "content_tokens_nllb_mined", "content_tokens_nllb_seed", "content_tokens_other"]:
    if c not in agg: agg[c] = 0.0
agg["rows_translation"] = agg.rows_nllb_mined + agg.rows_nllb_seed + agg.rows_other
agg["content_tokens_translation"] = agg.content_tokens_nllb_mined + agg.content_tokens_nllb_seed + agg.content_tokens_other

# ---- stage B schedule
sj = json.load(open(os.path.join(ARC, "work", "sched_g", "B", "schedule.json")))["draws_per_language"]
sb = pd.DataFrame([(k, v["packs"], v["draws"]) for k, v in sj.items()], columns=["iso", "packs_B", "draws_B"])
sb["tokens_B"] = sb.packs_B * PACK
sb["tokens_B_draws"] = sb.draws_B * PACK

# ---- instruction stream (tokens per language in the packs) from the report table
txt = open(os.path.join(ROOT, "reports", "intake2_instr_packs_2026-11-01.md")).read()
ins = {}
for m in re.finditer(r"^\| ([a-z]{3}) \| ([\d,]+) \| ([\d.]+)M \|", txt, re.M):
    ins[m.group(1)] = float(m.group(3)) * 1e6
instr = pd.DataFrame(list(ins.items()), columns=["iso", "instr_tokens"])

# ---- Aya aligned and Tumbuka: tokens per language from a sample of messages
con = duckdb.connect()
def inst_tokens(d, name):
    fs = os.path.join(ARC, "instr", d, "*.parquet")
    n = con.execute(f"select iso, count(*) n, sum(n_chars) chars from read_parquet('{fs}') group by 1").df()
    out = []
    for iso in n.iso:
        s = con.execute(f"select messages from read_parquet('{fs}') where iso='{iso}' using sample 1500 rows (reservoir, 7)").df()
        ids = tk.encode_batch(s.messages.tolist(), add_special_tokens=False)
        # the messages column is JSON text: its tokens over-count braces/keys slightly; scale by chars of the JSON text
        ch = s.messages.str.len().sum()
        out.append((iso, sum(len(a.ids) for a in ids) / ch))
    t = pd.DataFrame(out, columns=["iso", "tpc_json"])
    n = n.merge(t, on="iso")
    n["est_tokens_raw"] = n.chars * n.tpc_json
    n["source"] = name
    return n
inst_parts = pd.concat([inst_tokens("aya_collection_aligned", "aya_aligned"), inst_tokens("tumbuka_instruction", "tumbuka_mt"), inst_tokens("tumbuka_language", "tumbuka_mt")])
# scale so that Aya total = 169M and Tumbuka total = 163.1M tokens (values of the pack report)
scale = {"aya_aligned": 169e6, "tumbuka_mt": 163.1e6}
for s_, v in scale.items():
    m = inst_parts.source == s_
    inst_parts.loc[m, "est_tokens"] = inst_parts.loc[m, "est_tokens_raw"] * v / inst_parts.loc[m, "est_tokens_raw"].sum()
inst_parts.to_csv(os.path.join(H, "instruction_nllb_derived_by_language.csv"), index=False)
ip = inst_parts.pivot_table(index="iso", columns="source", values="est_tokens", aggfunc="sum", fill_value=0).reset_index()

df = sb.merge(agg, on="iso", how="outer").merge(instr, on="iso", how="left").merge(ip, on="iso", how="left")
for c in ["packs_B", "draws_B", "tokens_B", "tokens_B_draws", "instr_tokens", "aya_aligned", "tumbuka_mt"]:
    df[c] = df[c].fillna(0)
for c in [c for c in df.columns if c.startswith("rows_") or c.startswith("content_tokens_")]:
    df[c] = df[c].fillna(0)

# ---- window overhead factor f: languages with no instruction and no NLLB
cal = df[(df.instr_tokens == 0) & (df.rows_nllb_mined == 0) & (df.tokens_B > 5e6) & (df.content_tokens_translation > 0)].copy()
cal["f"] = cal.tokens_B / cal.content_tokens_translation
print("calibration languages", len(cal), "f median", cal.f.median(), "IQR", cal.f.quantile([.25, .75]).tolist())
cal[["iso", "tokens_B", "content_tokens_translation", "f"]].to_csv(os.path.join(H, "overhead_calibration.csv"), index=False)
f = float(cal.f.median())
df["tr_tokens_est"] = df.content_tokens_translation * f
df["nllb_mined_tokens"] = df.content_tokens_nllb_mined * f
df["nllb_seed_tokens"] = df.content_tokens_nllb_seed * f
df["other_tr_tokens"] = df.content_tokens_other * f
df["recon_tokens"] = df.tr_tokens_est + df.instr_tokens
df["recon_ratio"] = df.recon_tokens / df.tokens_B.replace(0, np.nan)

df["nllb_strict_tokens"] = df.nllb_mined_tokens + df.nllb_seed_tokens + df.aya_aligned
df["nllb_incl_tum_tokens"] = df.nllb_strict_tokens + df.tumbuka_mt
for c, nm in [("nllb_mined_tokens", "share_nllb_mined"), ("nllb_strict_tokens", "share_nllb_strict"), ("nllb_incl_tum_tokens", "share_nllb_incl_tum")]:
    df[nm] = df[c] / df.tokens_B.replace(0, np.nan)
df["share_nllb_mined_of_translation"] = df.nllb_mined_tokens / df.tr_tokens_est.replace(0, np.nan)
df["share_nllb_mined_rows_of_translation_rows"] = df.rows_nllb_mined / df.rows_translation.replace(0, np.nan)
df["non_nllb_tokens_B"] = df.tokens_B - df.nllb_incl_tum_tokens
df["non_nllb_tokens_B_strict"] = df.tokens_B - df.nllb_strict_tokens
df["non_nllb_translation_tokens"] = df.tr_tokens_est - df.nllb_mined_tokens - df.nllb_seed_tokens

old = pd.read_csv(os.path.join(ROOT, "reports", "data", "nllb_in_stageB_2026-11-01.csv")).rename(columns={"nllb_pairs_in_stage_b": "earlier_count_nllb_pairs_in_stage_b"})
df = df.merge(old[["iso", "earlier_count_nllb_pairs_in_stage_b"]], on="iso", how="left")
df["rows_check_ratio"] = df.rows_nllb_mined / df.earlier_count_nllb_pairs_in_stage_b
df = df.sort_values("tokens_B", ascending=False)
df.to_csv(os.path.join(H, "nllb_share_by_language.csv"), index=False)

tot = df[["tokens_B", "tokens_B_draws", "nllb_mined_tokens", "nllb_seed_tokens", "aya_aligned", "tumbuka_mt", "instr_tokens", "tr_tokens_est", "rows_nllb_mined", "rows_translation"]].sum()
print(tot)
print("stage B total tokens (packs)", tot.tokens_B / 1e9)
for k in ["nllb_mined_tokens", "aya_aligned", "tumbuka_mt", "nllb_seed_tokens"]:
    print(k, tot[k] / 1e6, "M tokens", tot[k] / tot.tokens_B * 100, "% of stage B")
json.dump({k: float(v) for k, v in tot.items()} | {"f": f}, open(os.path.join(H, "nllb_share_overall_stageB.json"), "w"), indent=1)
print(df[["iso", "tokens_B", "nllb_mined_tokens", "share_nllb_mined", "share_nllb_strict", "recon_ratio", "rows_check_ratio"]].head(60).round(3).to_string())

# ---- exposure-weighted overall figures (what the optimiser actually saw) ------------------------------------------------
df["expo"] = (df.draws_B / df.packs_B.replace(0, np.nan)).fillna(0)
ex = {}
for c in ["nllb_mined_tokens", "nllb_seed_tokens", "aya_aligned", "tumbuka_mt", "tokens_B"]:
    ex[c] = float((df[c] * df.expo).sum())
STEP = 131072
# stage B as executed: 6 shards of 4,826 + 5,305 + 5,305 + 5,306 + 5,306 + 5,305 = 31,353 steps (the 28,956-step B schedule plus tail-language replay packs cut from g0, which hold no NLLB-mined rows)
a0_tok, a_tok, b_tok = 5625 * STEP, 1500 * STEP, 31353 * STEP
# stage A drew 48,000 packs from g0 (445,221 packs; Aya aligned ~41.3k packs, Tumbuka ~39.9k packs): their expected share of A's draws
aya_tum_share_A = (169e6 + 163.1e6) / PACK / 445221
tot_all = a0_tok + a_tok + b_tok
res = dict(stageB_tokens_packs=float(df.tokens_B.sum()), stageB_tokens_drawn=ex["tokens_B"],
           nllb_mined_drawn=ex["nllb_mined_tokens"], nllb_seed_drawn=ex["nllb_seed_tokens"], aya_drawn_B=ex["aya_aligned"], tumbuka_drawn_B=ex["tumbuka_mt"],
           stageA_aya_tumbuka_est=a_tok * aya_tum_share_A, tokens_A0=a0_tok, tokens_A=a_tok, tokens_B_steps=b_tok, tokens_total=tot_all)
res["share_nllb_mined_of_stageB"] = ex["nllb_mined_tokens"] / ex["tokens_B"]
res["share_nllb_strict_of_stageB"] = (ex["nllb_mined_tokens"] + ex["nllb_seed_tokens"] + ex["aya_aligned"]) / ex["tokens_B"]
res["share_nllb_incl_tum_of_stageB"] = (ex["nllb_mined_tokens"] + ex["nllb_seed_tokens"] + ex["aya_aligned"] + ex["tumbuka_mt"]) / ex["tokens_B"]
strict_all = ex["nllb_mined_tokens"] + ex["nllb_seed_tokens"] + ex["aya_aligned"] + a_tok * aya_tum_share_A * (169 / 332.1)
incl_all = strict_all + ex["tumbuka_mt"] + a_tok * aya_tum_share_A * (163.1 / 332.1)
res["share_nllb_mined_of_all_SFT"] = ex["nllb_mined_tokens"] / tot_all
res["share_nllb_strict_of_all_SFT"] = strict_all / tot_all
res["share_nllb_incl_tum_of_all_SFT"] = incl_all / tot_all
res["share_nllb_mined_of_stageB_executed"] = ex["nllb_mined_tokens"] / b_tok
res["share_nllb_strict_of_stageB_executed"] = (ex["nllb_mined_tokens"] + ex["nllb_seed_tokens"] + ex["aya_aligned"]) / b_tok
res["share_nllb_incl_tum_of_stageB_executed"] = (ex["nllb_mined_tokens"] + ex["nllb_seed_tokens"] + ex["aya_aligned"] + ex["tumbuka_mt"]) / b_tok
res["note"] = "all_SFT = A0 (5,625 steps) + stage A (1,500) + stage B (31,353 executed) x 131,072 tokens; A0 data contain no NLLB-mined / Aya / Tumbuka rows"
json.dump(res, open(os.path.join(H, "nllb_share_overall_exposure.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
