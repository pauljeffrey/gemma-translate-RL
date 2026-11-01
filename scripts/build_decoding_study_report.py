#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render reports/final_model_decoding_study_2026-10-21.md from reports/data/final_model_decoding_study/{results,qe_results,extra}.json (written by the box
scripts study_*.py kept next to them). Stdlib only."""
from __future__ import annotations
import os

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "reports" / "data" / "final_model_decoding_study"
R, Q, X = (json.load(open(D / f)) for f in ("results.json", "qe_results.json", "extra.json"))
C, B, isos, GM = R["configs"], R["bootstrap"], R["isos"], R["groups_members"]

ORDER = ["greedy", "hf_greedy", "rep1.05", "rep1.1", "rep1.05out", "rep1.1out", "hf_rep1.1", "nr3", "nr4", "T0.3", "T0.5", "T0.7", "beam2", "beam4", "beam8",
         "mbr8_T0.7", "mbr9g_T0.7", "mbr8_T0.5", "mbr9g_T0.5"]
LABEL = {
    "greedy": "greedy (vLLM) = reference", "hf_greedy": "greedy (HF) cross-check",
    "rep1.05": "greedy + repetition_penalty 1.05 (vLLM, prompt + output)", "rep1.1": "greedy + repetition_penalty 1.1 (vLLM, prompt + output)",
    "rep1.05out": "greedy + repetition penalty 1.05, generated tokens only (HF)", "rep1.1out": "greedy + repetition penalty 1.1, generated tokens only (HF)",
    "hf_rep1.1": "greedy + repetition_penalty 1.1 (HF, prompt + output)",
    "nr3": "greedy + no_repeat_ngram_size 3 (HF)", "nr4": "greedy + no_repeat_ngram_size 4 (HF)",
    "T0.3": "sampling T 0.3 (vLLM, 1 sample)", "T0.5": "sampling T 0.5 (vLLM, 1 sample)", "T0.7": "sampling T 0.7 (vLLM, 1 sample)",
    "beam2": "beam search 2 (HF)", "beam4": "beam search 4 (HF)", "beam8": "beam search 8 (HF)",
    "mbr8_T0.7": "MBR: 8 samples T 0.7, pick the candidate with the highest mean chrF++ to the other 7", "mbr9g_T0.7": "MBR: same pool + the greedy output (9 candidates)",
    "mbr8_T0.5": "MBR: 8 samples T 0.5", "mbr9g_T0.5": "MBR: 8 samples T 0.5 + greedy",
}


def macro(cfg, key, members=isos):
    return statistics.fmean(C[cfg]["per_lang"][i][key] for i in members)


def ratio(cfg, num, tot, members=isos):
    n = sum(C[cfg]["per_lang"][i][num] for i in members)
    t = sum(C[cfg]["per_lang"][i][tot] for i in members)
    return n / t if t else None


def lid(cfg):
    v = [C[cfg]["per_lang"][i]["lid"] for i in isos if C[cfg]["per_lang"][i]["lid"] is not None]
    return statistics.fmean(v)


def bt(m, g, cfg):
    return B[m][g][cfg]


def dtxt(m, g, cfg):
    b = bt(m, g, cfg)
    if cfg == "greedy":
        return "0"
    star = "*" if (b["d_lo"] > 0 or b["d_hi"] < 0) else ""
    return f"{b['d_vs_greedy']:+.2f} [{b['d_lo']:+.2f}, {b['d_hi']:+.2f}]{star}"


def rel_cost(cfg):
    """seconds relative to greedy on the same engine (HF configs vs HF greedy, vLLM + MBR vs vLLM greedy)"""
    s = C[cfg]["seconds"]
    if C[cfg]["engine"] == "hf":
        return s / C["hf_greedy"]["seconds"]
    return s / C["greedy"]["seconds"]


out = []
P = out.append
P("# Sentence-level decoding study for the final RL model (2026-10-21)\n")
P(("Task D-49 (3). Which decoding rule should the released `" + os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c` use for English -> African sentence translation? ") +
  "20 languages x 200 sentences x 19 configurations, bootstrap CIs, plus AfriCOMET on the validated languages. Data and scripts: `reports/data/final_model_decoding_study/` "
  "(`results.json`, `qe_results.json`, `extra.json`, `study_*.py` = the box scripts in the order prep -> vLLM -> HF -> score -> QE -> extra); renderer `gemma_mt/scripts/build_decoding_study_report.py`. "
  "Companion: `reports/final_model_per_language_2026-10-21.md`.\n")

g, b4, b8, m5, m7 = "greedy", "beam4", "beam8", "mbr8_T0.5", "mbr9g_T0.7"
P("## 0. Result and recommendation\n")
P(f"* **Beam search wins on every aggregate**: macro chrF++ over the 20 languages greedy **{macro(g, 'chrf'):.2f}** -> beam 4 **{macro(b4, 'chrf'):.2f}** ({dtxt('chrf', 'all', b4)}) -> beam 8 **{macro(b8, 'chrf'):.2f}** ({dtxt('chrf', 'all', b8)}); "
  f"spBLEU {macro(g, 'bleu'):.2f} -> {macro(b4, 'bleu'):.2f} -> {macro(b8, 'bleu'):.2f}; AfriCOMET (reference) on the 18 validated languages "
  f"{Q['configs'][g]['ref']['macro']:.4f} -> {Q['configs'][b4]['ref']['macro']:.4f} ({Q['configs'][b4]['ref']['d']:+.4f}), QE {Q['configs'][g]['qe']['macro']:.4f} -> {Q['configs'][b4]['qe']['macro']:.4f}. Brackets: 95% paired bootstrap of the delta to greedy (1,000 resamples of the sentences of every language, equal language weight).")
P(f"* Beam 8 is only {macro(b8, 'chrf') - macro(b4, 'chrf'):.2f} chrF++ better than beam 4 and costs {rel_cost(b8) / rel_cost(b4):.1f}x more time per sentence (HF generate): **beam 4 is the knee**. Beam 2 gives {dtxt('chrf', 'all', 'beam2')} at {rel_cost('beam2'):.1f}x the HF greedy time.")
hy = X["hybrid"]["beam4+greedy_fallback"]["vs_greedy"]["chrf_all"]
P(f"* **Beam search loops where greedy does not**: loop rate (unfinished, repetitive or empty after decoding) greedy {100 * macro(g, 'loop'):.1f}% -> beam 4 {100 * macro(b4, 'loop'):.1f}% -> beam 8 {100 * macro(b8, 'loop'):.1f}%, concentrated in Nuer (`nus`: greedy 0.5%, beam 4 8%, beam 8 11.5%) and Kabiye (`kbp` 0% -> 2.5% -> 4%); "
  f"this is why beam 8 loses to greedy on `nus` (chrF++ {C[g]['per_lang']['nus']['chrf']:.1f} -> {C[b8]['per_lang']['nus']['chrf']:.1f}). A **greedy fallback** (keep the beam-4 output unless it is unfinished / repetitive / empty by `rl.guards`, then use the greedy one; {X['hybrid']['beam4+greedy_fallback']['n_fallback']} of 3,970 sentences fall back) "
  f"gives chrF++ {hy['point']:.2f} ({hy['d_vs_greedy']:+.2f} [{hy['lo']:+.2f}, {hy['hi']:+.2f}]), better than plain beam 4 in the weak group (+0.04 [+0.02, +0.06] against plain beam 4).")
P(f"* **MBR / consensus** (8 samples at T 0.7 or 0.5, pick the candidate with the highest mean chrF++ against the others) gives +0.5 chrF++ ({macro('mbr8_T0.7', 'chrf'):.2f} at T 0.7, {macro('mbr8_T0.5', 'chrf'):.2f} at T 0.5), "
  f"but **no BLEU gain** ({dtxt('bleu', 'all', 'mbr8_T0.5')}): MBR on chrF++ selects what chrF++ likes. It costs about 10-15x the vLLM greedy time as measured (8 samples + {C['mbr8_T0.7']['kw']['mbr_cpu_seconds_4procs']:.0f} s of CPU chrF++ on 4 processes for 3,970 sentences) and is the best config in the weak group by a hair (chrF++ weak {bt('chrf', 'weak', 'mbr8_T0.7')['point']:.2f} vs beam 4 {bt('chrf', 'weak', 'beam4')['point']:.2f}, intervals overlap).")
P(f"* **Things that hurt**: repetition_penalty 1.1 ({dtxt('chrf', 'all', 'rep1.1')} chrF++, spBLEU {dtxt('bleu', 'all', 'rep1.1')}) and `no_repeat_ngram_size` 3 ({dtxt('chrf', 'all', 'nr3')}) and 4 ({dtxt('chrf', 'all', 'nr4')}) are clearly worse than greedy; "
  f"temperature sampling is monotonically worse (T 0.3 {dtxt('chrf', 'all', 'T0.3')}, 0.5 {dtxt('chrf', 'all', 'T0.5')}, 0.7 {dtxt('chrf', 'all', 'T0.7')}). "
  f"Repetition penalty 1.05 is neutral on chrF++ ({dtxt('chrf', 'all', 'rep1.05')}) but lowers spBLEU ({dtxt('bleu', 'all', 'rep1.05')}) and number recall (below), and the model does not loop under greedy (0.1%), so there is nothing for it to fix.")
P(f"* **Copied numbers and names** (section 3): `no_repeat_ngram_size 3` keeps only {100 * ratio('nr3', 'num_kept', 'num_tot'):.0f}% of the source numbers that the human reference keeps (greedy {100 * ratio(g, 'num_kept', 'num_tot'):.0f}%) because a number copied twice in a sentence is a repeated n-gram; "
  f"repetition_penalty 1.1 over prompt + output keeps {100 * ratio('rep1.1', 'num_kept', 'num_tot'):.0f}% of the numbers and {100 * ratio('rep1.1', 'name_kept', 'name_tot'):.0f}% of the names (greedy {100 * ratio(g, 'name_kept', 'name_tot'):.0f}%), "
  f"because the source tokens are in the prompt and are penalised; restricting the penalty to generated tokens repairs most of it ({100 * ratio('rep1.1out', 'num_kept', 'num_tot'):.0f}% numbers) but still costs {-float(bt('chrf', 'all', 'rep1.1out')['d_vs_greedy']):.2f} chrF++.")
P("")
P("**Recommended default (quality): beam search with 4 beams, with a greedy fallback when the beam output is unfinished, repetitive or empty.** "
  f"Expected gain over greedy on sentences: about +0.7 chrF++ and +0.7 spBLEU macro, +0.011 AfriCOMET; the gain is positive in every group (strong {bt('chrf', 'strong', b4)['d_vs_greedy']:+.2f}, medium {bt('chrf', 'medium', b4)['d_vs_greedy']:+.2f}, weak {bt('chrf', 'weak', b4)['d_vs_greedy']:+.2f}) and largest in the middle of the quality range (South Ndebele, Shona, Bemba, Igbo, Amharic).")
P("**Recommended fast config: plain greedy, no penalty, no n-gram blocking** (vLLM, ~200 sentences/s on the shared RTX PRO 6000 at 0.08 GPU share, 12k tokens/s). "
  "If a consensus output is wanted (for example for pseudo-label generation), MBR over 8 samples at T 0.5 (+0.5 chrF++, no spBLEU gain) is the alternative; for plain translation serving it is dominated by beam 4.\n")

P("## 1. Setup\n")
P(f"* Model: RL-final `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c` `latest`, sha `{R['model_sha']}` (GRPO step 1433). Prompt `Translate the text to {{language}}:  {{english}}` in the chat-less format of `gemma_mt.format.build_prompt`; the language tag at the start of the completion is stripped before scoring.")
P("* Languages (20, spread over the quality range): " + ", ".join(f"`{i}`" for i in isos) + ". Test sentences: FLORES+ devtest for 17 languages and SmolSent for `ach`, `nbl`, `efi` (no FLORES+ code). "
  "**200 sentences per language, drawn at random (fixed seed) from the rows whose reference pair is not in the SFT text (`no_pair`)**; Hausa has only 170 such rows, so n = 3,970. The same sentences for every configuration (paired).")
P("* Quality groups (fixed assignment in `groups.json`, made before the final-model runs): **strong** = swh hau zul kin som (greedy chrF++ here 40.5-57.8), **medium** = ibo sna lug bem nbl (34.3-38.8), **weak** = yor amh wol tir kbp nus kab fon ach efi (13.4-29.3). "
  "Group numbers are macro means over the languages of the group.")
P("* Budget `max_new_tokens` 320 for every configuration. Engines: vLLM (greedy, sampling, repetition_penalty, 8-sample pools; one seed per sentence) and HF `generate` (beam, no_repeat_ngram, generated-token-only penalty, HF greedy cross-check), both bf16 on the same GPU slice. "
  f"vLLM greedy and HF greedy agree: chrF++ {macro(g, 'chrf'):.2f} vs {macro('hf_greedy', 'chrf'):.2f} (paired delta {dtxt('chrf', 'all', 'hf_greedy')}), so the two engines can be mixed in one table. "
  "HF beam search: default `length_penalty` 1.0, `early_stopping` False, batches of 80 / 40 / 20 sentences for 2 / 4 / 8 beams, left-padded.")
P("* Metrics per (configuration, language): chrF++ (word order 2) and spBLEU (sacrebleu `flores200` tokenizer), NFC; macro over languages; bootstrap = 1,000 paired resamples of the sentences within each language, languages weighted equally; `p_better` in the JSON = share of resamples where the config beats greedy. "
  "`finished` = stop token inside 320 tokens. `loop` = not finished, or `rl.guards.is_repetitive` / `is_self_repetitive`, or empty. `LID` = GlotLID right-language (sibling-tolerant) share. "
  "**Source-anchor recall** = of the numbers and capitalised names in the English sentence that the human reference also contains, the share the hypothesis contains (numbers compared with and without thousands/decimal separators, names case-folded). "
  f"The ceilings: of {R['raw_anchor_totals']['num']} source numbers {R['ref_kept_anchor_totals']['num']} survive into the references, of {R['raw_anchor_totals']['name']} capitalised tokens {R['ref_kept_anchor_totals']['name']} do (the rest are sentence-initial words, titles or translated names); recall is computed over the surviving ones only.")
P("* Tokens/s: generated tokens (up to and including the stop token) over wall-clock of the whole run; **vLLM and HF numbers are not comparable with each other** (continuous batching vs static padded batches); use the `x HF greedy` / `x vLLM greedy` column for the cost of a rule on the same engine.\n")

P("## 2. All configurations\n")
P("Macro over 20 languages. Delta = config minus greedy with the 95% paired-bootstrap interval, `*` = interval excludes 0.\n")
P("| config | chrF++ | delta chrF++ [95% CI] | spBLEU | delta spBLEU [95% CI] | AfriCOMET (18 langs) | finished % | loop % | LID % | numbers kept % | names kept % | tokens/s | cost vs same-engine greedy |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in ORDER:
    qe = Q["configs"][c]["ref"]["macro"]
    P(f"| {LABEL[c]} | {macro(c, 'chrf'):.2f} | {dtxt('chrf', 'all', c)} | {macro(c, 'bleu'):.2f} | {dtxt('bleu', 'all', c)} | {qe:.4f} | {100 * macro(c, 'finished'):.1f} | {100 * macro(c, 'loop'):.1f} | {100 * lid(c):.1f} | "
      f"{100 * ratio(c, 'num_kept', 'num_tot'):.1f} | {100 * ratio(c, 'name_kept', 'name_tot'):.1f} | {C[c]['tok_per_s']:.0f} ({C[c]['engine']}) | {rel_cost(c):.1f}x |")
P("")
P("AfriCOMET = `masakhane/africomet-stl-1.1` with reference, 18 of the 20 languages (the AfriCOMET-validated ones; not `ach`, `nbl`); the QE (no reference) macro and the paired CIs are in `qe_results.json`. "
  "The MBR rows' cost includes 8 samples (vLLM) and the CPU chrF++ matrix.\n")

P("## 3. Repetition penalty and n-gram blocking against copied names and numbers\n")
P("The prompt contains the English sentence; names and numbers are copied into the translation. HF's and vLLM's `repetition_penalty` divide the logit of every token already present in the *prompt or output*, so it penalises copying the source. `no_repeat_ngram_size` forbids any repeated n-gram, including a repeated number or name.\n")
P("| config | numbers kept % | names kept % | chrF++ | spBLEU | length ratio hyp/ref | tokens per sentence |")
P("|---|---|---|---|---|---|---|")
for c in ["greedy", "rep1.05", "rep1.05out", "rep1.1", "rep1.1out", "nr4", "nr3"]:
    P(f"| {LABEL[c]} | {100 * ratio(c, 'num_kept', 'num_tot'):.1f} | {100 * ratio(c, 'name_kept', 'name_tot'):.1f} | {macro(c, 'chrf'):.2f} | {macro(c, 'bleu'):.2f} | {macro(c, 'len_ratio'):.3f} | {macro(c, 'mean_tokens'):.1f} |")
P("")
P("Reading: (1) the effect on copied tokens is large and in the expected direction, the numbers are hit hardest (`nr3` 20%, `nr4` 49%, `rep1.1` 78%); (2) the generated-token-only penalty (`...out`) keeps the numbers and names but is still worse than greedy on chrF++, so the loss is not only from copied tokens "
  "(with the prompt+output penalty at 1.1 the hypotheses are about 10% longer relative to the reference: ratio 1.18 against 1.08 for greedy); (3) the greedy output of this model does not loop (0.1%), so the fixes have no failure to repair on this set. They remain relevant for hard cases (long documents, unseen languages), which this sentence study does not test.\n")

P("## 4. Winner per quality group\n")
for m, lab in (("chrf", "chrF++"), ("bleu", "spBLEU")):
    P(f"### {lab} (macro; delta to greedy with 95% CI)\n")
    P("| config | all (20) | strong (5) | medium (5) | weak (10) |")
    P("|---|---|---|---|---|")
    for c in ["greedy", "beam2", "beam4", "beam8", "mbr8_T0.7", "mbr9g_T0.7", "mbr8_T0.5", "rep1.05", "T0.3", "nr4"]:
        cells = []
        for grp in ("all", "strong", "medium", "weak"):
            b = bt(m, grp, c)
            cells.append(f"{b['point']:.2f} ({dtxt(m, grp, c) if c != 'greedy' else '-'})")
        P(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
    ranks = {}
    for grp in ("all", "strong", "medium", "weak"):
        best = max((c for c in B[m][grp]), key=lambda c: B[m][grp][c]["point"])
        ranks[grp] = best
    P("\nBest point estimate per group: " + ", ".join(f"{grp}: `{ranks[grp]}` ({B[m][grp][ranks[grp]]['point']:.2f})" for grp in ranks) + ".\n")
P("Beam 8 and beam 4 win the strong and medium groups; in the weak group the four best (MBR at T 0.7 23.81, MBR at T 0.5 23.78, MBR + greedy 23.75, beam 4 23.73, beam 4 with fallback 23.77) are inside each other's intervals, beam 8 loses ground (loops) and plain greedy is 0.6 behind. "
  "On spBLEU the beams gain +0.3 to +1.2 in every group and **MBR gains nothing** in any group.\n")

hyb = X["hybrid"]
P("### Beam search with greedy fallback\n")
P("| config | sentences falling back | chrF++ all | delta vs greedy [CI] | delta vs the plain beam [CI] | chrF++ strong / medium / weak (delta vs greedy) |")
P("|---|---|---|---|---|---|")
for name, plain in (("beam2+greedy_fallback", "beam2"), ("beam4+greedy_fallback", "beam4"), ("beam8+greedy_fallback", "beam8")):
    h = hyb[name]
    a, p = h["vs_greedy"], h["vs_" + plain]
    P(f"| {plain} + greedy fallback | {h['n_fallback']} | {a['chrf_all']['point']:.2f} | {a['chrf_all']['d_vs_greedy']:+.2f} [{a['chrf_all']['lo']:+.2f}, {a['chrf_all']['hi']:+.2f}] | "
      f"{p['chrf_all']['d_vs_' + plain]:+.2f} [{p['chrf_all']['lo']:+.2f}, {p['chrf_all']['hi']:+.2f}] | "
      + " / ".join(f"{a['chrf_' + g_]['point']:.1f} ({a['chrf_' + g_]['d_vs_greedy']:+.2f})" for g_ in ("strong", "medium", "weak")) + " |")
P("\nThe fallback costs one extra greedy decode for 0.6-1.0% of the sentences. It adds +0.02 chrF++ over plain beam 4 (CI [+0.01, +0.03]): small, but it removes the visible failures (a looping Nuer or Kabiye sentence).\n")

P("## 5. Per language\n")
P("chrF++ per language (200 sentences each; per-language intervals are not computed, see the group tables for intervals). Loop rates in the last column.\n")
P("| iso | group | greedy | beam 2 | beam 4 | beam 8 | MBR8 T0.5 | MBR9+g T0.7 | rep 1.05 | T 0.3 | nr4 | loop % greedy / beam 4 / beam 8 |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for grp in ("strong", "medium", "weak"):
    for i in [x for x in isos if R["groups"][x] == grp]:
        v = lambda c: C[c]["per_lang"][i]["chrf"]
        lp = X["loop_by_lang"]
        P(f"| `{i}` | {grp} | {v('greedy'):.1f} | {v('beam2'):.1f} | {v('beam4'):.1f} | {v('beam8'):.1f} | {v('mbr8_T0.5'):.1f} | {v('mbr9g_T0.7'):.1f} | {v('rep1.05'):.1f} | {v('T0.3'):.1f} | {v('nr4'):.1f} | "
          f"{100 * lp['greedy'][i]:.1f} / {100 * lp['beam4'][i]:.1f} / {100 * lp['beam8'][i]:.1f} |")
P("")
P("The beams help 19 of 20 languages at 4 beams (largest: South Ndebele +1.4, Amharic +1.3, Bemba +1.0, Zulu +1.0, Shona +1.0); the exception is Nuer (`nus`: -1.0 at 4 beams, -1.2 at 8) where beam search loops and MBR is better (+0.3). "
  "Acholi and Efik (SmolSent) behave like the FLORES+ weak languages.\n")

P("## 6. AfriCOMET\n")
P(f"AfriCOMET with reference and AfriCOMET-QE on the {len(Q['isos'])} validated languages ({Q['n']} sentences), macro, paired 95% CI of the delta to greedy:\n")
P("| config | AfriCOMET | delta [CI] | AfriCOMET-QE | delta [CI] |")
P("|---|---|---|---|---|")
for c in ORDER:
    r, q = Q["configs"][c]["ref"], Q["configs"][c]["qe"]
    if c == "greedy":
        P(f"| {LABEL[c]} | {r['macro']:.4f} | - | {q['macro']:.4f} | - |")
    else:
        P(f"| {LABEL[c]} | {r['macro']:.4f} | {r['d']:+.4f} [{r['d_lo']:+.4f}, {r['d_hi']:+.4f}] | {q['macro']:.4f} | {q['d']:+.4f} [{q['d_lo']:+.4f}, {q['d_hi']:+.4f}] |")
P("\nThe neural metrics agree with chrF++ that beam 8 ~ beam 4 are best (+0.011 AfriCOMET), that greedy ~ T 0.3 and that T 0.7, `no_repeat_ngram_size` and repetition_penalty 1.1 are worse. "
  "They disagree on two configurations: repetition_penalty 1.05 gains +0.009 AfriCOMET (+0.006 QE), more than MBR (+0.006), although it is neutral on chrF++, lowers spBLEU by 0.5 and drops the number recall from 95.7% to 89.8%; "
  "AfriCOMET does not seem to punish a changed or dropped number. We do not adopt it: the metric that disagrees seems insensitive to exactly the error the penalty introduces. MBR (T 0.5) gains +0.0055 AfriCOMET, half of beam 4; on chrF++ it gains 80% of beam 4's gain, as expected from a selection by chrF++.\n")

P("## 7. Cost\n")
P("| config | engine | seconds for 3,970 sentences | sentences/s | tokens/s | cost vs same-engine greedy |")
P("|---|---|---|---|---|---|")
for c in ["greedy", "hf_greedy", "rep1.05", "nr3", "nr4", "T0.5", "beam2", "beam4", "beam8", "mbr8_T0.7", "mbr9g_T0.7", "mbr8_T0.5"]:
    P(f"| {LABEL[c]} | {C[c]['engine']} | {C[c]['seconds']:.0f} | {C[c]['sent_per_s']:.1f} | {C[c]['tok_per_s']:.0f} | {rel_cost(c):.1f}x |")
P("\nThe box GPU is shared with other jobs (this run had <= 0.1 of its memory and a fraction of its compute), absolute speeds are indicative; the ratios between rows of the same engine are the useful part. "
  "Beam search was timed in HF `generate`, which is 7x slower than vLLM greedy at equal work; a serving stack with an efficient beam implementation would narrow the gap (not measured here).\n")

P("## 8. Caveats\n")
P("* 200 sentences per language, equal language weight: the intervals measure sentence-sampling noise only; per-language differences below ~0.5 chrF++ are not resolved. The three SmolSent languages use a different domain from the 17 FLORES+ ones.")
P("* Sampling configurations use one sample per sentence (seed per sentence), which is what a user would get; the average over several seeds would be a little less noisy but not different in expectation.")
P("* MBR utility = chrF++ (the metric also used for scoring). Optimising chrF++ to select among candidates is expected to favour chrF++ over BLEU or neural metrics; the AfriCOMET column is the check.")
P("* Sentence level only. Long documents (SmolDoc, LongDoc) have a different failure profile (truncation, repetition across many sentences); the conclusions about repetition penalty and n-gram blocking do not transfer to them without a separate test.")
P("* The conclusion is for the RL-final model. Beam search was not tried on the SFT model, which may benefit differently.")
(ROOT / "reports" / "final_model_decoding_study_2026-10-21.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("ok")
