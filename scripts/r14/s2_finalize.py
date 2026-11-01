"""R14 stage 2: finalize manifest.json (teacher quality on dev + devtest, yields, hold-out stats) and write READY_TEACHER."""
import csv, json, os, glob
R = "/workspace/jobs/r14"
m = json.load(open(f"{R}/teacher_v1/manifest.json"))
dev = {r["iso"]: r for r in csv.DictReader(open(f"{R}/out/tables_dev/nllb600m_per_language_benchmark.csv")) if r["subset"] == "all"}
dec = {r["iso"]: r for r in csv.DictReader(open(f"{R}/code/nllb600m_decision_table.csv"))}
pool = json.load(open(f"{R}/s2_pool_sizes.json"))
cfg = json.load(open(f"{R}/b_full.json"))
for l in m["languages"]:
    iso = l["language"]; d = dec.get(iso) or {}
    q = dict(l.get("teacher_quality_vs_human") or {})
    if iso in dev:
        q["teacher_flores_plus_dev_chrf_pp"] = float(dev[iso]["nllb_n33_beam4_chrf"]); q["teacher_flores_plus_dev_spbleu"] = float(dev[iso]["nllb_n33_beam4_spbleu"])
    l["teacher_quality_vs_human"] = q
    st = f"{R}/s2/select_stats_{iso}.json"
    if os.path.exists(st): l["english_window_selection"] = json.load(open(st))
    if iso in ("knc", "kon") or not l.get("yield_estimate_full_pool_tokens"):
        alias = {"knc": "sag", "kon": "kon"}.get(iso, iso)
        import pyarrow.parquet as pq
        n = 0
        for p in (f"/workspace/sft_build/gemma/rl_prompts_v3/{alias}.parquet", f"/workspace/jobs/r7/assets/rl_prompts_sci_v1/{alias}.parquet"):
            if os.path.exists(p): n += int(sum(pq.read_table(p, columns=["n_sentences"]).column(0).to_pylist()))
        l["pool_sentences_available"] = n
        if l.get("candidate_tokens"): l["yield_estimate_full_pool_tokens"] = int(l["candidate_tokens"] / max(1, l["sentences_translated"]) * n)
    l["budget_tokens"] = cfg["budgets"].get(iso)
m["notes"] = {
    "teacher": "facebook/nllb-200-3.3B (CC-BY-NC-4.0) via OpenNMT/nllb-200-3.3B-ct2-int8 (CTranslate2 int8_float16), beam 4, no_repeat_ngram_size 4, sentence level; documents/chunks = sentences joined by single spaces",
    "licence_class": "nc_teacher (every row); per-row columns in rows.parquet: source_dataset, english_licence, licence_class, teacher, teacher_decode, mode, n_tokens, prompt_len, target_len, ids_sha1",
    "english_sources": "RL prompt pools /workspace/sft_build/gemma/rl_prompts_v3 + /workspace/jobs/r7/assets/rl_prompts_sci_v1 (the language's own pool; knc uses the sag pool, no pool of its own)",
    "holdout": "R4 HoldOut (exact sentence >20 chars + word 8-gram) over AfriScience dev+test, AfriLongDoc hard+core, bench_dev_clean, bench_dev (all), FLORES+ dev+devtest English, NTREX, MAFAND, SmolSent, SALT English, rl-vs-sft pool",
    "sentence_filters": "empty; copy of the English (Jaccard>0.6); 3-gram loop; char-length ratio outside p50 -/+ 2.2*(p50-p05 / p95-p50) of the human FLORES dev ratio; GlotLID top-1 not in the language's labels (sentences >= min_chars); AfriCOMET-QE < t_fail anchor (QE-validated 'qe' mode languages only); chunk-level GlotLID",
    "kon": "NLLB kon_Latn matches the Kituba-like Kikongo of the SmolSent references; the model's current Kikongo is another variety (variety decision for the owner)",
    "knc": "Kanuri has no SFT data and no RL pool: windows from the sag pool",
    "budget_rule": "equal-token budget with weak languages weighted: 2.0x base 140k tokens for student chrF++ < 25 or Ge'ez/consensus (kea amh tir knc nus kbp bam kon), 1.5x for 25-35, 1.0x above; languages that could not fill the budget keep what passed the filters",
    "format": "R4 packed format (gemma_mt.data.prepare_sft): train/ + meta.json (tokenizer fingerprint of final_1433), seq_len 8192, audited with gemma_mt.data.audit_packed (0 problems)"}
json.dump(m, open(f"{R}/teacher_v1/manifest.json", "w"), indent=1, ensure_ascii=False)
json.dump(m, open(f"{R}/manifest.json", "w"), indent=1, ensure_ascii=False)
open(f"{R}/READY_TEACHER", "w").write(f"READY_TEACHER v1: {m['total_samples']} samples, {m['total_tokens']} tokens, {len(m['languages'])} languages at {R}/teacher_v1 (R4 packed format, seq_len 8192; audit OK). manifest: {R}/manifest.json. licence_class nc_teacher (NLLB-200-3.3B, CC-BY-NC). Supersedes teacher_v1_part1.\n")
print(m["total_tokens"], len(m["languages"]))
