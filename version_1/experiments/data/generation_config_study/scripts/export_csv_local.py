"""Local (stdlib): CSVs of the RL, document-level, repair and SFT results from json/*.json -> export/*.csv"""
import csv, json, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); J = D + "/json"; O = D + "/export"; os.makedirs(O, exist_ok=True)
def w(name, header, rows):
    with open(f"{O}/{name}", "w", newline="") as f:
        c = csv.writer(f); c.writerow(header); c.writerows(rows)
if os.path.exists(J + "/rl_summary.json"):
    R = json.load(open(J + "/rl_summary.json"))
    rows = []
    for n, c in R["configs"].items():
        m = c.get("reward_by_mode", {})
        rows.append([n, c["n"], m.get("qe"), m.get("blend"), m.get("consensus"), c["reward_macro"], c["reward_by_bin"].get("b64"), c["reward_by_bin"].get("b256"), c["ok_rate"], c["finish_stop"], c["group_std"], c["group_range"], c["group_best"], c["group_top4"], c["flat_groups"], c["mean_ntok"]])
    w("rl_rollouts.csv", ["setting", "completions", "reward_qe_mode", "reward_blend_mode", "reward_consensus_mode", "reward_macro", "reward_b64", "reward_b256", "ok_rate", "finish_stop", "group_std", "group_range", "group_best", "group_top4", "flat_groups", "mean_tokens"], rows)
    rows = [[n, b["n"], b["tokens"], b["kl_tok"], b["kl_seq"], b["mean_logp_tok"], b.get("share_tokens_outside_kept_set"), R.get("t1_flags", {}).get(n, {}).get("share_seq_with_cut_token"), R.get("t1_flags", {}).get(n, {}).get("reward_inside"), R.get("t1_flags", {}).get(n, {}).get("reward_with_cut_token")] for n, b in R.get("bias_own", {}).items()]
    w("rl_sampler_bias.csv", ["sampler_q", "sequences", "tokens", "kl_q_p_per_token", "kl_q_p_per_sequence", "mean_logp_p_per_token_of_own_samples", "share_tokens_outside_kept_set", "t1_share_sequences_with_cut_token", "t1_reward_inside", "t1_reward_with_cut_token"], rows)
if os.path.exists(J + "/doc_whole.json"):
    for kind, fn in (("core", "doc_whole_core.csv"), ("human", "doc_whole_human.csv")):
        T = json.load(open(J + "/doc_whole.json")); rows = []
        for b, t in T[kind].items():
            for n, c in t.items():
                if n.startswith("_"): continue
                fc = T["fail_core"][n].get(b) or T["fail_core"][n].get(str(b)) if kind == "core" else None
                rows.append([b, n, c["chrf"], c.get("chrf_lo"), c.get("chrf_hi"), c["d_chrf_vs_base"], c["d_lo"], c["d_hi"], c.get("bleu")] + ([fc["finished"], fc["loop"], fc["cov_sents"], fc["length_ratio_med"]] if fc else ["", "", "", ""]))
        w(fn, ["bin", "config", "chrf", "lo", "hi", "d_vs_greedy", "d_lo", "d_hi", "bleu", "finished", "loop", "sentence_coverage", "length_ratio"], rows)
if os.path.exists(J + "/unit_repair.json"):
    U = json.load(open(J + "/unit_repair.json"))
    w("doc_unit_repair.csv", ["repair", "units", "finished", "loop_or_unfinished", "chrf_vs_nllb", "len_ratio", "any_candidate_finished", "mean_candidate_finished"], [[k, v["n"], v["finished"], v["loop_or_unfinished"], v["chrf_vs_nllb"], v["len_ratio"], v.get("any_candidate_finished"), v.get("mean_cand_finished")] for k, v in U.items()])
if os.path.exists(J + "/doc_repair.json"):
    T = json.load(open(J + "/doc_repair.json")); rows = []
    for b, t in T["core"].items():
        for n, c in t.items():
            if not n.startswith("_"): rows.append([b, n, c["chrf"], c["d_chrf_vs_base"]])
    w("doc_repair_variants_core.csv", ["bin", "config", "chrf", "d_vs_R6_sentfix"], rows)
if os.path.exists(J + "/sft.json"):
    S = json.load(open(J + "/sft.json"))
    w("sft_sentence_level.csv", ["config", "chrf", "d_chrf", "lo", "hi", "d_strong", "d_medium", "d_weak", "spbleu", "finished", "loop", "rep3", "num_kept"], [[n, v["chrf_all"], v["d_chrf_all"], v["lo_chrf_all"], v["hi_chrf_all"], v["d_chrf_strong"], v["d_chrf_medium"], v["d_chrf_weak"], v["bleu_all"], v["finished"], v["loop"], v["rep3"], v["num_kept"]] for n, v in S.items()])
if os.path.exists(J + "/seeds.json"):
    S = json.load(open(J + "/seeds.json"))
    w("seed_variance.csv", ["config", "chrf_seeds", "mean", "sd", "mean_delta_vs_greedy"], [[n, " ".join(map(str, v["seeds_chrf"])), v["mean"], v["sd"], v["mean_d"]] for n, v in S.items()])
print(sorted(os.listdir(O)))
