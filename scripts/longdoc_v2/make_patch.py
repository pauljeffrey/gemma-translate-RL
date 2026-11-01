# SPDX-License-Identifier: AGPL-3.0-or-later
"""Builds patches/longdoc_v2_D60_{reward,chunk_plan}.patch: the D-60 hunks ONLY, as unified diffs against a pristine copy of the two files (other agents edit reward.py too).

  python make_patch.py --orig <pristine gemma_mt tree> --cur <current tree> --out <patches dir>

reward.py: the D-60 edits are re-applied to the pristine text (each anchor must occur exactly once), so the patch contains nothing else. chunk_plan.py: plain diff."""
import argparse, difflib, os, sys


def sub(s, old, new):
    assert s.count(old) == 1, f"anchor not unique / missing: {old[:70]!r} ({s.count(old)})"
    return s.replace(old, new)


def apply_reward(s, cur):
    """cur = the current reward.py (source of the D-60 text blocks, so the patch always equals what the tree really contains)."""
    def block(start, end):
        i = cur.index(start)
        j = cur.index(end, i) if end else len(cur)
        return cur[i:j]
    s = sub(s, "from gemma_mt.rl import guards\n", "from gemma_mt.rl import guards\nfrom gemma_mt.rl import longdoc\n")
    cfg = block("\n    # D-60: REWARD V2 FOR LONG DOCUMENTS", "\n\n@dataclass\nclass Sample")
    anchor = '    anchors_related_min: int = 1                       # prompts whose English sources share this many word 6-grams with the current one do not count as "other"\n'
    if anchor not in s:
        raise SystemExit("config anchor missing")
    s = sub(s, anchor, anchor + cfg.lstrip("\n").join(["\n", ""]) if False else anchor + cfg + "\n")
    # RewardOutput fields
    f1 = [l for l in cur.split("\n") if l.startswith("    chunk_rewards: List[")][0] + "\n"
    f2 = [l for l in cur.split("\n") if l.startswith("    longdoc: List[Optional")][0] + "\n"
    a = "    anchors_penalty: List[float] = field(default_factory=list)"
    i = s.index(a); j = s.index("\n", i) + 1
    s = s[:j] + f1 + f2 + s[j:]
    # __init__
    init = block("        self.longdoc_cal = None\n", "        self.qe_ens = None\n")
    s = sub(s, "        self.qe_ens = None\n        if cfg.qe_ensemble:", init + "        self.qe_ens = None\n        if cfg.qe_ensemble:")
    # per-sample bookkeeping
    bk = block("        long_i = [self.longdoc_cal", "        for i, s in enumerate(samples):\n            iso = langs[i]")
    s = sub(s, "        srcs = [nfc(s.source) if cfg.nfc else s.source for s in samples]       # D-29: the source is NFC too (every length / lexical rule counts characters)\n",
            "        srcs = [nfc(s.source) if cfg.nfc else s.source for s in samples]       # D-29: the source is NFC too (every length / lexical rule counts characters)\n" + bk)
    # H1 repetition checks
    old = '''            if cfg.repetition_guard and guards.is_repetitive(text):
                status[i] = "degenerate"
                continue
            if cfg.selfrep_guard != "off" and (cfg.selfrep_guard == "all" or modes[i][0] != "qe") and guards.is_self_repetitive(text):
                status[i] = "degenerate"                       # D-38: near-loops would agree with each other in the consensus pool
                continue
'''
    new = block("            if long_i[i] and cfg.longdoc_repetition:", "            if self.anchors is not None and cfg.anchors_stuff_gate") if "anchors_stuff_gate" in cur else block("            if long_i[i] and cfg.longdoc_repetition:", "            lo_ratio = cfg.min_len_ratio")
    s = sub(s, old, new)
    s = sub(s, "            okset = set(gated)\n", "            okset = {i for i in gated if rep_sev[i] < cfg.longdoc_pool_severity}       # D-60: near-loops stay out of the pool (rep_sev is 0 unless longdoc_v2)\n")
    s = sub(s, "            cp = plan_chunks(srcs[i], texts[i], chunking=cfg.chunking, chunk_align=cfg.chunk_align,",
            block('            if long_i[i] and cfg.chunking == "sentences" and cfg.longdoc_align', "            cp = plan_chunks(srcs[i]") + "            cp = plan_chunks(srcs[i], texts[i], chunking=cfg.chunking, chunk_align=cfg.chunk_align,")
    s = sub(s, "                        if cfg.length_penalty > 0:\n                            ex = self.shaper.length_excess(",
            block("                        if long_i[i] and cfg.longdoc_repetition and longdoc.has_stutter", "                        if cfg.length_penalty > 0:") + "                        if cfg.length_penalty > 0:\n                            ex = self.shaper.length_excess(")
    s = sub(s, "                rewards.append(aggregate(chunk_plan[i], seg_reward, self.penalty))\n", block("                if long_i[i]:                                                           # D-60", "                qe[i] = sum(raw)"))
    s = sub(s, "        rewards, weights = [], []\n        n_chunks =", "        rewards, weights = [], []\n" + block("        ld_info: List", "        n_chunks =") + "        n_chunks =")
    s = sub(s, "        out.science_deficit = sci_def\n", "        out.science_deficit = sci_def\n" + block("        if cfg.keep_chunk_rewards:\n            out.chunk_rewards", "        if anc is not None:"))
    s = sub(s, "    # ------------------------------------------------------------------------------------------ constraint penalties (D-56; rl/anchors.py)\n",
            block("    # ------------------------------------------------------------------------------------------ long documents (D-60", "    # ------------------------------------------------------------------------------------------ constraint penalties (D-56") + "    # ------------------------------------------------------------------------------------------ constraint penalties (D-56; rl/anchors.py)\n")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--cur", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for rel, fn in (("gemma_mt/rl/reward.py", "longdoc_v2_D60_reward.patch"), ("gemma_mt/rl/chunk_plan.py", "longdoc_v2_D60_chunk_plan.patch")):
        o = open(os.path.join(a.orig, rel)).read()
        c = open(os.path.join(a.cur, rel)).read()
        n = apply_reward(o, c) if rel.endswith("reward.py") else c
        d = "".join(difflib.unified_diff(o.splitlines(True), n.splitlines(True), "a/" + rel, "b/" + rel))
        open(os.path.join(a.out, fn), "w").write(d)
        print(fn, d.count("\n"), "lines")


if __name__ == "__main__":
    main()
