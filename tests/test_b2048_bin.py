# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-45: the b2048 length bin of the Gemma RL prompts: window filters, the appended-bin invariance of the schedule, the per-bin hold-out top-up, the staged
config, and the reward code path on ~100-sentence sources. CPU only."""
import collections
import importlib.util
import os
import random

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _builder():
    spec = importlib.util.spec_from_file_location("build_rl_prompts_b2048", os.path.join(ROOT, "scripts", "build_rl_prompts_b2048.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ window filters
GOOD = ("The village council met on Tuesday to discuss the new market. Farmers from the hills brought fresh vegetables and fruit to sell. "
        "Many people came early because the weather was cool and the road was dry. The council promised to repair the bridge before the rains. ")


def test_window_filter_accepts_plain_prose_and_names_each_failure():
    b = _builder()
    prose = GOOD
    assert b.window_filter(prose) is None
    assert b.window_filter(prose + " Visit www.example.com today.") == "artifact"
    assert b.window_filter(prose + " Copyright © 2020.") == "artifact"
    assert b.window_filter(prose + " " + "7 " * 200) == "numeric_or_math"                          # digits / characters > 0.06
    assert b.window_filter(prose + " THE MARKET OPENS NOW AND EVERYONE MUST COME TODAY PLEASE " * 3) == "shouting_or_acronyms"
    assert b.window_filter("word " * 5 + "internationalization " * 6 + prose) == "technical_vocabulary"   # > 3 % of the words have >= 13 letters
    assert b.window_filter("Photosynthesis characterizes electromagnetic phenomena notwithstanding. " * 2 + "Qwrtp zxcv bnm. " * 3) in ("technical_vocabulary", "hard_to_read")
    hard = "Notwithstanding unprecedented macroeconomic uncertainties, institutional investors reconsidered fundamental diversification methodologies extensively. " * 4
    assert b.window_filter(hard) in ("hard_to_read", "technical_vocabulary")
    assert b.window_filter(GOOD * 3) == "repetition"                                              # a 4-gram seen >= 3 times


def test_the_exact_rules_have_the_thresholds_of_the_old_windows():
    b = _builder()
    assert b.DIGIT_MAX == 0.06 and b.LONG_WORD_MAX == 0.03 and b.FLESCH_MIN == 30.0 and b.REPEAT_4GRAM_MAX == 2
    assert b.max_4gram_count("a b c d a b c d x y z") == 2 and b.max_4gram_count("Short.") == 0
    assert abs(b.flesch("The cat sat on the mat.") - (206.835 - 1.015 * 6 - 84.6 * 6 / 6)) < 1e-9      # 6 words, 1 sentence, 6 syllables
    assert LO_HI(b) == (1025, 2048)


def LO_HI(b):
    return b.LO, b.HI


def test_clean_doc_keeps_complete_sentences_only_and_has_no_newline():
    b = _builder()
    doc = "Home | About | Contact\nHistory\nThe town was founded in 1850 by settlers from the north.\nShare this:\nIt grew quickly after the railway arrived.\n- item one\n"
    out = b.clean_doc(doc)
    assert "\n" not in out and out == "The town was founded in 1850 by settlers from the north. It grew quickly after the railway arrived."


# ------------------------------------------------------------------------------------------------ schedule: the new bin is appended, steps 0..985 do not move
def _table(n_old_per_bin=300, n_new=100, langs=("yor", "hau", "ibo"), with_new=True):
    from gemma_mt.rl.data import RLPrompts

    eng, iso, lb = [], [], []
    for l in langs:                                              # old files in language order (the global row order of load_rl_prompts)
        for b in ("b64", "b256", "b512", "b1024"):
            for i in range(n_old_per_bin):
                eng.append(f"{l} {b} sentence {i}")
                iso.append(l)
                lb.append(b)
    if with_new:
        for l in langs:                                          # zz_b2048/<iso>.parquet sort after every old file
            for i in range(n_new):
                eng.append(f"{l} b2048 window {i}")
                iso.append(l)
                lb.append("b2048")
    return RLPrompts(eng, iso, ["x"] * len(eng), [10] * len(eng), {}, 0, lb)


def test_appending_b2048_changes_neither_the_stage_map_nor_the_draws_of_the_old_stages():
    from gemma_mt.rl.data import MIXED, PromptSchedule

    old_cur = {"b256": 7, "b512": 9, "b1024": 5}
    new_cur = {**old_cur, "b2048": 11}
    w = {"yor": 2.5}
    so = PromptSchedule(_table(with_new=False).iso, 0.0, 1234, length_bin=_table(with_new=False).length_bin, curriculum=old_cur, language_weights=w)
    sn = PromptSchedule(_table().iso, 0.0, 1234, length_bin=_table().length_bin, curriculum=new_cur, language_weights=w)
    staged = sum(old_cur.values())
    for s in range(staged):
        assert so.bin_of_step(s) == sn.bin_of_step(s)
    for s in range(staged, staged + 11):
        assert so.bin_of_step(s) == MIXED and sn.bin_of_step(s) == "b2048"
    assert all(sn.bin_of_step(s) == MIXED for s in range(staged + 11, staged + 20))
    to, tn = _table(with_new=False), _table()
    for s in range(staged):                                      # same prompts at the same steps (the old rows keep their global indices)
        io, inn = so.indices(s, 21), sn.indices(s, 21)
        assert io == inn and [to.english[i] for i in io] == [tn.english[i] for i in inn]
    for s in range(staged, staged + 11):                         # the new stage draws only b2048
        assert {tn.length_bin[i] for i in sn.indices(s, 21)} == {"b2048"}
    seen = set()
    for s in range(staged + 11, staged + 60):                    # the mixture draws uniformly over the 5 bins present
        seen |= {tn.length_bin[i] for i in sn.indices(s, 42)}
    assert seen == {"b64", "b256", "b512", "b1024", "b2048"}


def test_mixture_is_uniform_over_the_five_bins_present():
    from gemma_mt.rl.data import PromptSchedule

    t = _table(n_old_per_bin=200, n_new=200)
    sch = PromptSchedule(t.iso, 0.0, 7, length_bin=t.length_bin, curriculum={"b2048": 1})
    c = collections.Counter()
    for s in range(1, 120):
        c.update(t.length_bin[i] for i in sch.indices(s, 42))
    tot = sum(c.values())
    assert set(c) == {"b64", "b256", "b512", "b1024", "b2048"} and all(0.15 < v / tot < 0.25 for v in c.values())


# ------------------------------------------------------------------------------------------------ hold-out: the original selection is a subset of the new one
def test_split_eval_prompts_default_is_unchanged_and_min_per_bin_tops_up_only_the_thin_bins():
    from gemma_mt.data.prepare_sft import split_fraction
    from gemma_mt.rl.train_rl import split_eval_prompts

    p = _table(n_old_per_bin=400, n_new=30)
    tr0, ev0 = split_eval_prompts(p, 64, 1234)
    order = sorted(range(len(p)), key=lambda i: (split_fraction(p.english[i], 1234), i))
    assert ev0 == sorted(order[:64])                               # the original rule, byte for byte
    tr1, ev1 = split_eval_prompts(p, 64, 1234, min_per_bin=12)
    assert set(ev0) <= set(ev1)
    have = collections.Counter(p.length_bin[i] for i in ev1)
    assert all(have[b] >= 12 for b in ("b64", "b256", "b512", "b1024", "b2048"))
    extra = set(ev1) - set(ev0)
    assert extra and all(p.length_bin[i] in {b for b in have if collections.Counter(p.length_bin[j] for j in ev0)[b] < 12} for i in extra)
    assert not (set(tr1) & set(ev1)) and sorted(tr1 + ev1) == list(range(len(p)))
    # the added prompts are the thin bin's own smallest-hash prompts
    b2 = [i for i in order if p.length_bin[i] == "b2048"]
    assert set(b2[: have["b2048"]]) <= set(ev1)
    # no english text is on both sides (distinct texts here)
    assert not ({p.english[i] for i in tr1} & {p.english[i] for i in ev1})


# ------------------------------------------------------------------------------------------------ the staged config
def _yaml(name):
    pytest.importorskip("yaml")
    from gemma_mt.utils import load_yaml

    return load_yaml(os.path.join(ROOT, "configs", name))


def test_staged_config_loads_strictly_and_the_stage_arithmetic_holds():
    pytest.importorskip("torch")
    from gemma_mt.rl.reward import RewardConfig
    from gemma_mt.rl.train_rl import RLConfig, validate_config
    from gemma_mt.utils import dataclass_from_dict

    live, new = _yaml("rl_rtxpro_phase2c.yaml"), _yaml("rl_rtxpro_phase2c_b2048.yaml")
    cfg = dataclass_from_dict(RLConfig, new)
    validate_config(cfg)
    dataclass_from_dict(RewardConfig, cfg.reward)
    changed = {k for k in set(live) | set(new) if live.get(k) != new.get(k)}
    # D-59: the staged yaml was later also given the 2026-10-20 KL estimator (k3 -> k3c, applied to the live run at the b2048 restart); that is the only other difference
    assert (live.get("kl_estimator"), new.get("kl_estimator")) in {("k3", "k3c"), (new.get("kl_estimator"), new.get("kl_estimator"))}
    changed.discard("kl_estimator")
    kl_changed = live.get("kl_estimator") != new.get("kl_estimator")
    assert changed == {"prompts_path", "steps", "max_src_tokens", "curriculum_steps", "prompts_per_step_by_bin", "max_new_tokens_by_bin", "vllm_max_model_len", "eval_min_per_bin"}
    assert list(new["curriculum_steps"]) == ["b256", "b512", "b1024", "b2048"]                      # appended AFTER b1024
    assert {k: v for k, v in new["curriculum_steps"].items() if k != "b2048"} == live["curriculum_steps"]
    n = new["curriculum_steps"]["b2048"]
    per_step = new["prompts_per_step_by_bin"]["b2048"] * new["group_size"]
    assert n == round(100000 / per_step) and abs(n * per_step - 100000) < per_step
    assert new["steps"] == live["steps"] + n and new["steps"] - sum(new["curriculum_steps"].values()) == live["steps"] - sum(live["curriculum_steps"].values()) == 149    # the mixed stage is 149 steps
    assert new["max_src_tokens"] >= 2048 and new["max_new_tokens_by_bin"]["b2048"] == 7168 and new["max_new_tokens_by_bin"]["mixed"] >= new["max_new_tokens_by_bin"]["b2048"]
    assert new["vllm_max_model_len"] >= 2400 + max(new["max_new_tokens_by_bin"].values())
    for k, v in live.items():                                                                       # everything else is identical
        if k not in changed and not (k == "kl_estimator" and kl_changed):
            assert new[k] == v, k
    assert new["reward"] == live["reward"] and new["language_weights"] == live["language_weights"] and new["hub_repo_id"] == live["hub_repo_id"]
    assert new["eval_min_per_bin"] == 16 and new["eval_prompts"] == live["eval_prompts"]


def test_stage_end_branches_include_stage_b2048_and_old_ones_do_not_move():
    live, new = _yaml("rl_rtxpro_phase2c.yaml"), _yaml("rl_rtxpro_phase2c_b2048.yaml")

    def ends(cfg):
        acc, out = 0, {}
        for name, n in cfg["curriculum_steps"].items():
            acc += int(n)
            if acc <= cfg["steps"]:
                out[acc] = name
        out.setdefault(cfg["steps"], "mixed")
        return out

    eo, en = ends(live), ends(new)
    assert eo == {391: "b256", 837: "b512", 986: "b1024", 1135: "mixed"}
    assert en == {391: "b256", 837: "b512", 986: "b1024", 986 + new["curriculum_steps"]["b2048"]: "b2048", new["steps"]: "mixed"}


def test_training_with_a_missing_staged_bin_is_refused_not_skipped():
    pytest.importorskip("torch")
    from gemma_mt.rl.data import PromptSchedule

    t = _table(with_new=False)
    sch = PromptSchedule(t.iso, 0.0, 1, length_bin=t.length_bin, curriculum={"b1024": 3, "b2048": 5})
    assert ("b2048", 5) not in sch.stages                          # what the guard in train_rl catches: the stage silently vanishes from the schedule


# ------------------------------------------------------------------------------------------------ reward path on a ~100-sentence source
def _long_source(rng, n_sent):
    vocab = ["".join(rng.choice("abdefghiklmnoprstuwy") for _ in range(rng.randint(2, 8))) for _ in range(500)]
    return " ".join(" ".join(rng.choice(vocab) for _ in range(rng.randint(8, 22))).capitalize() + "." for _ in range(n_sent)), vocab


def test_reward_handles_100_sentence_sources_and_7k_token_outputs():
    pytest.importorskip("torch")
    from gemma_mt.fake_tokenizer import SPECIALS, FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
    from gemma_mt.rl.segments import split_sentences

    tok = FakeGemmaTokenizer()
    eot = SPECIALS["<end_of_turn>"]
    rng = random.Random(5)
    src, vocab = _long_source(rng, 100)
    sents = split_sentences(src)
    assert 90 <= len(sents) <= 100
    cfg = RewardConfig(chunking="sentences", consensus=True, lid_enabled=False)
    rw = TranslationReward(cfg, FakeScorer(), tok, [eot], lid=None)
    G = 8
    samples = []
    for g in range(G):
        out = []
        for s in sents:
            ws = [w[::-1] + "ɛ" if rng.random() > 0.08 else rng.choice(vocab) + "à" for w in s.lower().strip(".").split()]      # a deterministic word-for-word "translation", 8 % of the words differ per sample
            out.append(" ".join(ws).capitalize() + ".")
        if g in (3, 5):                                         # a different number of sentences: the alignment path
            out = out[:-2] if g == 3 else out + out[:2]
        text = "yor " + " ".join(out)
        ids = tok.encode(text, add_special_tokens=False) + [eot]
        samples.append(Sample(ids, "yor", src))
    assert max(len(s.token_ids) for s in samples) > 5000        # byte tokenizer: ~6-7k tokens, the size of a b2048 completion
    o = rw(samples, group_size=G)
    assert len(o.rewards) == G and collections.Counter(o.status)["ok"] >= G - 2, o.status
    assert all(-2.0 - 1e-6 <= r <= 2.25 + 1e-6 for r in o.rewards)
    assert o.stats.get("chunk/segments_per_sample", 0) > 50, o.stats


def test_consensus_cost_of_a_b2048_step_is_bounded():
    from gemma_mt.rl import consensus as cons

    rng = random.Random(1)
    vocab = ["".join(rng.choice("abdefghiklmnoprstuwyáàéèọẹṣ") for _ in range(rng.randint(2, 9))) for _ in range(2000)]
    base = " ".join(rng.choice(vocab) for _ in range(1700))                      # ~10,000 characters, a b2048 completion
    texts = []
    for _ in range(16):
        ws = base.split()
        for i in range(0, len(ws), 9):
            ws[i] = rng.choice(vocab)
        texts.append(" ".join(ws))
    import time

    t = time.time()
    v = cons.group_consensus([cons.Profile(x) for x in texts])
    dt = time.time() - t
    assert all(x is not None and 40 < x < 100 for x in v)
    assert dt < 60      # ~1-3 s per group of 16 on a quiet core (~7 s on the saturated rental); 21 groups per b2048 step: see reports/b2048_bin_2026-10-20.md


# ------------------------------------------------------------------------------------------------ end to end: a b2048 stage in the tiny CPU run
def test_cpu_smoke_runs_a_b2048_stage_then_the_mixture(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke
    from gemma_mt.smoke import ENGLISH, write_jsonl

    rows = []
    for b, tok in (("b64", 12), ("b256", 100), ("b2048", 1500)):
        for i in range(30):
            rows.append({"english": f"{ENGLISH[i % len(ENGLISH)]} ({b} {i})", "iso": ("yor", "hau", "ibo")[i % 3], "language": "", "src_afr_tokens": tok,
                         "source_dataset": "smoke", "length_bin": b})
    p = write_jsonl(str(tmp_path / "p.jsonl"), rows)
    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=20, overrides=dict(
        prompts_path=p, max_src_tokens=2048, length_bins=True, curriculum_steps={"b64": 1, "b2048": 2}, steps=5, eval_min_per_bin=3, eval_prompts=4, eval_every=2, save_every=0,
        prompts_per_step_by_bin={"b64": 3, "b2048": 2, "mixed": 3}, max_new_tokens_by_bin={"b64": 24, "b2048": 40, "mixed": 40}))
    assert summ["steps"] == 5
    import json

    recs = [json.loads(l) for l in open(os.path.join(str(tmp_path / "w"), "rl", "metrics.jsonl")) if l.strip()]
    ev = [r for r in recs if "eval/reward" in r]
    assert ev and all("eval/reward_bin/b2048" in r for r in ev), "the hold-out must contain b2048 prompts (eval_min_per_bin)"


# ------------------------------------------------------------------------------------------------ segments: the last-word lookup is bounded and decides exactly as before
def _old_split(text):
    """segments.split_sentences with the original unbounded `_LAST_WORD.search(head)`."""
    import re

    from gemma_mt.rl import segments as S

    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return []
    out, last = [], 0
    for m in S._CANDIDATE.finditer(t):
        end = m.end(2)
        start, term, next_pos = m.start(1), m.group(1), m.end()
        head = t[:start]
        mm = S._LAST_WORD.search(head)
        ok = True
        if next_pos < len(t) and t[next_pos].isalpha() and t[next_pos].islower() and "።" not in term:
            ok = False
        else:
            word = mm.group(1) if mm else ""
            bare = word.strip("\"'“‘(«[{").lower()
            if term == ".":
                if bare in S._ABBREV or bare.rstrip(".") in S._ABBREV:
                    ok = False
                elif re.fullmatch(r"[A-Za-z]", bare or "x") and word[:1].isupper():
                    ok = False
                elif re.fullmatch(r"(?:[A-Za-z]\.){2,}", bare + "."):
                    ok = False
                elif bare.isdigit() and len(bare) <= 3:
                    before = head[: mm.start(1)].rstrip()
                    if not before or before[-1] in S._TERMINATORS + ":;":
                        ok = False
        if ok:
            seg = t[last:end].strip()
            if seg:
                out.append(seg)
            last = m.end()
    tail = t[last:].strip()
    if tail:
        out.append(tail)
    return out


def test_split_sentences_with_the_bounded_last_word_is_identical_to_the_unbounded_one():
    from gemma_mt.rl.segments import split_sentences

    rng = random.Random(11)
    pieces = ["Dr.", "Mr. Smith", "e.g.", "U.S.", "J.", "etc. and", "3.5", "1.", "12.", "x" * 300 + ".", "y" * 255 + " Dr.", "z" * 257 + ".", "He said so.", "Why?", "Stop!", "።", "ሰላም።", "the end.\"", "(see fig. 2)", "p.m.", "No. 5"]
    for _ in range(400):
        text = " ".join(rng.choice(pieces) + (" " if rng.random() < 0.7 else "") + rng.choice(["Next", "next", "The", "", "2"]) for _ in range(rng.randint(1, 60)))
        assert split_sentences(text) == _old_split(text), text[:200]
    assert split_sentences("") == [] == _old_split("")


def test_topped_up_holdout_takes_every_copy_of_a_window_used_for_several_languages():
    from gemma_mt.rl.data import RLPrompts
    from gemma_mt.rl.train_rl import split_eval_prompts

    eng, iso, lb = [], [], []
    for i in range(200):
        eng.append(f"old {i}")
        iso.append("yor")
        lb.append("b64")
    for w in range(10):                                           # 10 b2048 windows, each used for 3 languages (3 rows with the same English)
        for l in ("yor", "hau", "ibo"):
            eng.append(f"long window {w}")
            iso.append(l)
            lb.append("b2048")
    p = RLPrompts(eng, iso, ["x"] * len(eng), [10] * len(eng), {}, 0, lb)
    tr, ev = split_eval_prompts(p, 20, 1234, min_per_bin=4)
    held = {p.english[i] for i in ev if p.length_bin[i] == "b2048"}
    assert len(held) >= 2 and not ({p.english[i] for i in tr} & held)           # 4 rows needed -> 2 windows (6 rows), no copy left in training
    assert sum(1 for i in ev if p.length_bin[i] == "b2048") == 3 * len(held)
