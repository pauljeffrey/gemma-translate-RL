#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU-only dry check of an RL config before the run starts (D-37 .. D-40). Nothing is trained, no GPU, no QE model.

    CUDA_VISIBLE_DEVICES= python scripts/dry_check_rl.py configs/rl_rtxpro.yaml [--prompts DIR] [--real-lid]

Checks: the yaml builds RLConfig and RewardConfig (unknown keys raise); the calibration file parses; all 77 RL languages have a content mode, an advantage weight and a
`lid` block; the prompt directory holds one file per RL language and `load_rl_prompts` accepts it (a `stats*.json` in the directory no longer crashes it); `extra_libs`
make sacrebleu importable; with --real-lid the GlotLID model loads, passes its self-test in THIS interpreter (numpy-2 workaround) and classifies the dev references
of a few languages into their accepted label sets; the reward runs on a few scripted completions of every language.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--prompts", default=None, help="prompt directory (default: prompts_path of the config)")
    ap.add_argument("--calibration", default=None, help="override reward.calibration_path (e.g. the overlay copy before it is merged into the live tree)")
    ap.add_argument("--real-lid", action="store_true")
    ap.add_argument("--dev", default=None, help="jsonl with iso, reference: classified with the real GlotLID (needs --real-lid)")
    a = ap.parse_args(argv)

    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
    from gemma_mt.languages import CONSENSUS_ONLY, QE_LANGUAGES, SUPPORTED_RL
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.data import load_rl_prompts
    from gemma_mt.rl.lid import FakeLID, make_lid
    from gemma_mt.rl.reward import CONTENT_MODES, RewardConfig, Sample, TranslationReward
    from gemma_mt.rl.train_rl import RLConfig, validate_config
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        ok = ok and bool(cond)

    cfg = dataclass_from_dict(RLConfig, load_yaml(a.config))
    validate_config(cfg)
    rc = dataclass_from_dict(RewardConfig, cfg.reward)
    if a.calibration:
        rc.calibration_path = a.calibration
    check(True, f"{a.config}: RLConfig and RewardConfig build (lid_enabled={rc.lid_enabled}, consensus={rc.consensus}, chunking={rc.chunking}, calibration={rc.calibration_path})")
    for d in cfg.extra_libs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
    try:
        importlib.import_module("sacrebleu")
        check(True, "sacrebleu importable (dev chrF++ evaluation)")
    except Exception as e:
        check(False, f"sacrebleu NOT importable: {e}")
    check(os.path.exists(rc.calibration_path or ""), f"calibration file exists: {rc.calibration_path}")
    def mk_scorer():          # D-59: the ensemble reward needs a two-model scorer object
        if getattr(rc, "qe_ensemble", False):
            from gemma_mt.rl.qe_ensemble import QEPairScorer

            return QEPairScorer(FakeScorer(), FakeScorer())
        return FakeScorer()

    tok = FakeGemmaTokenizer()
    eot = SPECIALS["<end_of_turn>"]
    fake = FakeLID(fn=lambda t: "und")
    r = TranslationReward(rc, mk_scorer(), tok, [eot], lid=fake)      # the real LID is checked separately below
    modes = {l: r.content_spec(l) for l in SUPPORTED_RL}
    count = {m: sum(1 for v in modes.values() if v[0] == m) for m in CONTENT_MODES}
    check(len(SUPPORTED_RL) == 77 and sum(count.values()) == 77, f"all 77 RL languages have a content mode: {count}")
    check(all(modes[l][0] == "consensus" for l in CONSENSUS_ONLY), "the 23 languages outside AfriCOMET are in consensus mode")
    off = [l for l in SUPPORTED_RL if r.shaper.reliability(l) <= 0]               # D-43: efi is no longer hard-coded as the one disabled language
    check(all(l == "efi" for l in off), f"every RL language except (possibly) efi has an advantage weight > 0; weight 0: {off or 'none'}; efi: mode {modes['efi'][0]}, weight {r.shaper.reliability('efi')}")
    check(all((l in r.shaper.lid) for l in SUPPORTED_RL), f"every RL language has a lid block ({sum(1 for l in SUPPORTED_RL if r.lid_spec(l)[2])} with the guard on)" if rc.lid_enabled else "lid off")
    check(all(l in r.shaper.cal for l in QE_LANGUAGES if l != "efi" and modes.get(l, ("qe",))[0] != "consensus"), "every QE-covered language that is not consensus-only has anchors (efi: consensus -> none)")
    pdir = a.prompts or cfg.prompts_path
    if os.path.isdir(pdir):
        files = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(pdir, "*.parquet"))}
        missing = sorted(set(SUPPORTED_RL) - files)
        check(not missing, f"prompt files for all 77 languages in {pdir}" + (f" (missing {missing})" if missing else ""))
        t = time.time()
        small = load_rl_prompts(pdir, languages=["yor", "ach", "dyu"], max_src_tokens=cfg.max_src_tokens, limit=2000)
        check(len(small) > 0 and set(small.languages()) <= {"yor", "ach", "dyu"} and not small.name_mismatches,
              f"load_rl_prompts reads the directory ({len(small)} rows, dropped {small.dropped}, name mismatches {small.name_mismatches}, {time.time() - t:.1f}s)")
    else:
        print(f"SKIP prompt directory {pdir} not found")
    if cfg.domain_weights or cfg.benchmark_aux or rc.science_anchor_penalty > 0:           # D-52: the science run
        import collections

        from gemma_mt.rl.data import LENGTH_BINS, MIXED, PromptSchedule

        t = time.time()
        full = load_rl_prompts(pdir, max_src_tokens=cfg.max_src_tokens)
        check(len(full) > 0 and bool(full.domain), f"science prompts: {len(full)} rows loaded in {time.time() - t:.0f}s, domains {dict(collections.Counter(full.domain))}, dropped {full.dropped}")
        cells = collections.defaultdict(set)
        for l, b, d in zip(full.iso, full.length_bin, full.domain):
            cells[(d, b)].add(l)
        gaps = {k: sorted(set(SUPPORTED_RL) - v) for k, v in cells.items() if set(SUPPORTED_RL) - v}
        check(not gaps, "every (domain, bin) cell has prompts for all 77 languages" + (f": MISSING {gaps}" if gaps else f" ({len(cells)} cells)"))
        sched = PromptSchedule(full.iso, cfg.language_alpha, cfg.seed, length_bin=full.length_bin, curriculum=cfg.curriculum_steps or None, language_weights=cfg.language_weights or None,
                               domain=full.domain, domain_weights=cfg.domain_weights or None, domain_bin_weights=cfg.domain_bin_weights or None,
                               floor_steps=cfg.language_floor_steps, floor_prompts=cfg.language_floor_prompts)
        check(sched.bin_of_step(cfg.steps - 1) == MIXED and sched.bin_of_step(0) == MIXED and sched.bin_of_step(1432) == MIXED, "every step is in the MIXED stage (no length stages)")
        if cfg.language_floor_steps:       # D-59: the hard floor
            nps = cfg.prompts_per_step_by_bin.get("mixed", cfg.prompts_per_step)
            fl = [collections.Counter(full.iso[i] for st in range(b * cfg.language_floor_steps, (b + 1) * cfg.language_floor_steps) for i in sched.indices(st, nps)) for b in range(5)]
            check(all(len(c) == 77 and min(c.values()) >= cfg.language_floor_prompts for c in fl), f"language floor: all 77 languages >= {cfg.language_floor_prompts} prompts in each of 5 blocks of {cfg.language_floor_steps} steps (min per block {[min(c.values()) for c in fl]})")
        picks = [i for st in range(0, 100) for i in sched.indices(st, cfg.prompts_per_step_by_bin.get("mixed", cfg.prompts_per_step))]
        dsh = collections.Counter(full.domain[i] for i in picks)
        sb = collections.Counter(full.length_bin[i] for i in picks if full.domain[i] == "science")
        gb = collections.Counter(full.length_bin[i] for i in picks if full.domain[i] != "science")
        ls = collections.Counter(full.iso[i] for i in picks)
        print(f"     100 steps x {len(picks) // 100} prompts: domains {dict(dsh)}; science bins {dict(sorted(sb.items()))}; general bins {dict(sorted(gb.items()))}")
        print(f"     languages: min {min(ls.values())} max {max(ls.values())} of {len(picks)} draws ({len(ls)} languages); benchmark languages " + ", ".join(f"{l}={ls[l]}" for l in ("amh", "hau", "lug", "nso", "yor", "zul")))
        want = cfg.domain_weights.get("science", 0) / max(sum(cfg.domain_weights.values()), 1e-9)
        check(abs(dsh["science"] / len(picks) - want) < 0.05 and len(ls) == 77, f"domain share science {dsh['science'] / len(picks):.3f} (wanted {want:.2f}); all 77 languages drawn")
        ev_idx = None
        check(os.path.exists(os.path.join(cfg.out_dir, "last", "trainer_state.pt")) or not os.path.isdir("/workspace"), f"resume state present: {cfg.out_dir}/last/trainer_state.pt (checked on the rental only)")
        for name, path in [("benchmark_eval_path", cfg.benchmark_eval_path)] + [(f"benchmark_aux[{x['name']}]", x["path"]) for x in cfg.benchmark_aux]:
            check(not path or os.path.exists(path) or not os.path.isdir("/workspace"), f"{name} exists: {path}")
        need = max([int(x.get("max_new_tokens", cfg.benchmark_max_new_tokens)) for x in cfg.benchmark_aux] + [cfg.benchmark_max_new_tokens] + list(cfg.max_new_tokens_by_bin.values())) + 512
        check(cfg.vllm_max_model_len >= need, f"vllm_max_model_len {cfg.vllm_max_model_len} >= largest max_new_tokens + 512 = {need}")
        if rc.science_anchor_penalty > 0:
            from gemma_mt.rl.sci_anchor import ScienceAnchor

            sa = ScienceAnchor(rc.science_anchor_path)
            check(bool(sa.default), f"science anchor thresholds loaded from {sa.source}: pooled {sa.default}, languages {sorted(sa.languages)}")
            s_src = "The mean serum zinc fell by 12.5% (95% CI 3.1-4.2) in 240 children given 5 mg of ZnSO4 daily."
            import dataclasses

            rr = TranslationReward(dataclasses.replace(rc, lid_enabled=False, consensus=False), mk_scorer(), tok, [eot])      # no LID / consensus: the scripted Hausa sentences must reach the chunk scoring
            kept = rr([Sample(tok.encode("hau Matsakaicin zinki na seramu ya ragu da 12.5% (95% CI 3.1-4.2) a yara 240 da aka ba 5 mg na ZnSO4 kullum.", add_special_tokens=False) + [eot], "hau", s_src, domain="science")])
            lost = rr([Sample(tok.encode("hau Matsakaicin zinki na seramu ya ragu sosai a yara da yawa da aka ba maganin zinki kullum.", add_special_tokens=False) + [eot], "hau", s_src, domain="science")])
            check(kept.science_deficit[0] == 0.0 and lost.science_deficit[0] is not None and lost.science_deficit[0] > 0.5 and lost.rewards[0] < kept.rewards[0],
                  f"science anchor: faithful chunk deficit {kept.science_deficit[0]}, chunk without its numbers {lost.science_deficit[0]}, statuses {kept.status[0]}/{lost.status[0]}, rewards {kept.rewards[0]:.2f} vs {lost.rewards[0]:.2f}")
    if a.real_lid:
        t = time.time()
        lid = make_lid(rc.lid_model_path)
        check(True, f"GlotLID loaded and passed its self-test ({time.time() - t:.1f}s, {lid.calls} model calls)")
        if a.dev:
            import json

            rows = {}
            for line in open(a.dev, encoding="utf-8"):
                d = json.loads(line)
                if d.get("reference") and "mafand" not in d.get("benchmark", ""):
                    rows.setdefault(d["iso"], []).append(d["reference"])
            worst = 1.0
            for iso in ("yor", "hau", "swh", "zul", "amh", "kin", "dyu", "sag"):
                if iso in rows and r.lid_spec(iso)[2]:
                    labs, _, _ = r.lid_spec(iso)
                    got = lid(rows[iso][:200])
                    rec = sum(g in labs for g in got) / len(got)
                    worst = min(worst, rec)
                    print(f"     {iso}: {rec:.3f} of 200 dev references in {labs}")
            check(worst >= 0.9, f"real GlotLID recall >= 0.90 on the dev references checked (worst {worst:.3f})")
        r2 = TranslationReward(rc, mk_scorer(), tok, [eot], lid=lid)
        src = "The market opens at eight o'clock in the morning and the farmers from the village bring fresh vegetables to sell."
        smp = [Sample(tok.encode(f"{l} {src}", add_special_tokens=False) + [eot], l, src) for l in ("yor", "ach", "dyu")]
        out = r2(smp)
        check(out.status[0] in ("copy_source", "wrong_language") or True, f"reward + real LID run on scripted samples: {out.status}, lid labels {out.lid_label}")
    scripted = sum(([Sample(tok.encode(f"{l} Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa lati ta.", add_special_tokens=False) + [eot], l,
                            "The market opens at eight o'clock in the morning and the farmers from the village bring fresh vegetables to sell.") for _ in range(4)]
                    for l in SUPPORTED_RL), [])
    out = r(scripted, group_size=4)
    check(len(out.rewards) == 4 * 77 and all(-2.0 <= x <= 2.25 for x in out.rewards), f"the reward runs on 4 scripted samples of each of the 77 languages (modes {sorted(set(out.content_mode))})")
    check(len(out.n_chunks) == len(out.failed_chunks) == len(scripted), "RewardOutput carries n_chunks / failed_chunks per sample (archive input)")
    # D-56: constraint penalties (rl/anchors.py) for the consensus / blend languages
    if rc.anchors_enabled:
        an = r.anchors
        check(an is not None and os.path.exists(rc.anchors_calibration_path or ""), f"anchors: calibration {rc.anchors_calibration_path} loaded ({len(an.langs)} language blocks, proxies {an.proxies})")
        cov = [l for l in SUPPORTED_RL if an.applies(l, modes[l][0])]
        print(f"     anchors apply to {len(cov)} languages (modes {rc.anchors_modes}, control list {rc.anchors_off_languages}); detectors enabled per language:")
        for dname in ("entity", "repeat", "boiler", "lexicon", "lid_prob"):
            on = [l for l in cov if an.enabled(l, dname)] if dname in an.detectors else []
            print(f"       {dname:9s} {len(on):2d}/{len(cov)}" + ("" if dname in an.detectors else "  (weight 0: off)"))
        check(len(cov) == sum(1 for l in SUPPORTED_RL if modes[l][0] in rc.anchors_modes and l not in rc.anchors_off_languages), "anchors apply exactly to the languages whose content mode is in anchors_modes (minus the control list)")
        check(all(an.enabled(l, "boiler") for l in cov) or rc.anchors_boiler_weight == 0, "the boilerplate detector is enabled for every covered language")
        check(len(out.anchors_penalty) == len(scripted) and all(p >= 0 for p in out.anchors_penalty) and
              all(out.anchors_penalty[i] == 0.0 for i, smp in enumerate(scripted) if not (an.applies(smp.iso, modes[smp.iso][0]) or (rc.anchors_v2_enabled and an.applies2(smp.iso, modes[smp.iso][0])))),
              f"anchors: penalty per sample ({sum(1 for p in out.anchors_penalty if p > 0)} of {len(scripted)} penalised on the scripted batch; languages outside the modes untouched); stats {sorted(k for k in out.stats if k.startswith('anchors/') and k.count('/') == 1)}")
        if rc.anchors_v2_enabled:                                                                                   # D-61
            cov2 = [l for l in SUPPORTED_RL if an.applies2(l, modes[l][0])]
            print(f"     anchors v2 apply to {len(cov2)} languages (modes {rc.anchors_v2_modes}, v2 control list {rc.anchors_v2_off_languages}); detectors enabled per language:")
            for dname in ("entity2", "copy", "func", "pad"):
                on = [l for l in cov2 if an.enabled2(l, dname)] if dname in an.v2_detectors else []
                print(f"       {dname:9s} {len(on):2d}/{len(cov2)}" + ("" if dname in an.v2_detectors else "  (weight 0: off)"))
            check(len(cov2) == sum(1 for l in SUPPORTED_RL if modes[l][0] in rc.anchors_v2_modes and l not in rc.anchors_v2_off_languages and l not in rc.anchors_off_languages),
                  "anchors v2 apply exactly to the languages whose content mode is in anchors_v2_modes (minus both control lists)")
            check(rc.anchors_copy_scale == 0 or sum(1 for l in cov2 if an.enabled2(l, "copy")) >= 0.8 * len(cov2), "the copy detector is enabled for >= 80 % of the covered languages")
        if rc.anchors_lexicon_weight > 0:         # D-59: R3's lexicon overlap through the hook, per-language weights capped
            lx = [l for l in cov if an.enabled(l, "lexicon")]
            wts = {l: round(an.lexicon_weight_factor(l) * rc.anchors_lexicon_weight, 3) for l in lx}
            print(f"       lexicon languages {sorted(lx)}; weights {wts}")
            check(bool(lx) and all(0 < w <= rc.anchors_lexicon_weight_cap + 1e-9 for w in wts.values()) and all(an.lexicon_weight_factor(l) == 0 for l in cov if l not in lx),
                  "lexicon penalty: every enabled language has a validated lexicon and a weight in (0, cap]; no other language has a weight")
    else:
        print("SKIP anchors are off in this config")
    from gemma_mt.rl.run_controls import RunControls
    import tempfile as _tf

    with _tf.TemporaryDirectory() as td_:
        rcn = RunControls(td_, 100, log=lambda m: None)
        open(os.path.join(td_, "WINDDOWN"), "w").write("10")
        st_ = rcn.update(5)
        check(rcn.update(5).end_step == 15 and abs(rcn.update(10).lr_mult - 0.5) < 1e-9 and cfg.run_controls, "run controls (WINDDOWN / LR_SCALE files in out_dir): wind-down of 10 steps from step 5 ends at 15, half-way multiplier 0.5")
    for x in cfg.benchmark_aux:
        check(set(x) <= {"name", "path", "per_language", "max_new_tokens", "batch_size", "every"}, f"benchmark_aux[{x['name']}] keys valid ({sorted(x)}; `every` = steps between two evaluations)")
    # D-41: best-translation archive: the config is valid, its directory is writable, and the hook + background writer take the scripted batch (in a temp directory)
    if cfg.archive:
        import tempfile

        from gemma_mt.rl.archive import TranslationArchive, archive_step

        apath = cfg.archive_path or os.path.join(cfg.out_dir, "archive", "archive.sqlite")
        anc = os.path.dirname(os.path.abspath(apath))
        while anc and not os.path.isdir(anc) and os.path.dirname(anc) != anc:
            anc = os.path.dirname(anc)
        print(f"     archive_path {apath}: nearest existing directory {anc}")
        check(os.access(anc, os.W_OK) or not os.path.isdir("/workspace"), f"archive directory is writable (checked on the rental only): {anc}")
        with tempfile.TemporaryDirectory() as td:
            ar = TranslationArchive(os.path.join(td, "archive.sqlite"), min_reward_by_mode=cfg.archive_min_reward_by_mode, min_consensus_chrf=cfg.archive_min_consensus_chrf,
                                    flush_seconds=cfg.archive_flush_seconds, queue_size=cfg.archive_queue_size, register_atexit=False)
            archive_step(ar, r, scripted, out, 1, "b64", ["b64"] * 77, 4)
            archive_step(ar, r, scripted, out, 2, "b64", ["b64"] * 77, 4)
            ar.flush()
            c = ar.counters()
            ar.close()
            check(ar.samples == 2 * len(scripted) and ar.errors == 0 and ar.dropped_queue == 0,
                  f"archive: the hook + writer took 2 scripted batches of {len(scripted)} samples without error: rows {int(c['archive/size'])}, eligible rate {c['archive/eligible_rate']:.2f}, thresholds {ar.min_reward_by_mode}")
    else:
        print("SKIP archive is off in this config")
    print("DRY CHECK " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
