# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline replay of the adaptive language-weight updater (D-65, rl/lang_weights.py) over a finished (or running: read-only) metrics.jsonl.

Prints the weight trajectory the updater WOULD have produced, update by update, with the factor components and flags of the languages that moved, and can write it as CSV.

Two kinds of metrics files:
  * runs with `lang_stats_log: true` / `adaptive_lang_weights: true`  -> the exact per-language step statistics `lang/n|reward|radj|fail/<iso>` (+ `reward/anchors/penalty/<iso>`)
  * older files (the 2c run, the running final run before it is restarted with the patched trainer): PROXIES from keys that always existed: which languages were drawn (the replayed schedule
    `--draws`, else the union of the per-language keys of that step), failure rates `reward/fail/<class>/<iso>`, `reward/anchors/penalty/<iso>`, and (optionally) the greedy hold-out reward
    `eval/reward/<iso>` as a 2-sample observation. NO per-language training reward exists in those files, so the reward terms (boost for under-performance, saturation shrink) are silent there;
    only failure-driven moves and the dev chrF++ gates (bench/chrf/<iso>, bench_aux/<set>/chrf/<iso>) can act. Said plainly in the output.

    python scripts/replay_lang_weights.py --metrics runs/rl2c/metrics.jsonl --calib calibration/shaping_flores_dev_v8.json --config runs/rl2c/config.json \
        --draws reports/data/lang_draws_by_phase.json --out-csv /tmp/traj.csv
    python scripts/replay_lang_weights.py --metrics runs/final/metrics.jsonl --calib ... --base-json '{"yor": 2.5, ...}' --every 2 --first-step 2 --params '{"min_groups": 1}'
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional

FAIL_CLASSES = ("bad_prefix", "copy_source", "dangling_marker", "degenerate", "empty", "length_ratio", "no_end_of_turn", "post_eos_rubbish", "special_tokens", "wrong_language")
LOOP_CLASSES = ("degenerate", "no_end_of_turn", "post_eos_rubbish", "dangling_marker")
DEV_N = {"bench": 40, "general_dev": 40, "weak_dev": 60, "afrisci_dev_doc": 25}


class DummySchedule:
    def __init__(self):
        self.language_weights = {}

    def set_language_weights(self, w):
        self.language_weights = dict(w)


def load_base(config_path: Optional[str], base_json: Optional[str]) -> Dict[str, float]:
    if base_json:
        return {k: float(v) for k, v in json.loads(base_json).items()}
    if not config_path:
        return {}
    if config_path.endswith(".json"):
        d = json.load(open(config_path))
        return {k: float(v) for k, v in (d.get("config", d).get("language_weights") or {}).items()}
    import yaml                                                                           # a yaml config: needs PyYAML (the box venv has it)

    return {k: float(v) for k, v in (yaml.safe_load(open(config_path)).get("language_weights") or {}).items()}


def per_lang(row: dict, prefix: str) -> Dict[str, float]:
    out = {}
    for k, v in row.items():
        if k.startswith(prefix):
            tail = k[len(prefix):]
            if re.fullmatch(r"[a-z]{3}", tail):
                out[tail] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--calib", required=True, help="calibration json (content mode per language)")
    ap.add_argument("--config", default=None, help="run config.json (or yaml): language_weights = the base weights")
    ap.add_argument("--base-json", default=None, help="base weights as a JSON dict (overrides --config)")
    ap.add_argument("--draws", default=None, help="lang_draws_by_phase.json (scripts/replay_schedule_draws.py): exact prompts per language per step of the 2c run")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--first-step", type=int, default=100)
    ap.add_argument("--params", default=None, help="AdaptiveParams overrides as JSON")
    ap.add_argument("--treated", default=None, help="comma separated languages the updater may move (default all)")
    ap.add_argument("--no-eval-reward", action="store_true", help="do not use eval/reward/<iso> as a (2-sample) reward observation")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    from gemma_mt.languages import canonical_iso
    from gemma_mt.rl import lang_weights as lw

    calib = json.load(open(a.calib))
    languages = sorted(k for k in calib if len(k) == 3 and isinstance(calib[k], dict))
    modes = {l: (calib[l].get("content") or {}).get("mode", "qe") for l in languages}
    base = load_base(a.config, a.base_json)
    params = lw.AdaptiveParams.from_dict(json.loads(a.params) if a.params else None)
    params.first_step = a.first_step
    draws: Optional[List[List[str]]] = json.load(open(a.draws))["steps_2c"] if a.draws else None
    sched = DummySchedule()
    tmp = tempfile.mkdtemp(prefix="replay_lw_")
    ctl = lw.LangWeightController(tmp, sched, base, languages, modes, file_control=False, adaptive=True, every=a.every, params=params,
                                  treated=a.treated.split(",") if a.treated else None, canon=canonical_iso, log=(lambda m: None))
    ctl.set_dev_sizes({(k, None): v for k, v in DEV_N.items()})
    rows = [json.loads(l) for l in open(a.metrics) if l.strip()]
    mode_kind = "lang/*" if any(k.startswith("lang/reward/") for r in rows for k in r) else "proxy"
    last = 0
    traj = []                                       # (step, {iso: weight}, diagnostics of the update)
    ctl.update(0)
    traj.append((0, dict(ctl.in_force), {}))
    n_train = 0
    for row in rows:
        if "reward" in row and "step" in row and not row.get("baseline"):
            s = int(row["step"])
            if s <= last:
                continue                            # a resumed run repeats steps: the first occurrence stands
            last = s
            n_train += 1
            before = ctl.n_updates
            ctl.update(s - 1)
            if ctl.n_updates > before:
                traj.append((s - 1, dict(ctl.in_force), {l: d for l, d in ctl._last_diag.items() if d.get("why") == "ok"}))
            G = a.group_size
            if mode_kind == "lang/*":
                ns, rs, fs, ra = per_lang(row, "lang/n/"), per_lang(row, "lang/reward/"), per_lang(row, "lang/fail/"), per_lang(row, "lang/radj/")
                pen = per_lang(row, "reward/anchors/penalty/")
                for l, n in ns.items():
                    rew = (row["reward"] + ra[l]) if l in ra else rs.get(l)
                    ctl.window.add_aggregate(l, "all", n, n / G, rew, fail=fs.get(l, 0.0), pen=pen.get(l, 0.0))
            else:
                if draws is not None and s - 1 < len(draws):
                    cnt = Counter(draws[s - 1])
                else:
                    drawn = set()
                    for pre in ("reward/lid/fail_rate/", "reward/consensus/chrf/", "reward/anchors/penalty/", "reward/qe_ens/min_share/", "reward/diacritic/deficit/"):
                        drawn |= set(per_lang(row, pre))
                    for c in FAIL_CLASSES:
                        drawn |= set(per_lang(row, f"reward/fail/{c}/"))
                    cnt = Counter({l: 1 for l in drawn})
                fail = defaultdict(float)
                loop = defaultdict(float)
                for c in FAIL_CLASSES:
                    for l, v in per_lang(row, f"reward/fail/{c}/").items():
                        fail[l] += v
                        if c in LOOP_CLASSES:
                            loop[l] += v
                pen = per_lang(row, "reward/anchors/penalty/")
                for l, g in cnt.items():
                    ctl.window.add_aggregate(l, "all", g * G, g, None, fail=min(fail.get(l, 0.0), 1.0), loop=min(loop.get(l, 0.0), 1.0), pen=pen.get(l, 0.0))
        elif "eval/reward" in row or any(k.startswith("bench") for k in row):
            s = int(row["step"])
            ctl.observe_eval(s, row)
            if mode_kind == "proxy" and not a.no_eval_reward:
                for l, v in per_lang(row, "eval/reward/").items():
                    ctl.window.add_aggregate(l, "eval", 2, 1, float(v))
    before = ctl.n_updates
    ctl.update(last)
    if ctl.n_updates > before:
        traj.append((last, dict(ctl.in_force), {l: d for l, d in ctl._last_diag.items() if d.get("why") == "ok"}))

    # ---------------------------------------------------------------------------------------------------------------- report
    print(f"metrics: {a.metrics} ({n_train} distinct training steps, last step {last}); signal source: {mode_kind}")
    if mode_kind == "proxy":
        print("NOTE: this file has no per-language training reward; the reward-driven terms are silent, only failure / penalty / dev-chrF terms can move a weight.")
    print(f"updater: every {a.every} steps from step {a.first_step}, {len(ctl.treated)} treated languages, {ctl.n_updates} updates")
    b = {l: float(base.get(l, 1.0)) for l in languages}
    prev = b
    for s, w, fl in traj[1:]:
        moved = sorted(((w[l] / prev[l] - 1.0, l) for l in languages if abs(w[l] / prev[l] - 1.0) > 0.02), key=lambda t: -abs(t[0]))
        flagged = sorted(l for l, d in fl.items() if d["flags"])
        if not a.quiet:
            print(f"  update @ {s:5d}: {len(moved):2d} languages moved >2% " + ", ".join(f"{l} {prev[l]:.2f}->{w[l]:.2f}" for _, l in moved[:8]) + (f" | flagged: {','.join(flagged[:10])}" if flagged else ""))
        prev = w
    if ctl.n_updates:
        final = traj[-1][1]
        flags_last = traj[-1][2]
        ch = sorted(((final[l] / b[l] - 1.0, l) for l in languages), key=lambda t: t[0])
        print(f"final vs base: sum {sum(final.values()):.2f} / {sum(b.values()):.2f}; largest decreases " + ", ".join(f"{l} {b[l]:.2f}->{final[l]:.2f}" for _, l in ch[:6]) +
              "; largest increases " + ", ".join(f"{l} {b[l]:.2f}->{final[l]:.2f}" for _, l in ch[-6:][::-1]))
        print("flags at the last update:", dict(Counter(f for x in flags_last.values() for f in x["flags"])))
        print("factor summary at the last update (languages with a signal):", len(flags_last), "| boosted >1.05:", sum(1 for x in flags_last.values() if x["f"] > 1.05),
              "| shrunk <0.95:", sum(1 for x in flags_last.values() if x["f"] < 0.95))
    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["step", "iso", "mode", "base", "weight", "ratio"])
            for s, w, _ in traj:
                for l in languages:
                    wr.writerow([s, l, modes[l], b[l], round(w[l], 5), round(w[l] / b[l], 5)])
        print("wrote", a.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
