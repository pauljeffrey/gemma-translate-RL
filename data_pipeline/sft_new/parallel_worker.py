"""Extra build workers for `sft_new.combine --all` (same output folder, separate dedupe database).

    python -m sft_new.parallel_worker --out DIR --name A --steps old_eng_sft
    python -m sft_new.parallel_worker --out DIR --name B --from-end 40            # the last 40 plan steps, taken from the end backwards

How it coexists with the running sequential build: before a step starts, the worker writes a placeholder `stats/<tag>.json`, which the main build
reads as "already done" and skips; run_one overwrites the placeholder with the real statistics when the step finishes. A worker dedupes against its
OWN sqlite file (seen_<name>.sqlite), so nothing is shared with the main build's database; duplicates across workers are removed afterwards by one
pass over the `id` column of all shards (ids are content hashes, so the first occurrence in plan order wins).

Safety: each worker runs at whatever priority it is started with (use nice -n 19) and stops BEFORE a step when swap use or free disk is past a limit;
placeholders of steps it did not finish are removed so the main build picks them up.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def swap_used_gb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", nargs="*", default=[])
    ap.add_argument("--from-end", type=int, default=0, help="take this many steps from the END of the plan, last first")
    ap.add_argument("--max-swap-gb", type=float, default=0.0, help="do not start another step while swap use is above this (0 = no check)")
    ap.add_argument("--queue", action="store_true", help="work through the WHOLE plan, claiming each step atomically (several workers can share it)")
    ap.add_argument("--big-first", nargs="*", default=["old_eng_sft", "aya_interlanguage", "old_african_sft", "aya_collection", "aya_translate",
                                                       "sft_new_existing", "old_tasks", "old_summarization", "tds"])
    a = ap.parse_args()
    if not os.environ.get("HF_HOME"):
        sys.exit("set HF_HOME to a scratch folder first")
    from sft_new import combine as K
    from sft_new.dedup import Seen

    names = list(a.steps)
    if a.queue:
        full = K.plan(False)
        names += [n for n in a.big_first if n in full] + [n for n in full if n not in a.big_first]
    if a.from_end:
        names += list(reversed(K.plan(False)))[: a.from_end]
    stats_dir = a.out / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    guard = K.DiskGuard(a.out)
    seen = Seen(a.out / f"seen_{a.name}.sqlite")
    claimed: list[Path] = []
    try:
        for name in names:
            tag = K.safe(name)
            done = list(stats_dir.glob(f"{tag}*.json"))
            if done:
                print(f"{name}: already claimed or done", flush=True)
                continue
            if a.max_swap_gb and swap_used_gb() > a.max_swap_gb:
                print(f"swap use {swap_used_gb():.1f} GB is above {a.max_swap_gb} GB: worker {a.name} stops before {name}", flush=True)
                break
            ph = stats_dir / f"{tag}.json"
            if a.queue:                                                             # atomic claim: only one worker gets a step
                claim = stats_dir / f"{tag}.claim"
                try:
                    os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue
                claimed.append(claim)
            else:
                ph.write_text(json.dumps({"placeholder": True, "worker": a.name}))  # the main build skips a step whose stats file exists
                claimed.append(ph)
            for attempt in range(3):
                try:
                    K.run_one(name, a.out, seen, guard, None, False, (os.environ['HF_ORG_LEGACY'] + "/sft-all"))
                    claimed.remove(claimed[-1])
                    if a.queue:
                        (stats_dir / f"{tag}.claim").unlink(missing_ok=True)
                    break
                except RuntimeError:
                    raise
                except Exception as e:                       # network hiccup: retry the step from its start
                    print(f"{name}: attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
            else:                                            # three failures: release the claim so another worker (or a later run) can try it
                print(f"{name}: FAILED after 3 attempts", flush=True)
                (stats_dir / f"{tag}.claim").unlink(missing_ok=True)
                if claimed:
                    claimed.pop()
    finally:
        seen.close()
        for ph in claimed:                                   # an unfinished claim must not hide the step from the other workers / the main build
            if ph.exists() and (ph.suffix == ".claim" or json.loads(ph.read_text()).get("placeholder")):
                ph.unlink()


if __name__ == "__main__":
    main()
