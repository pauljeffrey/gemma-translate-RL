#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Initialise the dedicated language-tag rows of a (grown) checkpoint: row of `<unused{i}>` := mean embedding of the sub-tokens of the old text tag.

    python scripts/init_tag_rows.py --model ckpt/gemma3-up6-g9 --out ckpt/gemma3-up6-g9-tags
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from gemma_mt.growth import init_tag_rows
    t = init_tag_rows(a.model, a.out)
    up = os.path.join(a.model, "upcycle.json")
    if os.path.exists(up):
        info = json.load(open(up)); info["tag_rows_initialised"] = len(t)
        json.dump(info, open(os.path.join(a.out, "upcycle.json"), "w"), indent=1)
    print(f"initialised {len(t)} tag rows -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
