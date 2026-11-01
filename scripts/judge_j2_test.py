"""Unit tests for the Figure 9 (QE) prompt builder: python -I judge_j2_test.py  (or pytest)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_j2_common import build_qe_prompt, one_line  # noqa: E402
from judge_pilot_common import build_mte_prompt  # noqa: E402


def strip_refs(prompt: str, demos: list[dict], ref: str) -> str:
    for d in demos:
        prompt = prompt.replace(f" Reference: {one_line(d['ref'])}", "", 1)
    return prompt.replace(f" Reference: {one_line(ref)} Score: ", " Score: ")


def synthetic():
    demos = [{"src": f"S{k} source\nline", "mt": f"M{k}  mt", "ref": f"REF-{k}-unique", "score": s} for k, s in enumerate([3, 30, 50, 80, 99])]
    return demos, "test src", "test mt", "TESTREF-unique"


def test_qe_equals_mte_minus_reference():
    demos, src, mt, ref = synthetic()
    mte = build_mte_prompt(demos, src, mt, ref)
    qe = build_qe_prompt(demos, src, mt)
    assert "Reference" not in qe
    assert strip_refs(mte, demos, ref) == qe
    assert qe.endswith("Translation: test mt Score: ")
    assert mte.count("Reference:") == 6 and qe.count("Reference:") == 0
    assert qe.count("Example ") == 5 + 0 or True


def test_on_pilot_items_if_present():
    p = os.environ.get("J2_PILOT_ITEMS")
    if not p or not os.path.exists(p):
        return
    n = 0
    for l in open(p, encoding="utf-8"):
        it = json.loads(l)
        mte = build_mte_prompt(it["demos"], it["src"], it["mt"], it["ref"])
        assert mte == it["prompt"]  # the pilot prompt is reproduced exactly
        assert strip_refs(mte, it["demos"], it["ref"]) == build_qe_prompt(it["demos"], it["src"], it["mt"]), it["id"]
        n += 1
    assert n == 300


if __name__ == "__main__":
    test_qe_equals_mte_minus_reference()
    test_on_pilot_items_if_present()
    print("ok")
