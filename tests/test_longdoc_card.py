# SPDX-License-Identifier: AGPL-3.0-or-later
"""Composition tables of the benchmark card (longdoc_card)."""
from gemma_mt.eval import longdoc_card as C


def _r(source, domain, b, lic="CC-BY-4.0", cls="permissive", split="public"):
    return {"source": source, "domain": domain, "bin": b, "licence": lic, "licence_class": cls, "split": split}


def test_composition_counts_add_up_and_licences_listed():
    rows = [_r("wmt", "news", 64, "research", "unclear")] * 3 + [_r("govuk", "government_public", 512, "OGL-3.0")] * 2 + [_r("wikipedia", "encyclopedia_travel", 4096, "CC-BY-SA-4.0", "share_alike"),
            _r("wikipedia", "encyclopedia_travel", 256, "CC-BY-SA-4.0", "share_alike")]
    md = C.composition_markdown(rows)
    assert "| wmt | research | 3 | 0 | 0 | 0 | 3 | 42.9% |" in md and "| **all** |  | 3 | 1 | 2 | 1 | 7 | 100% |" in md
    assert "#### Source x domain" in md and "| **all** | encyclopedia_travel" not in md and "government_public" in md
    assert "| share_alike | 2 | 28.6% |" in md and "| unclear | 3 | 42.9% |" in md
    assert C.composition_markdown([]) == "(no documents)\n"


def test_funnel_markdown_orders_drops_first_and_keeps_candidates():
    md = C.funnel_markdown({"pool": 100, "drop_a": 5, "drop_b": 20, "candidates": 75, "candidates_bin64": 70, "paragraphs_removed": 9})
    lines = md.splitlines()
    assert lines[2].startswith("| pool") and lines[3].startswith("| drop_b") and lines[4].startswith("| drop_a") and lines[5].startswith("| candidates |") and "paragraphs_removed" in md
    assert C.split_counts([{"split": "private"}, {"split": "public"}, {}]) == {"private": 1, "public": 2}


def test_push_preflight_refuses_private_rows_missing_canary_and_tokens(tmp_path):
    import importlib.util
    import json
    import os

    spec = importlib.util.spec_from_file_location("push_hub", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "longdoc_push_hub.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    canary = P.CANARY
    st = tmp_path / "stage"
    (st / "core").mkdir(parents=True)
    (st / "README.md").write_text("# card\n" + canary + "\n")
    (st / "DATASHEET.md").write_text("# sheet\n")
    (st / "core" / "docs.jsonl").write_text(json.dumps({"doc_id": "a", "text": "x", "canary": canary, "split": "public"}) + "\n")
    got = P.preflight(str(st))
    assert got["core/docs.jsonl"]["rows"] == 1 and set(got) == {"README.md", "DATASHEET.md", "core/docs.jsonl"}
    (st / "core" / "docs.jsonl").write_text(json.dumps({"doc_id": "a", "text": "x", "canary": canary, "split": "private"}) + "\n")
    for bad, needle in (("private row", "private row"), ):
        try:
            P.preflight(str(st))
            raise AssertionError("not refused")
        except SystemExit as e:
            assert needle in str(e)
    (st / "core" / "docs.jsonl").write_text(json.dumps({"doc_id": "a", "text": "x", "split": "public"}) + "\n")
    try:
        P.preflight(str(st))
        raise AssertionError("not refused")
    except SystemExit as e:
        assert "without the canary" in str(e)
    (st / "core" / "docs.jsonl").write_text(json.dumps({"doc_id": "a", "text": "x", "canary": canary}) + "\n")
    (st / "notes.md").write_text("key hf_" + "A" * 30)
    try:
        P.preflight(str(st))
        raise AssertionError("not refused")
    except SystemExit as e:
        assert "token-like" in str(e)
    os.remove(st / "notes.md")
    (st / "private").mkdir()
    (st / "private" / "docs.jsonl").write_text("{}\n")
    try:
        P.preflight(str(st))
        raise AssertionError("not refused")
    except SystemExit as e:
        assert "private path" in str(e)
