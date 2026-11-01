"""End-to-end tests on DUMMY data for the Gemini Batch pipeline (no network, no key).  Run: python -m pytest paper/scripts/test_gemini_batch.py"""
import csv
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemini_batch_common as C  # noqa: E402
import gemini_batch_cost as COST  # noqa: E402
import gemini_batch_ingest as ING  # noqa: E402
import gemini_batch_requests as REQ  # noqa: E402
import gemini_count_tokens as CNT  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PRICES = os.path.join(REPO, "reports", "data", "longctx", "gemini_prices_2026-10-23.json")


def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture()
def bench(tmp_path):
    docs, items = [], []
    # 12 core docs in b512 (to exercise sampling), 1 sm human doc, 1 sci_dev doc
    for b in ("b512", "b1024", "b2048"):
        for k in range(8):
            sents = ["Sentence %d of doc %d in %s." % (j, k, b) + " Extra words here." * (j + 1) for j in range(5)]
            docs.append(dict(doc_id="d%d#%s" % (k, b), set="core", bin=b, text="\n\n".join([" ".join(sents[:2]), " ".join(sents[2:])]),
                             sents=sents, para_starts=[0, 2], src_tokens=50))
            for iso in C.HUMAN_LANGS:
                items.append(dict(iid="core|d%d#%s|%s" % (k, b, iso), doc_id="d%d#%s" % (k, b), iso=iso, set="core", bin=b))
    sm = ["Desserts are nice.", "They can be sweet.", "Some are salty.", "Cake is common."]
    docs.append(dict(doc_id="G1:sm1", set="human", bin="sm", text=" ".join(sm), sents=sm, para_starts=[0], src_tokens=20))
    for iso in C.HUMAN_LANGS[:3]:
        items.append(dict(iid="hum|G1:sm1|%s" % iso, doc_id="G1:sm1", iso=iso, set="human", bin="sm"))
    sci = ["Hepatitis B is studied.", "Mutations escape vaccines.", "We review the evidence."]
    docs.append(dict(doc_id="sci_dev:1", set="sci", bin="sci_dev", text=" ".join(sci), sents=sci, para_starts=[0], src_tokens=15))
    for iso in C.SCI_LANGS:
        items.append(dict(iid="sci|sci_dev:1|%s" % iso, doc_id="sci_dev:1", iso=iso, set="sci", bin="sci_dev"))
    dp, ip = str(tmp_path / "docs.jsonl"), str(tmp_path / "items.jsonl")
    wjl(dp, docs)
    wjl(ip, items)
    return dp, ip, docs, items


def run_req(bench, out, *extra):
    dp, ip = bench[0], bench[1]
    return REQ.main(["--docs", dp, "--items", ip, "--out-dir", str(out), "--model", "gemini-2.5-flash-lite", *extra])


def test_whole_requests_schema_and_manifest(bench, tmp_path):
    out = tmp_path / "req"
    m = run_req(bench, out, "--mode", "whole", "--langs", "hau,amh", "--sets", "core", "--thinking-budget", "0")
    assert m["requests"] == 48
    line = json.loads(open(out / "requests_01.jsonl", encoding="utf-8").readline())
    assert set(line) == {"key", "request"}
    r = line["request"]
    assert r["contents"][0]["parts"][0]["text"].startswith("Sentence 0")
    assert "\n\n" in r["contents"][0]["parts"][0]["text"]            # paragraph break preserved in the source
    sysm = r["system_instruction"]["parts"][0]["text"]
    assert ("Hausa" in sysm) or ("Amharic" in sysm)
    assert r["generation_config"]["temperature"] == 0
    assert r["generation_config"]["thinking_config"] == {"thinking_budget": 0}
    assert r["generation_config"]["max_output_tokens"] >= 1024
    assert line["key"] in {i["iid"] for i in bench[3]}
    # SHA256 manifest verifies
    man = json.load(open(out / "manifest.json"))
    for f in man["files"]:
        assert hashlib.sha256(open(out / f["name"], "rb").read()).hexdigest() == f["sha256"]
    assert "requests_01.jsonl" in open(out / "SHA256SUMS").read()
    # no secrets / key material anywhere
    assert "AIza" not in open(out / "requests_01.jsonl").read()


def test_camel_casing(bench, tmp_path):
    out = tmp_path / "req"
    run_req(bench, out, "--mode", "whole", "--langs", "hau", "--sets", "human", "--casing", "camel", "--thinking-level", "low")
    r = json.loads(open(out / "requests_01.jsonl").readline())["request"]
    assert "systemInstruction" in r and "generationConfig" in r
    assert r["generationConfig"]["maxOutputTokens"] > 0 and r["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


def test_chunk_mode_keys_and_index(bench, tmp_path):
    out = tmp_path / "req"
    run_req(bench, out, "--mode", "chunk", "--chunk-size", "2", "--langs", "hau", "--sets", "core", "--bins", "b512")
    idx = [json.loads(l) for l in open(out / "requests_index.jsonl")]
    assert len(idx) == 8 * 3                      # 5 sentences -> chunks of 2,2,1
    assert idx[0]["key"].endswith("@@c0000") and C.split_key(idx[0]["key"])[1] == 0
    first = [r for r in idx if r["iid"] == idx[0]["iid"]]
    assert [(r["lo"], r["hi"]) for r in first] == [(0, 2), (2, 4), (4, 5)]
    assert first[1]["para_break_before"] == [True, False]          # para_starts = [0, 2]
    req = json.loads(open(out / "requests_01.jsonl").readline())
    assert req["request"]["contents"][0]["parts"][0]["text"].count("\n") == 1
    assert "exactly 2 line(s)" in req["request"]["system_instruction"]["parts"][0]["text"]


def test_split_files(bench, tmp_path):
    out = tmp_path / "req"
    m = run_req(bench, out, "--mode", "whole", "--langs", "hau,swh", "--sets", "core", "--max-requests-per-file", "10")
    assert len(m["files"]) == 5 and sum(f["requests"] for f in m["files"]) == 48
    m2 = run_req(bench, tmp_path / "req2", "--mode", "whole", "--langs", "hau", "--sets", "core", "--max-file-bytes", "3000")
    assert len(m2["files"]) > 1 and all(f["bytes"] <= 3000 for f in m2["files"])


def test_reduced_selection_deterministic(bench):
    items = bench[3]
    a = C.select_items(items, "reduced")
    b = C.select_items(items, "reduced")
    assert [i["iid"] for i in a] == [i["iid"] for i in b]
    core = [i for i in a if i["set"] == "core"]
    assert len(core) == 5 * 3 * len(C.REDUCED_LANGS)
    assert {i["iso"] for i in core} == set(C.REDUCED_LANGS)
    assert len([i for i in a if i["set"] == "sci"]) == 6
    assert C.select_items(items, "reduced", seed=1) != a


def test_pilot(bench, tmp_path):
    out = tmp_path / "pil"
    m = run_req(bench, out, "--pilot", "3", "--langs", "hau,yor")
    assert m["mode"] == "pilot" and m["requests"] == 6
    keys = [json.loads(l)["key"] for l in open(out / "requests_01.jsonl")]
    assert all(k.startswith("pilot|G1:sm1|") for k in keys)


# ------------------------------------------------------------------ ingestion -------------------------------------------------
def resp(text, finish="STOP", prompt=100, cand=50, thoughts=0, thought_part=None, camel=True):
    parts = ([{"text": thought_part, "thought": True}] if thought_part else []) + [{"text": text}]
    um = dict(promptTokenCount=prompt, candidatesTokenCount=cand, thoughtsTokenCount=thoughts, totalTokenCount=prompt + cand + thoughts)
    return {"candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": finish}], "usageMetadata": um}


def test_ingest_end_to_end(bench, tmp_path):
    out = tmp_path / "req"
    run_req(bench, out, "--mode", "whole", "--langs", "hau", "--sets", "core", "--bins", "b512")
    keys = [json.loads(l)["key"] for l in open(out / "requests_01.jsonl")]
    assert len(keys) == 8
    lines = [
        {"key": keys[0], "response": resp("Ga fassara.\n\nAya ta biyu.", prompt=1000, cand=2000, thoughts=500)},
        {"key": keys[1], "response": resp("Here is the translation:\n```\nSaƙo na ɗaya.\n\nSaƙo na biyu.\n```")},
        {"key": keys[2], "response": {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": {"promptTokenCount": 70}}},
        {"key": keys[3], "response": resp("Rabin fassara", finish="MAX_TOKENS", cand=8192)},
        {"key": keys[4], "error": {"code": 500, "message": "internal"}},
        {"key": keys[5], "response": resp("", finish="STOP")},
        {"key": keys[6], "response": resp("Amsa", thought_part="thinking about it...", thoughts=300)},
        # keys[7] is missing from the results on purpose
    ]
    rp = tmp_path / "res.jsonl"
    wjl(rp, lines)
    rows, rep = ING.ingest([str(rp)], str(out / "requests_index.jsonl"), str(tmp_path / "o"), PRICES, "gemini-2.5-flash-lite", "batch")
    by = {r["iid"]: r for r in rows}
    st = {k: by[k]["status"] for k in keys}
    assert [st[k] for k in keys] == ["ok", "ok", "blocked", "truncated", "error", "empty", "ok", "missing"]
    assert by[keys[0]]["hyp_text"] == "Ga fassara.\n\nAya ta biyu."
    assert by[keys[1]]["hyp_text"] == "Saƙo na ɗaya.\n\nSaƙo na biyu."          # preface line and fences stripped
    assert by[keys[3]]["hyp_text"] == "Rabin fassara" and by[keys[3]]["finish"] == "MAX_TOKENS"
    assert by[keys[6]]["hyp_text"] == "Amsa"                                      # thought part skipped
    assert by[keys[2]]["hyp_text"] == "" and by[keys[2]]["finish"].startswith("BLOCK")
    assert rep["status_counts"] == {"ok": 3, "blocked": 1, "truncated": 1, "error": 1, "empty": 1, "missing": 1}
    u = rep["usage_total"]
    assert u["prompt"] == 1000 + 100 + 70 + 100 + 100 + 100 and u["thoughts"] == 800
    p = json.load(open(PRICES))["models"]["gemini-2.5-flash-lite"]["batch"]
    exp = (u["prompt"] * p["in"] + (u["candidates"] + u["thoughts"]) * p["out"]) / 1e6
    assert rep["actual_usd"] == pytest.approx(exp, abs=1e-4)
    written = [json.loads(l) for l in open(tmp_path / "o" / "pseudo_gemini.jsonl")]
    assert len(written) == 8 and set(written[0]) >= {"iid", "hyp_text", "finish", "usage"}


def test_ingest_chunked_reassembly_and_mismatch(bench, tmp_path):
    out = tmp_path / "req"
    run_req(bench, out, "--mode", "chunk", "--chunk-size", "2", "--langs", "hau", "--sets", "core", "--bins", "b512")
    idx = [json.loads(l) for l in open(out / "requests_index.jsonl")]
    iid = idx[0]["iid"]
    mine = [r for r in idx if r["iid"] == iid]
    results = [
        {"key": mine[2]["key"], "response": resp("S5")},                          # out of order on purpose
        {"key": mine[0]["key"], "response": resp("S1\nS2")},
        {"key": mine[1]["key"], "response": resp("S3\n\nS4\nS4b")},               # 3 lines for 2 sentences -> line_mismatch
    ]
    for r in idx:
        if r["iid"] != iid:
            results.append({"key": r["key"], "response": resp("a\nb" if r["hi"] - r["lo"] == 2 else "a")})
    rp = tmp_path / "res.jsonl"
    wjl(rp, results)
    rows, rep = ING.ingest([str(rp)], str(out / "requests_index.jsonl"), str(tmp_path / "o"))
    r0 = {r["iid"]: r for r in rows}[iid]
    assert r0["status"] == "line_mismatch" and r0["n_chunks"] == 3
    assert len(r0["hyp_sents"]) == 5 and r0["hyp_sents"][:2] == ["S1", "S2"] and r0["hyp_sents"][4] == "S5"
    assert r0["hyp_text"] == "S1 S2\n\nS3 S4 S4b S5"
    assert "\n\n" in r0["hyp_text"]                                            # paragraph break before sentence 2 (para_starts [0,2])
    others = [r for r in rows if r["iid"] != iid]
    assert all(r["status"] == "ok" for r in others)


def test_clean_text_edge_cases():
    assert ING.clean_text("```text\nAbc\n```") == "Abc"
    assert ING.clean_text("Sure! Here's the translation:\nAbc\n\ndef") == "Abc\n\ndef"
    assert ING.clean_text("Ina so: wannan ba preface ba") == "Ina so: wannan ba preface ba"


# ------------------------------------------------------------------ counting + cost ------------------------------------------------
def test_count_and_cost_pipeline(bench, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    dp, ip = bench[0], bench[1]
    os.replace(dp, work / "docs.jsonl")
    os.replace(ip, work / "items.jsonl")
    sys.argv = ["x", "--work", str(work), "--out", str(tmp_path / "counts"), "--no-tokenizer"]
    CNT.main()
    items = COST.load_items(str(tmp_path / "counts" / "gemini_counts_items.csv"))
    meta = COST.load_meta(str(tmp_path / "counts" / "gemini_counts_meta.csv"))
    assert len(items) == 3 * 8 * 12 + 3 + 6
    prices = json.load(open(PRICES))
    rows, sel = COST.build(items, meta, prices, "current", "chars4")
    idx = {(r["kind"], r["selection"], r["mode"], r["model"], r["tier"], r["thinking"], r["mult"]): r for r in rows}
    # hand recomputation for one cell: reduced_core / whole / 2.5 flash-lite / batch / off / 2.0
    s = sel["reduced_core"]
    assert len(s) == 5 * 3 * 8
    src = sum(r["chars4"] for r in s)
    inp = src + len(s) * (meta["instr_tokens_whole_mean"] + meta["framing_tokens_assumed"])
    p = prices["models"]["gemini-2.5-flash-lite"]["batch"]
    exp = (inp * p["in"] + src * 2.0 * p["out"]) / 1e6
    got = idx[("main", "reduced_core", "whole", "gemini-2.5-flash-lite", "batch", "off/default", 2.0)]["usd_total"]
    assert got == pytest.approx(exp, abs=1e-4)
    # batch is exactly half of standard (same tokens), chunking costs more than whole, hidden thinking raises the bill
    std = idx[("main", "reduced_core", "whole", "gemini-2.5-flash", "standard", "off", 2.75)]["usd_total"]
    bat = idx[("main", "reduced_core", "whole", "gemini-2.5-flash", "batch", "off", 2.75)]["usd_total"]
    assert std == pytest.approx(2 * bat, rel=1e-3)
    c1 = idx[("main", "reduced_core", "chunk1", "gemini-2.5-flash", "batch", "off", 2.75)]["usd_total"]
    assert c1 > bat
    assert idx[("main", "reduced_core", "whole", "gemini-2.5-flash", "batch", "default", 2.75)]["usd_total"] > bat
    md = COST.render_md(rows, sel, meta, prices, "current", "chars4")
    assert "REDUCED core" in md and "50-sentence pilot" in md
    # pilot rows exist
    assert any(r["kind"] == "pilot" for r in rows)


def test_cost_reproduces_committed_csv(tmp_path):
    """The committed cost CSV must be exactly what the script computes from the committed counts + prices."""
    committed = os.path.join(REPO, "reports", "data", "longctx", "gemini_cost_table.csv")
    counts = os.path.join(REPO, "reports", "data", "longctx", "gemini_counts_items.csv")
    if not (os.path.exists(committed) and os.path.exists(counts)):
        pytest.skip("real counts not present")
    items = COST.load_items(counts)
    meta = COST.load_meta(os.path.join(os.path.dirname(counts), "gemini_counts_meta.csv"))
    rows, _ = COST.build(items, meta, json.load(open(PRICES)), "current", "chars4")
    with open(committed, newline="") as f:
        old = list(csv.DictReader(f))
    assert len(old) == len(rows)
    for a, b in zip(old, rows):
        assert float(a["usd_total"]) == pytest.approx(b["usd_total"], abs=1e-4)
