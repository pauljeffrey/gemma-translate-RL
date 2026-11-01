# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests of the long-document contamination scanner (gemma_mt/eval/longdoc_contam.py, scripts/longdoc_contam_scan.py) on tiny synthetic corpora."""

import importlib.util
import json
import os
import random
import zlib

import pytest

np = pytest.importorskip("numpy")

from gemma_mt.eval import longdoc_contam as L   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = [f"w{i}x" for i in range(5000)]


def rand_words(rng, n):
    return [rng.choice(VOCAB) for _ in range(n)]


def fake_encode(texts):
    """A deterministic stand-in tokenizer: one token per whitespace-separated word, ids >= 1000 (never a layout token)."""
    return [[1000 + zlib.crc32(w.encode()) % 50000 for w in t.split()] for t in texts]


def record(i, words, paras=1):
    k = max(1, len(words) // paras)
    text = "\n\n".join(" ".join(words[j: j + k]) for j in range(0, len(words), k))
    return {"id": f"d{i}", "text": text}


# ------------------------------------------------------------------------------------------------ hashing
def test_normalisation_is_the_contamination_module_rule():
    from gemma_mt.eval.contamination import normalise
    assert normalise("Don't STOP—now!  ") == "don t stop now"
    assert normalise("Café No.5") == "café no 5"
    d = L.prepare_docs([{"id": "a", "text": "Don't STOP,\n\nnow!! Go—go."}])[0]
    assert d.n_words == 6 and d.para_bounds == [(0, 3), (3, 6)]
    assert [(0, 3), (3, 6)] == d.para_bounds


def test_mix64_numpy_equals_python():
    xs = [0, 1, 2, 262144, 2 ** 40 + 7, 2 ** 64 - 1]
    got = L.mix64(np.array(xs, dtype=np.uint64))
    assert [int(g) for g in got] == [L.mix64_int(x) for x in xs]


def test_rolling_hash_equals_naive_hash():
    rng = random.Random(0)
    toks = np.array([rng.randrange(0, 262144) for _ in range(300)], dtype=np.uint32)
    ids = L.token_ids64(toks)
    for n in (1, 2, 8, 12):
        h = L.rolling_hashes(ids, n)
        assert len(h) == len(toks) - n + 1
        for i in (0, 1, 17, len(h) - 1):
            assert int(h[i]) == L.naive_ngram_hash([L.mix64_int(int(t)) for t in toks[i: i + n]])
    assert L.rolling_hashes(ids[:5], 8).size == 0
    # word ids are stable (blake2b) and distinct for distinct words
    c = L.new_word_cache()
    a = L.word_ids64(["alpha", "beta", "alpha"], c)
    assert a[0] == a[2] and a[0] != a[1]


def test_region_hashes_never_cross_a_region():
    rng = random.Random(1)
    regions = [np.array([rng.randrange(1000) for _ in range(k)], dtype=np.uint32) for k in (20, 5, 9, 31, 8)]
    ids = L.token_ids64(np.concatenate(regions))
    n = 8
    got = L.region_ngram_hashes(ids, [len(r) for r in regions], n)
    want = np.concatenate([L.rolling_hashes(L.token_ids64(r), n) for r in regions])
    assert np.array_equal(got, want)
    assert list(L.counts_per_region([20, 5, 9, 31, 8], n)) == [13, 0, 2, 24, 1]


def test_longest_run_and_stats():
    b = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert L.longest_run(b) == 3 and L.longest_run(np.zeros(4, bool)) == 0 and L.longest_run(np.zeros(0, bool)) == 0
    assert L.run_units(3, 8) == 10 and L.run_units(0, 8) == 0
    s, r, k = L.hit_stats(b, 8)
    assert abs(s - 6 / 9) < 1e-9 and r == 10 and k == 6
    assert L.hit_stats(np.zeros(0, bool), 8) == (0.0, 0, 0)


def test_mark_hits_and_membership():
    uniq = np.array([3, 9, 20, 41], dtype=np.uint64)
    hit = np.zeros(4, dtype=np.uint8)
    n = L.mark_hits(uniq, hit, np.array([9, 9, 5, 41, 100], dtype=np.uint64), block=2)
    assert n == 3 and list(hit) == [0, 1, 0, 1]
    assert list(L.membership(uniq, np.array([3, 4, 100], dtype=np.uint64))) == [True, False, False]


# ------------------------------------------------------------------------------------------------ shares
def test_a_40_percent_copy_has_share_near_0_4_in_word_space():
    rng = random.Random(2)
    train = [" ".join(rand_words(rng, 600)) for _ in range(20)]
    win = train[7].split()
    doc_words = win[100:500] + rand_words(rng, 600)                   # 400 copied words + 600 fresh
    full = [record(0, win[100:500] + win[100:500][:0] + rand_words(rng, 600), paras=5)]
    neg = [record(1, rand_words(rng, 1000), paras=4)]
    docs = L.prepare_docs([record(2, doc_words, paras=1)] + full + neg)
    rows, info = L.scan_documents(docs, L.Sources(rl_texts=train), L.ScanConfig(processes=1))
    assert abs(rows[0]["rl_share"] - 393 / 993) < 0.01
    assert rows[0]["rl_longest_run"] == 400 and rows[1]["rl_longest_run"] == 400
    assert rows[2]["rl_share"] == 0.0
    # the same copy in a document that splits into 5 paragraphs: shares agree, the paragraphs fully inside the copy are verbatim
    assert abs(rows[1]["rl_share"] - 393 / 993) < 0.02
    assert rows[1]["paragraph_exact"] >= 1 and rows[2]["paragraph_exact"] == 0


def test_benchmark_hits_name_the_benchmark_and_join_neighbouring_rows():
    rng = random.Random(3)
    s1, s2, s3 = (rand_words(rng, 25) for _ in range(3))
    bench = {"flores": [" ".join(s1), " ".join(s2), " ".join(s3)], "other": [" ".join(rand_words(rng, 30))]}
    doc = rand_words(rng, 100) + s1 + s2 + rand_words(rng, 100)       # two consecutive rows copied: the join n-grams must hit too
    docs = L.prepare_docs([record(0, doc)])
    rows, _ = L.scan_documents(docs, L.Sources(bench=bench), L.ScanConfig(processes=1))
    r = rows[0]
    assert r["bench_hit_benchmarks"] == ["flores"] and r["bench_longest_run"] == 50
    assert r["bench_hit_ngrams"]["flores"] == 50 - 8 + 1


# ------------------------------------------------------------------------------------------------ the packed SFT stream
HEADER = [2, 105, 2364, 107, 40414, 506, 1816, 531, 99999, 236787, 138]
TAIL = [106, 107, 105, 4368, 107]


def build_packed(path, english_texts, corrupt=()):
    os.makedirs(path, exist_ok=True)
    toks, off, plen, slen, tlen = [], [], [], [], []
    for k, t in enumerate(english_texts):
        e = fake_encode([t])[0]
        tail = list(TAIL)
        if k in corrupt:
            tail[1] = 5
        prompt = HEADER + e + tail
        target = [7001, 7002, 7003, 106, 107]
        off.append(len(toks))
        toks += prompt + target
        plen.append(len(prompt))
        tlen.append(len(target) - 1)
        slen.append(len(prompt) + len(target))
    n = len(english_texts)
    np.save(os.path.join(path, "sample_offset.npy"), np.array(off, dtype=np.uint64))
    np.save(os.path.join(path, "prompt_len.npy"), np.array(plen, dtype=np.uint32))
    np.save(os.path.join(path, "sample_len.npy"), np.array(slen, dtype=np.uint32))
    np.save(os.path.join(path, "target_len.npy"), np.array(tlen, dtype=np.uint32))
    np.save(os.path.join(path, "iso_id.npy"), np.zeros(n, dtype=np.uint16))
    np.save(os.path.join(path, "direction.npy"), np.zeros(n, dtype=np.uint8))
    np.save(os.path.join(path, "pack_start.npy"), np.array([0, n], dtype=np.uint64))
    np.array(toks, dtype=np.uint32).tofile(os.path.join(path, "tokens.u32.bin"))
    return path


def test_english_bounds_find_the_english_and_reject_a_foreign_layout(tmp_path):
    rng = random.Random(4)
    texts = [" ".join(rand_words(rng, k)) for k in (30, 12, 45, 20)]
    sp = L.PackedSplit(build_packed(str(tmp_path / "train"), texts, corrupt=(2,)))
    toks, lengths, bad = sp.chunk(0, 4)
    assert bad == 1 and list(lengths) == [30, 12, 20]
    exp = np.concatenate([np.array(fake_encode([texts[i]])[0], dtype=np.uint32) for i in (0, 1, 3)])
    assert np.array_equal(toks, exp)
    allt, alll, _ = sp.chunk(0, 4, scope="all")
    assert alll.sum() == sp.sample_len.sum()


def test_sft_token_scan_and_word_scan_agree_on_a_partial_copy(tmp_path):
    rng = random.Random(5)
    train_texts = [" ".join(rand_words(rng, rng.randint(40, 400))) for _ in range(300)]
    build_packed(str(tmp_path / "train"), train_texts[:250])
    build_packed(str(tmp_path / "val"), train_texts[250:])
    win = train_texts[10].split()
    L_ = min(len(win), 200)
    copied = win[:L_]
    doc60 = copied + rand_words(rng, int(L_ * 2 / 3))                  # ~60% copied
    fresh = rand_words(rng, 300)
    recs = [record(0, doc60), record(1, fresh), record(2, train_texts[260].split())]      # the last one lives in the VAL split
    docs = L.prepare_docs(recs, fake_encode, 8, 12)
    src = L.Sources(sft_dirs=[str(tmp_path / "train"), str(tmp_path / "val")])
    rows, info = L.scan_documents(docs, src, L.ScanConfig(token_n=12, processes=1, sft_words=False))
    exp = (L_ - 12 + 1) / (len(doc60) - 12 + 1)
    assert abs(rows[0]["sft_share"] - exp) < 0.01, rows[0]
    assert rows[0]["sft_longest_run"] == L_ and rows[1]["sft_share"] == 0.0
    assert rows[2]["sft_share"] > 0.99 and rows[2]["sft_longest_run"] == len(train_texts[260].split())
    assert info["sft"]["invalid_samples"] == 0 and info["sft"]["regions"] == 300


def test_multiprocess_scan_equals_single_process(tmp_path):
    rng = random.Random(6)
    texts = [" ".join(rand_words(rng, rng.randint(30, 80))) for _ in range(400)]
    build_packed(str(tmp_path / "train"), texts)
    doc = texts[3].split() + rand_words(rng, 50) + texts[222].split()[:40]
    docs = L.prepare_docs([record(0, doc)], fake_encode)
    uniq = L.CandidateIndex([docs[0].tok_hash]).uniq
    h1, _ = L.scan_sft_stream([str(tmp_path / "train")], {12: uniq}, processes=1, chunk_samples=50, log=lambda m: None)
    h2, _ = L.scan_sft_stream([str(tmp_path / "train")], {12: uniq}, processes=3, chunk_samples=50, log=lambda m: None)
    assert np.array_equal(h1[12], h2[12]) and h1[12].sum() > 0


# ------------------------------------------------------------------------------------------------ duplicates
def test_duplicates_keep_the_first_and_ignore_boilerplate():
    rng = random.Random(7)
    base = rand_words(rng, 400)
    boiler = rand_words(rng, 40)
    hs = []
    for ws in (base + boiler, base[:300] + rand_words(rng, 200) + boiler, rand_words(rng, 400) + boiler, base + boiler):
        hs.append(L.rolling_hashes(L.word_ids64(ws), 8))
    rank = [0, 2, 3, 1]                      # doc 0 first, then doc 3 (an exact copy of 0), then 1, then 2
    dup_of, share = L.find_duplicates(hs, rank, 0.5, max_group=3)
    assert dup_of[0] == -1                   # the first one is kept
    assert dup_of[3] == 0 and share[3] > 0.85          # (the 4-document boilerplate windows are ignored)
    assert dup_of[1] in (0, 3) and 0.5 < share[1] < 0.75
    assert dup_of[2] == -1 and share[2] == 0.0       # shares only the boilerplate (4 documents > max_group=3 -> ignored)


def test_flag_rules():
    th = L.Thresholds()
    base = dict(sft_share=0.0, rl_share=0.0, bench_share=0.0, dup_share=0.0, sft_longest_run=0, rl_longest_run=0, bench_longest_run=0)
    assert L.flag_row(base, th)[0] == "clean"
    assert L.flag_row({**base, "rl_share": 0.019}, th)[0] == "clean"
    assert L.flag_row({**base, "rl_share": 0.02}, th)[0] == "weak"
    assert L.flag_row({**base, "sft_share": 0.0999}, th)[0] == "weak"
    assert L.flag_row({**base, "bench_share": 0.10}, th)[0] == "contaminated"
    assert L.flag_row({**base, "rl_longest_run": 50}, th)[0] == "contaminated"
    assert L.flag_row({**base, "rl_longest_run": 49}, th)[0] == "clean"
    assert L.flag_row({**base, "sft_longest_run": 64}, th)[0] == "contaminated"
    assert L.flag_row({**base, "dup_share": 0.5}, th)[0] == "contaminated"
    assert L.flag_row({**base, "dup_share": 0.3}, th)[0] == "contaminated"          # 0.3 >= weak_below 0.10
    assert L.flag_row({**base, "paragraph_exact": 3}, L.Thresholds(paragraph_exact_max=3))[0] == "contaminated"
    assert L.flag_row({**base, "paragraph_exact": 3}, th)[0] == "clean"
    # the thresholds are parameters
    assert L.flag_row({**base, "rl_share": 0.05}, L.Thresholds(clean_below=0.1, weak_below=0.3))[0] == "clean"


# ------------------------------------------------------------------------------------------------ the CLI
def test_cli_end_to_end_with_benchmarks_and_duplicates(tmp_path):
    spec = importlib.util.spec_from_file_location("longdoc_contam_scan", os.path.join(HERE, "..", "scripts", "longdoc_contam_scan.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    rng = random.Random(8)
    sent = rand_words(rng, 40)
    bdir = tmp_path / "bench" / "ntrex" / "NTREX-128"
    bdir.mkdir(parents=True)
    (bdir / "newstest2019-src.eng.txt").write_text(" ".join(sent) + "\n" + " ".join(rand_words(rng, 30)) + "\n")
    long_doc = rand_words(rng, 700)
    pool = tmp_path / "pool"
    pool.mkdir()
    with open(pool / "a.jsonl", "w") as f:
        f.write(json.dumps(record(0, rand_words(rng, 500))) + "\n")
        f.write(json.dumps(record(1, long_doc)) + "\n")
        f.write(json.dumps(record(2, long_doc[:200] + rand_words(rng, 200))) + "\n")
        f.write(json.dumps(record(3, rand_words(rng, 60) + sent + rand_words(rng, 100) + rand_words(rng, 200))) + "\n")
    with open(pool / "b.jsonl", "w") as f:
        f.write(json.dumps(record(4, long_doc[:600])) + "\n")          # a shorter copy of d1, in another file
    out = tmp_path / "out"
    argv = ["--pool-dir", str(pool), "--out-dir", str(out), "--bench-root", str(tmp_path / "bench"), "--processes", "1"]
    assert cli.main(argv) == 0
    rows = {r["id"]: r for r in L.iter_jsonl(str(out / "a.flags.jsonl"))}
    rows.update({r["id"]: r for r in L.iter_jsonl(str(out / "b.flags.jsonl"))})
    assert list(rows) == ["d0", "d1", "d2", "d3", "d4"]
    assert rows["d0"]["flag"] == "clean" and rows["d1"]["flag"] == "clean"
    assert rows["d3"]["bench_hit_benchmarks"] == ["ntrex"] and rows["d3"]["bench_longest_run"] == 40
    assert rows["d3"]["flag"] == "weak"                                # one 40-word benchmark sentence in a 400-word document: ~8% of its windows
    assert rows["d4"]["dup_of"] == "d1" and rows["d4"]["flag"] == "contaminated"      # the shorter copy loses
    assert rows["d2"]["dup_of"] == "" and 0.2 < rows["d2"]["dup_share"] < 0.5
    summ = json.load(open(out / "SUMMARY.json"))
    assert summ["files"]["a"]["n_docs"] == 4 and summ["files"]["b"]["flags"] == {"contaminated": 1}
    # cached scan: a second run changes nothing and does not rescan (cache files keep their mtime)
    cm = os.path.getmtime(out / "cache" / "a.scan.jsonl")
    assert cli.main(argv) == 0
    assert os.path.getmtime(out / "cache" / "a.scan.jsonl") == cm
    # a pool file that changes (a new document appended) is rescanned and its flags rewritten; the untouched file keeps its cache
    with open(pool / "b.jsonl", "a") as f:
        f.write(json.dumps(record(5, rand_words(rng, 300))) + "\n")
    assert cli.main(argv) == 0
    assert os.path.getmtime(out / "cache" / "a.scan.jsonl") == cm
    ids_b = [r["id"] for r in L.iter_jsonl(str(out / "b.flags.jsonl"))]
    assert ids_b == ["d4", "d5"]
