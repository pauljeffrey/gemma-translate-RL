"""CPU tests of the post-GPU chain (run on the box: /workspace/work/venv/bin/python -m pytest chain/test_chain.py -q ; needs the tokenizers in /workspace/work/tok_*)."""
import json, os, subprocess, sys, tempfile
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
AFRO = os.environ.get("AFRO_TOK", "/workspace/work/tok_afro/tokenizer.json")
GEMMA = os.environ.get("GEMMA_TOK", "/workspace/work/tok_google/tokenizer.json")
need_tok = pytest.mark.skipif(not (os.path.exists(AFRO) and os.path.exists(GEMMA)), reason="tokenizers not available")


def pairs_dir(tmp, n=300):
    en = [f"This is sentence number {i}, a short one about the market." for i in range(n)]
    tg = [f"Nke a bu ahiri nke {i}, obere otu banyere ahia." for i in range(n)]
    d = os.path.join(tmp, "pairs"); os.makedirs(d)
    pq.write_table(pa.table({"english": en, "target": tg, "iso": ["ibo"] * n, "source_dataset": ["src/a"] * n, "licence": ["cc-by"] * n, "flipped": [False] * n}), d + "/pairs-0000.parquet")
    return d


@need_tok
def test_windows_bins_and_order(tmp_path):
    import windows
    d = pairs_dir(str(tmp_path))
    out = str(tmp_path / "w")
    windows.main(["--src", d, "--out", out, "--afro-tokenizer", AFRO, "--gemma-tokenizer", GEMMA])
    t = pq.read_table(out + "/windows-0000.parquet").to_pydict()
    assert sum(t["n_pairs"]) == 300
    # consecutive pairs of one run stay consecutive and in order on both sides
    seq = [int(x) for e in t["english"] for x in __import__("re").findall(r"sentence number (\d+)", e)]
    assert seq == list(range(300))
    seq_t = [int(x) for e in t["target"] for x in __import__("re").findall(r"nke (\d+)", e)]
    assert seq_t == list(range(300))
    assert all(max(a, b) <= 1025 for a, b in zip(t["src_afr_tokens"], t["tgt_afr_tokens"]))
    assert set(t["length_bin"]) <= {"b64", "b256", "b512", "b1024"}


@need_tok
def test_duplicates_across_dirs_removed(tmp_path):
    import windows
    d = pairs_dir(str(tmp_path)); d2 = str(tmp_path / "pairs2"); os.makedirs(d2)
    pq.write_table(pq.read_table(d + "/pairs-0000.parquet"), d2 + "/pairs-0000.parquet")
    windows.main(["--src", d, d2, "--out", str(tmp_path / "w"), "--stats-only", "--afro-tokenizer", AFRO])
    assert json.load(open(tmp_path / "w" / "stats.json"))["duplicate_across_dirs"] == 300


def test_stub_is_refused_by_gate(tmp_path):
    import qe_score, gate
    d = pairs_dir(str(tmp_path))
    qe_score.main(["--src", d, "--out", str(tmp_path / "s"), "--stub", "--chunk", "100"])
    assert os.path.isdir(str(tmp_path / "s_STUB")) and not os.path.exists(str(tmp_path / "s"))
    with pytest.raises(SystemExit):
        gate.load_scores(str(tmp_path / "s_STUB"), False)


def test_gate_rules(tmp_path):
    import gate
    rng = np.random.default_rng(0)
    n = 6000
    keys = np.arange(1, n + 1, dtype=np.uint64)
    q = np.clip(rng.normal(0.5, 0.12, n), 0, 1).astype(np.float32)
    d = tmp_path / "pairs"; d.mkdir()
    pq.write_table(pa.table({"english": ["a b c"] * n, "target": ["x y z"] * n, "iso": ["ibo"] * n, "key": pa.array(keys, pa.uint64())}), str(d / "pairs-0000.parquet"))
    s = tmp_path / "sc"; s.mkdir()
    pq.write_table(pa.table({"key": pa.array(keys, pa.uint64()), "qe": pa.array(q, pa.float32()), "scorer": ["m"] * n}), str(s / "x.parquet"))
    class A: pass
    a = A(); a.src = [str(d)]; a.scores = str(s); a.refs_scores = None; a.refs = None; a.out = str(tmp_path / "g"); a.allow_stub = False; a.min_refs = 30; a.min_rows = 2000
    gate.fit(a)
    th = json.load(open(tmp_path / "g" / "thresholds.json"))["languages"]["ibo"]
    assert abs(th["keep_rate_old"] - 0.85) < 0.01                      # 15th percentile
    assert 0.05 <= 1 - th["keep_rate_cal"] <= 0.40 + 1e-9              # floor 5%, cap 40%
    assert th["thr_old"] >= 0.20


def test_sampler_arms_same_passes(tmp_path):
    import sampler
    PD, ALL = sampler._pd()
    import csv
    p = tmp_path / "p.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "packs"])
        for iso, n in (("ibo", 4000), ("hau", 3000), ("amh", 800), ("fon", 300), ("yor", 2000), ("kea", 50)): w.writerow([iso, n])
    sampler.plan(type("A", (), dict(stage="A", packed_dir=None, packs_csv=str(p), out=str(tmp_path / "A0"), seed=1, steps=0, per_step=0, alpha=0.5, budget_tokens=0))())
    sampler.arms(type("A", (), dict(schedule=str(tmp_path / "A0"), out=str(tmp_path)))())
    a0 = np.load(tmp_path / "A0" / "schedule.npy").reshape(-1); pl = np.load(tmp_path / "A0" / "pack_lang.npy")
    s1 = np.load(tmp_path / "S1" / "schedule.npy").reshape(-1)
    spec = [ALL.index(i) for i in ("amh", "fon", "yor", "kea")]
    exp = a0[np.isin(pl[a0], spec)]
    assert np.array_equal(exp[:len(s1)], s1) and len(exp) - len(s1) < 32          # same draws, same order


# ----------------------------------------------------------------------------- commercial-use rule (Amendment 4): NC rows never reach windows / scores / gate
def _fake_tok(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"[UNK]": 0}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace(); t.save(path)


def _pairs_nc(tmp, name, rows):
    d = os.path.join(tmp, name); os.makedirs(d)
    pq.write_table(pa.table({"english": [r[0] for r in rows], "target": [r[1] for r in rows], "iso": [r[2] for r in rows], "source_dataset": [r[3] for r in rows], "licence": [r[4] for r in rows],
                             "flipped": [False] * len(rows)}), d + "/pairs-0000.parquet")
    return d


def test_windows_drop_nc_before_dedupe(tmp_path):
    import windows
    tk = str(tmp_path / "tok.json"); _fake_tok(tk)
    shared = ("The harvest was good this year in the village.", "Girbi ya yi kyau a bana a kauye.", "hau")
    d1 = _pairs_nc(str(tmp_path), "specialist", [(*shared, "r10:opus", "non_commercial"), ("A second NC sentence about water.", "Jumla ta biyu game da ruwa.", "hau", "r10:opus", "non_commercial")])
    d2 = _pairs_nc(str(tmp_path), "data", [(*shared, "src/a", "cc-by"), ("A commercial sentence about rain.", "Jumla ta kasuwanci game da ruwan sama.", "hau", "src/a", "cc-by")])
    out = str(tmp_path / "w")
    windows.main(["--src", d1, d2, "--out", out, "--stats-only", "--afro-tokenizer", tk, "--oversample", ""])
    st = json.load(open(out + "/stats.json"))
    assert st["dropped_nc"] == 2 and st["dropped_nc_by_iso"] == {"hau": 2} and st["pairs"] == 2        # the commercial copy of the shared pair survives although the NC copy came first
    import nc_filter
    nc_filter.guard_windows(type("A", (), dict(windows=out))())
    windows.main(["--src", d1, d2, "--out", str(tmp_path / "w2"), "--stats-only", "--afro-tokenizer", tk, "--oversample", "", "--include-nc"])
    with pytest.raises(SystemExit):
        nc_filter.guard_windows(type("A", (), dict(windows=str(tmp_path / "w2"))))


def test_quarantine_dir_refused(tmp_path):
    import common
    with pytest.raises(SystemExit):
        common.assert_clean_src(["/x/quarantine_nc"])


def test_gate_excludes_nc(tmp_path):
    import gate
    n = 3000
    keys = np.arange(1, n + 1, dtype=np.uint64)
    q = np.concatenate([np.full(1000, 0.10), np.linspace(0.3, 0.9, 2000)]).astype(np.float32)          # the first 1000 rows are NC with terrible scores
    lic = ["non_commercial"] * 1000 + ["cc-by"] * 2000
    d = tmp_path / "pairs"; d.mkdir()
    pq.write_table(pa.table({"english": ["a b c"] * n, "target": ["x y z"] * n, "iso": ["amh"] * n, "licence": lic, "key": pa.array(keys, pa.uint64())}), str(d / "pairs-0000.parquet"))
    s = tmp_path / "sc"; s.mkdir()
    pq.write_table(pa.table({"key": pa.array(keys, pa.uint64()), "qe": pa.array(q, pa.float32()), "scorer": ["m"] * n}), str(s / "x.parquet"))
    class A: pass
    a = A(); a.src = [str(d)]; a.scores = [str(s)]; a.refs_scores = None; a.refs = None; a.out = str(tmp_path / "g"); a.allow_stub = False; a.min_refs = 30; a.min_rows = 0; a.exclude_nc = True; a.totals = None
    gate.fit(a)
    th = json.load(open(tmp_path / "g" / "thresholds.json"))["languages"]["amh"]
    assert th["n_scored"] == 2000 and th["thr_old"] > 0.3          # NC rows do not pull the 15th percentile down
    a.rule = "old"; gate.apply_(a)
    keep = set(pq.read_table(str(tmp_path / "g" / "keep_old.parquet"))["key"].to_pylist())
    assert not (keep & set(range(1, 1001)))


def test_allow_list_keeps_ibom_tagged(tmp_path):
    import windows, nc_filter
    tk = str(tmp_path / "tok.json"); _fake_tok(tk)
    d1 = _pairs_nc(str(tmp_path), "specialist", [("An NC sentence about water.", "Jumla ta biyu game da ruwa.", "amh", "masakhane/AfriDocMT", "non_commercial")])
    d2 = _pairs_nc(str(tmp_path), "data", [("An ibom sentence about the river.", "Ofon ibom jumla game da kogi.", "ibb", "howard-nlp/ibom-mt", "non_commercial"), ("A commercial sentence about rain.", "Jumla ta kasuwanci game da ruwan sama.", "hau", "src/a", "cc-by")])
    out = str(tmp_path / "w")
    windows.main(["--src", d1, d2, "--out", out, "--afro-tokenizer", tk, "--oversample", "", "--nc-allow", "howard-nlp/ibom-mt"])
    st = json.load(open(out + "/stats.json"))
    assert st["dropped_nc"] == 1 and st["dropped_nc_by_iso"] == {"amh": 1} and st["nc_exception_pairs"] == {"ibom-mt|ibb": 1} and st["pairs"] == 2
    t_ = pq.read_table(out + "/windows-0000.parquet").to_pydict()
    assert {(i, e) for i, e in zip(t_["iso"], t_["nc_exception"])} == {("ibb", "ibom-mt"), ("hau", "")}
    nc_filter.guard_windows(type("A", (), dict(windows=out))())                  # tagged NC windows pass the guard
    import common
    assert common.parse_nc_allow(["howard-nlp/ibom-mt"]) == {"howard-nlp/ibom-mt": "ibom-mt"}


def test_sampler_select_subset_commands(tmp_path):
    import sampler, csv
    PD, ALL = sampler._pd()
    p = tmp_path / "p.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "packs"])
        for iso, n in (("ibo", 4000), ("amh", 800), ("tir", 500), ("fon", 300), ("yor", 2000)): w.writerow([iso, n])
    sampler.plan(type("A", (), dict(stage="A", packed_dir=None, packs_csv=str(p), out=str(tmp_path / "A0"), seed=1, steps=0, per_step=0, alpha=0.5, budget_tokens=0, seen_from=None))())
    sampler.select(type("A", (), dict(schedule=str(tmp_path / "A0"), out=str(tmp_path / "G0"), languages="amh,tir", name="G0"))())
    sampler.subset(type("A", (), dict(schedule=str(tmp_path / "A0"), out=str(tmp_path / "A2"), frac=0.25, stratify="language", seed=1234, languages=None, name="A2"))())
    a0 = np.load(tmp_path / "A0" / "schedule.npy").reshape(-1); pl = np.load(tmp_path / "A0" / "pack_lang.npy")
    s = np.load(tmp_path / "A2" / "schedule.npy").reshape(-1)
    for l in np.unique(pl[a0]):
        assert abs(int((pl[s] == l).sum()) / int((pl[a0] == l).sum()) - 0.25) < 0.05


def test_nc_hidden_rules(tmp_path):
    import nc_hidden as NH
    rows = []
    for i in range(30):
        rows.append((f"Sentence number {i} about the market and the river.", f"Jumla ta {i} game da kasuwa da kogi.", "hau", "UBC-NLP/AfroLingu-MT", "unspecified", ""))
    rows.append(("A row with a hidden NC licence text.", "Jumla mai lasisin boye.", "hau", "x/y", "cc-by", "cc-by-nc-sa-4.0 upstream"))
    d = _pairs_nc(str(tmp_path), "data", [(r[0], r[1], r[2], r[3], r[4]) for r in rows])
    t = pq.read_table(d + "/pairs-0000.parquet").to_pydict(); t["licence_raw"] = [r[5] for r in rows]
    pq.write_table(pa.table(t), d + "/pairs-0000.parquet")
    hv = np.unique(np.array([NH.h("hau", NH.nm(rows[0][0]), NH.nm(rows[0][1])), NH.h("hau", NH.nm(rows[1][0]), NH.nm(rows[1][1]))], np.uint64))
    idx = str(tmp_path / "idx.npz"); np.savez(idx, labels=np.array(["mafand-mt(CC-BY-NC)"]), s0=hv)
    rep = str(tmp_path / "rep")
    NH.report(type("A", (), dict(src=[d], index=idx, cpt=None, nc_allow=[], drop_source=[], min_pairs=2000, out=rep))())
    summ = json.load(open(rep + "/summary.json"))
    assert summ["R1_pairs"] == 1 and summ["R2_pairs"] == 2                                  # one by licence text, two by an NC-origin pair match
    assert any(x[0] == "hau" for x in summ["languages_below_min_after_default_rules"])
    csvt = open(rep + "/nc_hidden_by_language_source.csv").read()
    assert "UBC-NLP/AfroLingu-MT" in csvt and "KEPT by owner" in csvt                                 # whole-source candidate listed, not dropped
    with pytest.raises(SystemExit):                                                          # hau would end below 2,000: refuse without --allow-below-min
        NH.apply_(type("A", (), dict(src=[d], index=idx, cpt=None, nc_allow=[], drop_source=[], min_pairs=2000, out=str(tmp_path / "k.parquet"), allow_below_min=False))())
    NH.apply_(type("A", (), dict(src=[d], index=idx, cpt=None, nc_allow=[], drop_source=[], min_pairs=2000, out=str(tmp_path / "k.parquet"), allow_below_min=True))())
    assert len(pq.read_table(str(tmp_path / "k.parquet"))) == 3


def test_hidden_merge_keep(tmp_path):
    import nc_hidden as NH
    pq.write_table(pa.table({"key": pa.array([1, 2, 3, 4, 5], pa.uint64())}), str(tmp_path / "keep.parquet"))
    pq.write_table(pa.table({"key": pa.array([2, 5, 99], pa.uint64())}), str(tmp_path / "drop.parquet"))
    NH.merge_keep(type("A", (), dict(keep=str(tmp_path / "keep.parquet"), drop=str(tmp_path / "drop.parquet"), out=str(tmp_path / "final.parquet")))())
    assert sorted(pq.read_table(str(tmp_path / "final.parquet"))["key"].to_pylist()) == [1, 3, 4]


def test_qe_probe_stub(tmp_path):
    import subprocess, sys
    d = tmp_path / "pairs"; d.mkdir()
    n = 3000
    pq.write_table(pa.table({"english": [f"This is sentence {i} about the market." for i in range(n)], "target": [f"Nke a bu ahiri {i}." for i in range(n)], "iso": ["ibo"] * n,
                             "source_dataset": ["s"] * n, "licence": ["cc-by"] * n}), str(d / "pairs-0000.parquet"))
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "qe_probe.py"), "--pairs", str(d), "--out", str(tmp_path / "probe"), "--procs", "1", "2", "--n", "300", "--stub", "--python", sys.executable], capture_output=True, text=True)
    assert "BEST procs=" in r.stdout, r.stdout + r.stderr


# ----------------------------------------------------------------------------------------------------------- Amendment 6: extra sources (nde ful knc tzm), owner tags, 24 specialist languages
def test_extra_parsers(tmp_path):
    import extra_sources as X
    (tmp_path / "n.txt").write_text("GEN 1:1 Ekuqaleni uNkulunkulu wadala amazulu lomhlaba.\nGEN 1:2 Umhlaba wawungelasimo njalo.\nGEN 1:3 \nEXO 2:1 Indoda yasuka.\n", encoding="utf-8")
    (tmp_path / "e.txt").write_text("GEN 1:1 In the beginning, God created the heavens and the earth.\nGEN 1:2 The earth was formless and empty.\nGEN 1:3 God said, let there be light.\n", encoding="utf-8")
    r = list(X.rows_nde(str(tmp_path / "n.txt"), str(tmp_path / "e.txt")))
    assert [x["target"][:9] for x in r] == ["Ekuqaleni", "Umhlaba w"] and all(x["iso"] == "nde" and x["licence"] == "share_alike" for x in r)     # empty verse and verse without an English side are not paired
    (tmp_path / "g.tsv").write_text('eng\tkau\nIs that your "new" friend?\tAde sawanem belin wa?\nSecond one\t\n', encoding="utf-8")
    r = list(X.rows_tsv(str(tmp_path / "g.tsv"), "eng", "kau", "knc", "extra-twb:x", "cc-by", "x", "human", "g"))
    assert len(r) == 2 and r[0]["english"].startswith("Is that") and '"new"' in r[0]["english"]
    (tmp_path / "p.tsv").write_text("source_string\ttarget_string\n%1$s has a new update\t%1$s aleqqem\nOpen the file\tXdem afaylu\n", encoding="utf-8")
    r = list(X.rows_tsv(str(tmp_path / "p.tsv"), "source_string", "target_string", "tzm", "extra-pontoon:x", "mpl", "x", "ui_strings", "p", drop_placeholders=True))
    assert [x["english"] for x in r] == ["Open the file"]
    pq.write_table(pa.table({"similarity": [1.2, 1.06, 1.051], "English": ["a b c d", "e f g h", "i j k l"], "Fulah": ["x", "y", "z"]}), str(tmp_path / "m.parquet"))
    r = list(X.rows_parquet_pairs(str(tmp_path / "m.parquet"), "English", "Fulah", "ful", "extra-nllb_mined:x", "unspecified", "x", "mined", "m", sim_col="similarity", min_sim=1.06))
    assert [x["aux"] for x in r] == pytest.approx([1.2, 1.06], rel=1e-5)          # rows below the margin are dropped before anything else
    assert X.ok("The quick brown fox jumps over the lazy dog.", "Kuru man karuwu sorn adoma") and not X.ok("Hi", "Hi") and not X.ok("See https://x.org now please", "Kuru man karuwu sorn")


def test_extra_pipeline_dedupe_filter_and_schema(tmp_path, monkeypatch):
    import extra_sources as X
    from common import key_of
    sys.path.insert(0, os.path.dirname(HERE))
    import decon as D
    protected = "The committee approved the budget for the new hospital in the northern region yesterday."
    b = D.Builder(); b.add("benchX", protected); ix = b.finish()
    f = X.Filter.__new__(X.Filter); f.ix = ix; f.version = "protected-test"
    en_ok = ["The farmers sold their maize at the market today.", "She walked to the river to fetch some water.", protected, "Another clean sentence about the school."]
    def fake(name, tmp, a):
        if name == "ful_wikimedia":
            for e, t in zip(en_ok, ["Gooto ɓeydi ndiyam", "O yahi to maayo", "Komiti nde sanjii nafaqa ngam suudu ñaw nde keso ley bannge hooreejo hannde", "Dow ekkol ndee"]):
                yield X.row(e, t, "ful", "extra-opus:wikimedia-v20260327-en-ff", "share_alike", "x", "human", "w")
            yield X.row("short", "x", "ful", "extra-opus:wikimedia-v20260327-en-ff", "share_alike", "x", "human", "w")          # fails ok()
        elif name == "ful_nllb_mined":                                     # same pair as the first wikimedia row (first source keeps it) + one new pair + the protected English side with another Fulah side
            yield X.row(en_ok[0], "Gooto ɓeydi ndiyam", "ful", "extra-nllb_mined:y", "unspecified", "x", "mined", "m", aux=1.2)
            yield X.row("The teacher read a long story to the children.", "Jannginoowo ɗum jaŋngi tinndinoore", "ful", "extra-nllb_mined:y", "unspecified", "x", "mined", "m", aux=1.1)
            yield X.row(protected, "Komiti ɗum sanjii nafaqa ngam suudu ñaw keso ley bannge hooreejo hannde", "ful", "extra-nllb_mined:y", "unspecified", "x", "mined", "m", aux=1.1)
    monkeypatch.setattr(X, "fetch_rows", fake)
    a = type("A", (), dict(tmp=str(tmp_path / "t")))()
    stats, removed, kept, errors = X.process(["ful_wikimedia", "ful_nllb_mined"], a, f, only_iso={"ful"})
    assert not errors
    tg = sorted(r["target"] for r in kept)
    assert not any(t.startswith("Komiti") for t in tg) and len(kept) == 4          # protected English side removed from both sources, duplicate and too-short rows gone
    assert sum(1 for r in kept if r["source_dataset"].startswith("extra-nllb_mined")) == 1
    assert all(r["key"] == key_of(r["english"], r["iso"], r["target"]) for r in kept)
    X.write_outputs(str(tmp_path / "out"), stats, removed, kept, errors, "protected-test", False)
    t = pq.read_table(str(tmp_path / "out" / "pairs-0000.parquet"))
    assert t.schema.field("key").type == pa.uint64() and set(X.SCHEMA_COLS) <= set(t.column_names)
    s = json.load(open(tmp_path / "out" / "stats" / "summary.json"))
    assert s["kept_by_language"] == {"ful": 4} and s["protected_index_version"] == "protected-test"
    assert sum(1 for line in open(tmp_path / "out" / "stats" / "removed_by_protected_set.csv")) == 3          # header + 2 removals (one per source)


def test_owner_tags_and_24_specialist_languages():
    from common import owner_tag, SPECIALIST, SPECIALIST_20, NEW_LANGS
    assert owner_tag("UBC-NLP/AfroLingu-MT") == "afrolingu-mt" and owner_tag("weak-repo:afrolingu-mt") == "afrolingu-mt"
    assert owner_tag("extra-nllb_mined:michsethowusu/english-fulah_sentence-pairs") == "nllb_mined" and owner_tag("extra-bible:ebible-nde+engwebp") == "bible_derived"
    assert owner_tag("extra-twb:CLEAR-Global/Gamayun-kits/kau-eng") == "twb_gamayun_terms" and owner_tag("michsethowusu/english-hausa_sentence-pairs_mt560") == ""
    assert len(SPECIALIST_20) == 20 and SPECIALIST[:20] == SPECIALIST_20 and SPECIALIST[20:] == ["nde", "ful", "knc", "tzm"] == NEW_LANGS


def test_new_languages_registered_and_ids_stable():
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo, "gemma_mt"))
    try:
        from gemma_mt import languages as L
    except Exception as e:                                                  # pragma: no cover
        pytest.skip(f"gemma_mt not importable here: {e}")
    for iso, name in (("nde", "North Ndebele"), ("ful", "Fulah"), ("knc", "Kanuri"), ("tzm", "Tamazight")):
        assert iso in L.ALL_ISOS and L.target_tag_and_name(iso) == (iso, name) and not L.LANGUAGES[iso].rl_eligible          # registered, SFT-only (no RL pools, no reward calibration)
    assert L.ALL_ISOS[-2:] == ("tzm", "ful") and L.iso_to_id("ful") == 84 and L.iso_to_id("nde") == 35          # the ids that existing packs / schedules use do not move


@need_tok
def test_windows_owner_exception_column(tmp_path):
    import windows
    n = 60
    d = tmp_path / "pairs"; d.mkdir()
    pq.write_table(pa.table({"english": [f"This is sentence number {i} about the weather." for i in range(n)], "target": [f"Kuru man karuwu sorn {i} ade ma kerma." for i in range(n)], "iso": ["ful"] * n,
                             "source_dataset": ["extra-nllb_mined:y"] * n, "licence": ["unspecified"] * n, "flipped": [False] * n}), str(d / "pairs-0000.parquet"))
    windows.main(["--src", str(d), "--out", str(tmp_path / "w"), "--afro-tokenizer", AFRO, "--gemma-tokenizer", GEMMA])
    t = pq.read_table(str(tmp_path / "w" / "windows-0000.parquet")).to_pydict()
    assert set(t["owner_exception"]) == {"nllb_mined"}
    assert any(k.startswith("nllb_mined|ful") for k in json.load(open(tmp_path / "w" / "stats.json"))["owner_exception_windows"])


# ----------------------------------------------------------------------------------------------------------- QE cap (monotone hash sample) and the fast scorer
def _cap_pairs(tmp, name="pairs", sizes=(("ibo", 6000), ("hau", 3000), ("bam", 40)), nc=None):
    from common import key_of
    d = os.path.join(str(tmp), name); os.makedirs(d, exist_ok=True)
    en, tg, iso, lic = [], [], [], []
    for l, n in sizes:
        for i in range(n):
            en.append(f"Sentence {i} of {l} about the market and the river."); tg.append(f"Target {l} number {i} gamma delta."); iso.append(l); lic.append("cc-by")
    if nc:
        for i in range(nc):
            en.append(f"Non commercial sentence {i} here."); tg.append(f"Nc target {i} here too."); iso.append("amh"); lic.append("non_commercial")
    keys = [key_of(e, i, t) for e, i, t in zip(en, iso, tg)]
    pq.write_table(pa.table({"english": en, "target": tg, "iso": iso, "licence": lic, "source_dataset": ["s/x"] * len(en), "key": pa.array(keys, pa.uint64())}), d + "/pairs-0000.parquet")
    return d


def test_cap_is_monotone_and_sized(tmp_path):
    from common import cap_counts, cap_thresholds, in_cap
    d = _cap_pairs(tmp_path, nc=500)
    counts = cap_counts([d], True, set())
    assert counts == {"ibo": 6000, "hau": 3000, "bam": 40}                      # NC rows are not scoreable, so not counted
    keys = pq.read_table(d + "/pairs-0000.parquet", columns=["key", "iso"]).to_pydict()
    sel = {n: {(k, i) for k, i in zip(keys["key"], keys["iso"]) if in_cap(k, i, cap_thresholds(counts, n))} for n in (500, 1000, 2000, 5000)}
    assert sel[500] <= sel[1000] <= sel[2000] <= sel[5000]                        # a larger cap keeps a SUPERSET
    got = collections_counter([i for _, i in sel[1000]])
    assert abs(got["ibo"] - 1000) < 120 and abs(got["hau"] - 1000) < 120 and got["bam"] == 40            # about N per language, thin languages in full
    assert cap_thresholds(counts, 0) is None and in_cap(5, "ibo", None)


def collections_counter(xs):
    import collections
    return collections.Counter(xs)


def test_qe_score_cap_and_resume_scores_only_new_keys(tmp_path):
    d = _cap_pairs(tmp_path)
    cc = str(tmp_path / "cap_counts.json")
    subprocess.run([sys.executable, os.path.join(HERE, "qe_cap.py"), "count", "--src", d, "--out", cc], check=True, capture_output=True)
    out = str(tmp_path / "sc")
    def run(n):
        r = subprocess.run([sys.executable, os.path.join(HERE, "qe_score.py"), "--src", d, "--out", out, "--stub", "--cap-per-lang", str(n), "--cap-counts", cc, "--skip-scored", out + "_STUB", "--chunk", "400"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        return r.stdout
    import pyarrow.dataset as pads
    run(500)
    k1 = set(pads.dataset(out + "_STUB", format="parquet").to_table(columns=["key"])["key"].to_pylist())
    run(1500)
    k2 = set(pads.dataset(out + "_STUB", format="parquet").to_table(columns=["key"])["key"].to_pylist())
    assert k1 < k2 and 2500 < len(k2) < 3500 and 900 < len(k1) < 1300           # ~500+500+40, then ~1500+1500+40
    n_files = len([f for f in os.listdir(out + "_STUB") if f.endswith(".parquet")])
    run(1500)                                                                     # nothing new to score
    assert len([f for f in os.listdir(out + "_STUB") if f.endswith(".parquet")]) == n_files


def test_gate_reports_before_cap_and_unscored_is_not_kept(tmp_path):
    import gate
    d = _cap_pairs(tmp_path, sizes=(("ibo", 5000),))
    keys = np.array(pq.read_table(d + "/pairs-0000.parquet", columns=["key"])["key"].to_pylist(), np.uint64)
    scored = keys[:3000]                                                          # 2,000 pairs were never scored
    s = tmp_path / "sc"; s.mkdir()
    q = np.linspace(0.3, 0.9, 3000).astype(np.float32)
    pq.write_table(pa.table({"key": pa.array(scored, pa.uint64()), "qe": pa.array(q), "scorer": ["m"] * 3000}), str(s / "x.parquet"))
    class A: pass
    a = A(); a.src = [d]; a.scores = str(s); a.refs_scores = None; a.refs = None; a.out = str(tmp_path / "g"); a.allow_stub = False; a.min_refs = 30; a.min_rows = 2000
    gate.fit(a)
    rows = list(__import__("csv").DictReader(open(tmp_path / "g" / "keep_rate_per_language.csv")))
    assert int(rows[0]["n_pairs_before_cap"]) == 5000 and int(rows[0]["n_scored"]) == 3000
    a.rule = "old"; gate.apply_(a)
    keep = set(pq.read_table(tmp_path / "g" / "keep_old.parquet")["key"].to_pylist())
    assert keep <= set(int(x) for x in scored) and len(keep) > 2000               # the 2,000 never-scored pairs are NOT kept
    r = list(__import__("csv").DictReader(open(tmp_path / "g" / "gated_old_summary.csv")))[0]
    assert (int(r["pairs_before_cap"]), int(r["scored_after_cap"])) == (5000, 3000) and int(r["kept_after_gate"]) == len(keep)


def test_fast_qe_matches_per_pair_scores_and_caches_sources():
    torch = pytest.importorskip("torch")
    import qe_fast
    torch.manual_seed(0)
    DIM = 16
    class Tok:
        pad_token_id = 0
        def __call__(self, texts, truncation=True, max_length=510, add_special_tokens=True, return_attention_mask=False):
            return {"input_ids": [[1] + [2 + (ord(c) % 50) for c in t][:max_length] + [3] for t in texts]}
    class Enc:
        tokenizer = Tok(); max_positions = 512; output_units = DIM
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.encoder = Enc(); self.emb = torch.nn.Embedding(64, DIM); self.estimator = torch.nn.Sequential(torch.nn.Linear(4 * DIM, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        def get_sentence_embedding(self, ids, mask):            # padding-independent mean pooling, like comet's masked average
            e = self.emb(ids) * mask.unsqueeze(-1); return e.sum(1) / mask.sum(1, keepdim=True)
    m = M().eval()
    rng = np.random.default_rng(1)
    srcs = ["".join(chr(97 + int(x)) for x in rng.integers(0, 26, rng.integers(3, 120))) for _ in range(60)]
    srcs = srcs + srcs[:20]                                      # repeated English sources
    mts = ["".join(chr(97 + int(x)) for x in rng.integers(0, 26, rng.integers(3, 200))) for _ in range(80)]
    fq = qe_fast.FastQE(m, "cpu", False, token_budget=300, max_batch=7, cache_gb=0.01)
    got = fq.score(srcs, mts)
    def one(s, t):
        a = m.get_sentence_embedding(*(lambda ids: (torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long)))(Tok()([t])["input_ids"][0]))
        b = m.get_sentence_embedding(*(lambda ids: (torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long)))(Tok()([s])["input_ids"][0]))
        return float(m.estimator(torch.cat((a, b, a * b, torch.abs(a - b)), 1)))
    with torch.no_grad():
        ref = np.array([one(s, t) for s, t in zip(srcs, mts)], np.float32)
    assert np.abs(got - ref).max() < 1e-5 and np.corrcoef(got, ref)[0, 1] > 0.99999
    assert fq.stats["src_unique_chunk"] == 60 and fq.stats["src_encoded"] == 60
    got2 = fq.score(srcs, mts)                                   # every source is now a cache hit
    assert np.abs(got2 - ref).max() < 1e-5 and fq.stats["src_cache_hits"] == 60 and fq.stats["src_encoded"] == 60
