#!/usr/bin/env python3
"""Extra SFT sources for the four languages added by Amendment 6 (2026-10-30): nde ful knc tzm. Runs ON THE INSTANCE before windows/QE (run_chain.sh step 14).

    python extra_sources.py --out $HUB/extra --index work/protected_index/protected_index.npz --manifest work/protected_index/protected_manifest.json --tmp work/_extra_dl
    python extra_sources.py --estimate --sent-hashes protected_sentence_hashes.npz --out /tmp/extra_est        # laptop: counts BEFORE the strict filter + an R1-only (exact sentence) estimate

What it does: downloads each source into --tmp (raw copies are deleted as soon as they are read; the 1.5 M Fulah set is streamed by row group and filtered on the fly), standardises it to the pair
schema (english, target, iso, source_dataset, licence, licence_raw, kind, source_family, flipped, key, aux), cleans (whitespace, the length / ratio / letter rules of the old builder), removes exact
duplicates (key = xxh64(english.casefold, iso, target.casefold), the id used everywhere in the chain; first source in the priority order below keeps the row) and applies the STRICT decontamination
filter of the clean rebuild (protected index protected-2026-10-28.3, rules R1 exact, R2 contiguous, R3 shared 8-gram, BOTH sides, any language). The output directory is a normal chain input directory
(put it LAST in --src so that rows already in data/ or specialist/ keep their licence class). Stats: stats/per_language_source.csv, stats/removed_by_protected_set.csv, stats/summary.json,
stats/nde_crosscheck.json (nde only).

Sources (priority order inside a language; licence class = what the uploader / OPUS states; `tag` = disclosure tag of Amendment 6, written to windows as owner_exception):
  nde  1 eBible nde (Biblica, CC BY-SA 4.0) aligned by verse id to the public-domain eBible engwebp (World English Bible)     share_alike  tag bible_derived
  ful  1 OPUS wikimedia v20260327 en-ff (CC BY-SA)                                    2 OPUS XLEnt v1.2 en-ff (Wikipedia-derived entities / phrases)
       3 stephanedonna/english_fulfulde (Adamawa Fulfulde `fub` Bible vs KJV, no licence stated)   unspecified  tag bible_derived (dialect: Adamawa, kept under `ful` by owner decision)
       4 michsethowusu/english-fulah_sentence-pairs (NLLB-mined, 1.51 M pairs with a LASER margin score `similarity`; streamed, rows below --ful-min-sim dropped; owner-approved)   unspecified  tag nllb_mined
  knc  1 CLEAR-Global/Gamayun-kits kau-eng kit5k (TWB community translations of Tatoeba sentences; CC-BY-4.0 under the TWB Gamayun terms)    cc-by  tag twb_gamayun_terms
       2 IsahMBukar/kanurimt full_dataset.csv (7,513 pairs; compiles TWB Gamayun, TICO-19 (protected) and Tatoeba; kept only after the strict filter)   cc-by  tag kanurimt_compilation
  tzm  1 Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight eng_Latn-tzm_Tfng.csv (6,193 human-translated Wikipedia sentences, Tifinagh; CC-BY-SA-4.0)   share_alike  tag nllb_seed
       2 OPUS translatewiki v2026-07-01 en-tzm (CC-BY 3.0 / mixed; UI strings, placeholders dropped)   3 Tamazight-NLP/Pontoon-Translations en-tzm.tsv (MPL-2.0; Latin script, placeholders dropped)
NOT used: fat, gaa (non-commercial Ghana NLP sets), the NLLB-mined tzm set (owner has not approved it), GATITOS, SalomonMetre13/nde_en (cross-check only), silver sets.
"""
import argparse, collections, csv, io, json, os, re, shutil, sys, time, unicodedata, urllib.request, zipfile

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from common import key_of

LANGS = ("nde", "ful", "knc", "tzm")             # default of --languages; "efi" (owner-provided Efik NT sheet, 2026-10-30) is added only when asked for: --languages efi --efi-nt-tsv FILE
FUL_MIN_SIM = 1.06
OPUS = {"wikimedia": "https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses/en-ff.txt.zip",
        "xlent": "https://object.pouta.csc.fi/OPUS-XLEnt/v1.2/moses/en-ff.txt.zip",
        "translatewiki_tzm": "https://object.pouta.csc.fi/OPUS-translatewiki/v2026-07-01/moses/en-tzm.txt.zip"}
EBIBLE = {"nde": "https://ebible.org/Scriptures/nde_vpl.zip", "eng": "https://ebible.org/Scriptures/engwebp_vpl.zip"}
URL = re.compile(r"https?://|www\.", re.I)
PLACEHOLDER = re.compile(r"\{\{|\}\}|\$\d|%\d*\$?[sd]|%[sd]|\{[0-9a-z_]+\}|<[a-z/][^>]*>|&[a-z]+;", re.I)
SCHEMA_COLS = ["english", "target", "iso", "source_dataset", "licence", "licence_raw", "kind", "source_family", "flipped", "key", "aux"]


def norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def ok(en, tg):
    """the length / ratio / letter rules of the old builder (build_pairs.ok)"""
    if not (3 <= len(en) <= 1500 and 3 <= len(tg) <= 2500):
        return False
    r = len(tg) / max(1, len(en))
    if not (0.25 <= r <= 4.5):
        return False
    if en.casefold() == tg.casefold() or URL.search(en) or URL.search(tg):
        return False
    letters = sum(c.isalpha() for c in en)
    if letters / len(en) < 0.4 or not re.search(r"[A-Za-z]{2}", en):
        return False
    return sum(c.isalpha() for c in tg) / len(tg) >= 0.35


def row(en, tg, iso, source, lic, lic_raw, kind, family, aux=float("nan")):
    return dict(english=en, target=tg, iso=iso, source_dataset=source, licence=lic, licence_raw=lic_raw, kind=kind, source_family=family, flipped=False, aux=aux)


# ------------------------------------------------------------------------------------------------------------------------------------ parsers (take local files; tested on fixtures)
def parse_vpl(path):
    """eBible verse-per-line text: `GEN 1:1 text` -> {verse id: text}"""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.match(r"^(\S+ \d+:\d+)\s+(.*)$", line)
        if m and m.group(2).strip():
            out[m.group(1)] = m.group(2).strip()
    return out


def rows_nde(nde_vpl, eng_vpl):
    n, e = parse_vpl(nde_vpl), parse_vpl(eng_vpl)
    for vid, tg in n.items():
        if vid in e:
            yield row(e[vid], tg, "nde", "extra-bible:ebible-nde+engwebp", "share_alike", "CC BY-SA 4.0 (Biblica, eBible nde) aligned by verse id to engwebp (public domain)", "bible_aligned", "ebible-nde")


def parse_moses(en_path, tg_path):
    with open(en_path, encoding="utf-8") as a, open(tg_path, encoding="utf-8") as b:
        for x, y in zip(a, b):
            yield x.rstrip("\n"), y.rstrip("\n")


def rows_opus(en_path, tg_path, iso, source, lic, lic_raw, kind, family, drop_placeholders=False):
    for en, tg in parse_moses(en_path, tg_path):
        if drop_placeholders and (PLACEHOLDER.search(en) or PLACEHOLDER.search(tg)):
            continue
        yield row(en, tg, iso, source, lic, lic_raw, kind, family)


def rows_parquet_pairs(path, en_col, tg_col, iso, source, lic, lic_raw, kind, family, sim_col=None, min_sim=None, batch=50000):
    """stream a parquet of pairs by row group batches (the 1.5 M Fulah set is never held in memory); rows with similarity < min_sim are dropped before anything else"""
    pf = pq.ParquetFile(path)
    cols = [en_col, tg_col] + ([sim_col] if sim_col else [])
    for rb in pf.iter_batches(batch_size=batch, columns=cols):
        d = rb.to_pydict()
        sims = d[sim_col] if sim_col else [float("nan")] * len(d[en_col])
        for en, tg, s in zip(d[en_col], d[tg_col], sims):
            if sim_col and min_sim is not None and (s is None or s < min_sim):
                continue
            if en is None or tg is None:
                continue
            yield row(en, tg, iso, source, lic, lic_raw, kind, family, aux=float(s) if s is not None else float("nan"))


def rows_tsv(path, en_col, tg_col, iso, source, lic, lic_raw, kind, family, delimiter="\t", drop_placeholders=False):
    csv.field_size_limit(10 ** 8)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE if delimiter == "\t" else csv.QUOTE_MINIMAL):
            en, tg = r.get(en_col), r.get(tg_col)
            if en is None or tg is None:
                continue
            if drop_placeholders and (PLACEHOLDER.search(en) or PLACEHOLDER.search(tg)):
                continue
            yield row(en, tg, iso, source, lic, lic_raw, kind, family)


# ------------------------------------------------------------------------------------------------------------------------------------ downloads
def http_get(url, dst, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sabiyarn-extra-sources"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            return dst
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def hf_get(repo, fn, tmp):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fn, repo_type="dataset", token=os.environ.get("HF_TOKEN"), local_dir=os.path.join(tmp, repo.replace("/", "__")))


def unzip_to(zpath, d):
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(d)
    return d


def find(d, suffix):
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(suffix):
                return os.path.join(r, f)
    raise FileNotFoundError(f"{suffix} in {d}")


def fetch_rows(name, tmp, a):
    """generator of raw rows of one source; raw files are removed when the generator is exhausted"""
    d = os.path.join(tmp, name); os.makedirs(d, exist_ok=True)
    try:
        if name == "nde":
            n = unzip_to(http_get(EBIBLE["nde"], d + "/nde.zip"), d + "/n"); e = unzip_to(http_get(EBIBLE["eng"], d + "/eng.zip"), d + "/e")
            yield from rows_nde(find(n, "_vpl.txt"), find(e, "_vpl.txt"))
        elif name == "efi_nt":
            yield from rows_tsv(a.efi_nt_tsv, "english", "efik", "efi", "extra-bible:owner-efik-nt-xlsx", "unspecified", "owner-provided spreadsheet EFIK_7K+.xlsx (Efik New Testament verses vs KJV English); provenance and licence not confirmed; disclosure", "bible_aligned", "efik_nt_xlsx")
        elif name == "ful_wikimedia":
            z = unzip_to(http_get(OPUS["wikimedia"], d + "/w.zip"), d + "/z")
            yield from rows_opus(find(z, ".en-ff.en"), find(z, ".en-ff.ff"), "ful", "extra-opus:wikimedia-v20260327-en-ff", "share_alike", "CC BY-SA (Wikimedia content translation, OPUS)", "human", "opus-wikimedia")
        elif name == "ful_xlent":
            z = unzip_to(http_get(OPUS["xlent"], d + "/x.zip"), d + "/z")
            yield from rows_opus(find(z, ".en-ff.en"), find(z, ".en-ff.ff"), "ful", "extra-opus:XLEnt-v1.2-en-ff", "share_alike", "CC BY-SA (XLEnt, Wikipedia-derived)", "entity_phrases", "opus-xlent")
        elif name == "ful_fub_bible":
            for split in ("train", "validation", "test"):          # the uploader's three splits are all Bible verses (30,802 pairs in total); no benchmark use
                p = hf_get("stephanedonna/english_fulfulde", f"data/{split}-00000-of-00001.parquet", d)
                yield from rows_parquet_pairs(p, "source", "target", "ful", "extra-bible:stephanedonna/english_fulfulde", "unspecified", "no licence stated (Adamawa Fulfulde Bible vs KJV); disclosure", "bible_aligned", "fub-bible")
        elif name == "ful_nllb_mined":
            p = hf_get("michsethowusu/english-fulah_sentence-pairs", "data/train-00000-of-00001.parquet", d)
            yield from rows_parquet_pairs(p, "English", "Fulah", "ful", "extra-nllb_mined:michsethowusu/english-fulah_sentence-pairs", "unspecified", "NLLB-mined (owner-approved 2026-10-30), LASER margin in aux", "mined", "nllb-mined-ful",
                                          sim_col="similarity", min_sim=a.ful_min_sim)
        elif name == "knc_gamayun":
            p = hf_get("CLEAR-Global/Gamayun-kits", "parallel/kau-eng/kit5k.tsv", d)
            yield from rows_tsv(p, "eng", "kau", "knc", "extra-twb:CLEAR-Global/Gamayun-kits/kau-eng", "cc-by", "CC-BY-4.0 under the TWB Gamayun terms (derivatives must be open and free)", "human", "gamayun-kau-eng")
        elif name == "knc_kanurimt":
            p = hf_get("IsahMBukar/kanurimt", "data/full_dataset.csv", d)
            yield from rows_tsv(p, "source", "target", "knc", "extra-kanurimt:IsahMBukar/kanurimt", "cc-by", "CC-BY-4.0 (compilation of TWB Gamayun, TICO-19, Tatoeba; each source's terms apply)", "human", "kanurimt", delimiter=",")
        elif name == "tzm_seed":
            p = hf_get("Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight", "eng_Latn-tzm_Tfng.csv", d)
            yield from rows_tsv(p, "source_sentence", "target_sentence", "tzm", "extra-nllb_seed:Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight", "share_alike", "CC-BY-SA-4.0 (NLLB-Seed, human translations of Wikipedia sentences)", "human", "nllb-seed-tzm", delimiter=",")
        elif name == "tzm_translatewiki":
            z = unzip_to(http_get(OPUS["translatewiki_tzm"], d + "/t.zip"), d + "/z")
            yield from rows_opus(find(z, ".en-tzm.en"), find(z, ".en-tzm.tzm"), "tzm", "extra-opus:translatewiki-v2026-07-01-en-tzm", "cc-by", "CC-BY-3.0 (OPUS translatewiki; mixed upstream licences)", "ui_strings", "opus-translatewiki", drop_placeholders=True)
        elif name == "tzm_pontoon":
            p = hf_get("Tamazight-NLP/Pontoon-Translations", "en-tzm.tsv", d)
            yield from rows_tsv(p, "source_string", "target_string", "tzm", "extra-pontoon:Tamazight-NLP/Pontoon-Translations", "mpl", "MPL-2.0 (Mozilla Pontoon)", "ui_strings", "pontoon-tzm", drop_placeholders=True)
        else:
            raise KeyError(name)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# priority order = order of the list (first occurrence of a pair keeps its row)
SOURCES = ["efi_nt", "nde", "ful_wikimedia", "ful_xlent", "ful_fub_bible", "ful_nllb_mined", "knc_gamayun", "knc_kanurimt", "tzm_seed", "tzm_translatewiki", "tzm_pontoon"]
TAGS = {"efi_nt": "bible_derived", "nde": "bible_derived", "ful_fub_bible": "bible_derived", "ful_nllb_mined": "nllb_mined", "knc_gamayun": "twb_gamayun_terms", "knc_kanurimt": "kanurimt_compilation", "tzm_seed": "nllb_seed"}


# ------------------------------------------------------------------------------------------------------------------------------------ strict filter
class Filter:
    """STRICT protected-set filter (decon.py semantics): a pair is dropped when EITHER side has ANY hit (R1 exact sentence, R2 contiguous, R3 shared 8-gram)"""

    def __init__(self, index, manifest):
        import protected_index_loader as PL
        self.ix, self.man = PL.load(index, manifest)
        self.version = self.man.get("version")

    def check(self, rows):
        ra = self.ix.check_texts([r["english"] for r in rows]); rb = self.ix.check_texts([r["target"] for r in rows])
        out = []
        for x, y in zip(ra, rb):
            hit = None
            for side, z in (("english", x), ("target", y)):
                for rule, m in zip(("R1", "R2", "R3"), z[:3]):
                    if m and hit is None:
                        hit = f"{rule}_{side}"
            out.append(hit)
        return out


class R1Estimate:
    """laptop estimate: exact-sentence (R1-like) match of either side against protected_sentence_hashes.npz (full text and sentence parts). Not the full strict rule (no R2/R3)."""

    def __init__(self, path):
        import decon as D
        z = np.load(path); self.keys = np.sort(z["s_keys"]); self.D = D

    def _hit(self, text):
        D = self.D
        hs = []
        for sm, n in D.variants(text):
            if n:
                hs.append(D.sent_hash(n))
                hs += [D.sent_hash(p) for p in D.parts_of(text, sm) if p]
        if not hs:
            return False
        q = np.array(hs, np.uint64); pos = np.searchsorted(self.keys, q); pos[pos >= len(self.keys)] = len(self.keys) - 1
        return bool((self.keys[pos] == q).any())

    def check(self, rows):
        return ["R1est" if (self._hit(r["english"]) or self._hit(r["target"])) else None for r in rows]


# ------------------------------------------------------------------------------------------------------------------------------------ main
def process(sources, a, filt, only_iso=None):
    stats = collections.defaultdict(lambda: collections.Counter())
    removed = collections.Counter()
    seen = set(); kept_rows = []; errors = {}
    for name in sources:
        src_iso = "efi" if name.startswith("efi") else "ful" if name.startswith("ful") else ("knc" if name.startswith("knc") else ("tzm" if name.startswith("tzm") else "nde"))
        if only_iso and src_iso not in only_iso:
            continue
        t0 = time.time(); buf = []; n_raw = 0
        try:
            def flush(buf):
                if not buf:
                    return
                res = filt.check(buf) if filt else [None] * len(buf)
                for r, h in zip(buf, res):
                    k = (r["iso"], r["source_dataset"])
                    if h:
                        stats[k]["removed_protected"] += 1; removed[(r["iso"], r["source_dataset"], h)] += 1
                    else:
                        stats[k]["kept"] += 1; kept_rows.append(r)
            for r in fetch_rows(name, a.tmp, a):
                k = (r["iso"], r["source_dataset"]); stats[k]["raw_rows"] += 1; n_raw += 1
                en, tg = norm_ws(unicodedata.normalize("NFC", r["english"])), norm_ws(unicodedata.normalize("NFC", r["target"]))
                if not ok(en, tg):
                    stats[k]["dropped_clean"] += 1; continue
                kk = key_of(en, r["iso"], tg)
                if kk in seen:
                    stats[k]["dropped_duplicate"] += 1; continue
                seen.add(kk); r["english"], r["target"], r["key"] = en, tg, kk; stats[k]["after_dedupe"] += 1
                buf.append(r)
                if len(buf) >= 5000:
                    flush(buf); buf = []
            flush(buf)
            print(f"[{name}] {n_raw:,} raw rows, {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            errors[name] = repr(e)[:300]; print(f"[{name}] FAILED: {errors[name]}", flush=True)
    return stats, removed, kept_rows, errors


def name_of(source_dataset):
    return {"extra-bible:owner-efik-nt-xlsx": "efi_nt", "extra-bible:ebible-nde+engwebp": "nde", "extra-opus:wikimedia-v20260327-en-ff": "ful_wikimedia", "extra-opus:XLEnt-v1.2-en-ff": "ful_xlent", "extra-bible:stephanedonna/english_fulfulde": "ful_fub_bible",
            "extra-nllb_mined:michsethowusu/english-fulah_sentence-pairs": "ful_nllb_mined", "extra-twb:CLEAR-Global/Gamayun-kits/kau-eng": "knc_gamayun", "extra-kanurimt:IsahMBukar/kanurimt": "knc_kanurimt",
            "extra-nllb_seed:Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight": "tzm_seed", "extra-opus:translatewiki-v2026-07-01-en-tzm": "tzm_translatewiki", "extra-pontoon:Tamazight-NLP/Pontoon-Translations": "tzm_pontoon"}.get(source_dataset, "")


def write_outputs(out, stats, removed, kept_rows, errors, version, estimate):
    os.makedirs(out + "/stats", exist_ok=True)
    if kept_rows:
        for i in range(0, len(kept_rows), 500_000):
            ch = kept_rows[i:i + 500_000]
            t = {c: [r.get(c) for r in ch] for c in SCHEMA_COLS}
            pq.write_table(pa.table({**t, "key": pa.array(t["key"], pa.uint64()), "aux": pa.array(t["aux"], pa.float32())}), f"{out}/pairs-{i // 500_000:04d}.parquet", compression="zstd")
    with open(out + "/stats/per_language_source.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "source", "raw_rows", "dropped_by_clean_rules", "dropped_duplicate", "after_dedupe", "removed_by_protected_filter" + ("_R1_ONLY_ESTIMATE" if estimate else ""), "kept"])
        for (iso, src), c in sorted(stats.items()):
            w.writerow([iso, src, c["raw_rows"], c["dropped_clean"], c["dropped_duplicate"], c["after_dedupe"], c["removed_protected"], c["kept"]])
    with open(out + "/stats/removed_by_protected_set.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "source", "first_rule_and_side", "rows"])
        for (iso, src, h), n in sorted(removed.items(), key=lambda kv: -kv[1]):
            w.writerow([iso, src, h, n])
    per = collections.Counter()
    for r in kept_rows:
        per[r["iso"]] += 1
    json.dump({"kept_by_language": dict(per), "kept_total": len(kept_rows), "protected_index_version": version, "estimate_only_R1": estimate, "errors": errors, "ful_min_sim": FUL_MIN_SIM}, open(out + "/stats/summary.json", "w"), indent=1)


def nde_crosscheck(a, kept_rows):
    """cross-check of the nde Bible alignment against SalomonMetre13/nde_en (30,839 verses, KJV, no licence): NOT used for training; reports how many of our nde verses occur there"""
    try:
        p = hf_get("SalomonMetre13/nde_en", "nde_en.json", a.tmp)
        d = json.load(open(p))
        theirs = {re.sub(r"\W+", " ", x["translation"]["nde"].casefold()).strip() for x in d}
        ours = [re.sub(r"\W+", " ", r["target"].casefold()).strip() for r in kept_rows if r["iso"] == "nde"]
        hit = sum(1 for o in ours if o in theirs)
        res = {"ours_nde_verses_kept": len(ours), "theirs_verses": len(theirs), "ours_found_in_theirs": hit, "share": round(hit / max(1, len(ours)), 4), "note": "same Biblica text expected; differences = revisions / punctuation"}
    except Exception as e:
        res = {"error": repr(e)[:200]}
    shutil.rmtree(os.path.join(a.tmp, "SalomonMetre13__nde_en"), ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tmp", default="work/_extra_dl"); ap.add_argument("--index"); ap.add_argument("--manifest")
    ap.add_argument("--languages", nargs="*", default=list(LANGS)); ap.add_argument("--ful-min-sim", type=float, default=FUL_MIN_SIM, help="NLLB-mined Fulah: drop rows with a LASER margin below this (the set runs from 1.05 up; 1.06 keeps about 41%%)")
    ap.add_argument("--estimate", action="store_true", help="laptop mode: no protected index; counts before the strict filter plus an R1-only estimate from --sent-hashes; the kept pairs are written too (R1-filtered only), so that windows.py --stats-only can project them")
    ap.add_argument("--efi-nt-tsv", default=None, help="owner-provided Efik NT sheet as TSV with columns book, efik, english"); ap.add_argument("--sent-hashes", default=None); ap.add_argument("--allow-missing", nargs="*", default=[], help="source names whose download failure is not fatal")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.tmp, exist_ok=True)
    filt, version = None, None
    if a.estimate:
        if a.sent_hashes:
            filt = R1Estimate(a.sent_hashes)
    else:
        if not (a.index and a.manifest):
            raise SystemExit("--index and --manifest (protected index, sealed) are required: the strict filter is not optional")
        filt = Filter(a.index, a.manifest); version = filt.version
    stats, removed, kept, errors = process(SOURCES, a, filt, only_iso=set(a.languages))
    if "nde" in a.languages and not errors.get("nde"):
        json.dump(nde_crosscheck(a, kept), open(a.out + "/nde_crosscheck.json", "w"), indent=1)
    write_outputs(a.out, stats, removed, kept, errors, version, a.estimate)
    fatal = {k: v for k, v in errors.items() if k not in a.allow_missing}
    per = collections.Counter(r["iso"] for r in kept)
    print("kept by language:", dict(per), "| errors:", errors or "none")
    shutil.rmtree(a.tmp, ignore_errors=True)
    if fatal:
        raise SystemExit(f"source download/parse failed: {fatal} (re-run; or --allow-missing NAME to continue without it)")


if __name__ == "__main__":
    main()
