"""A2 long-document evaluation: shared helpers (run on the box with venv_gemma; PYTHONPATH=/workspace/sft_build/gemma_mt_2c:/workspace/sft_build/pylibs)."""
import json, os, random, gzip, collections
A2 = "/workspace/jobs/a2"
ROOT = "/workspace/jobs/r12/doc"
DATA = A2 + "/data/hf"
CORE = DATA + "/core"
WORK = A2 + "/work"   # read-only (R6 items, refs, requests)
MYWORK = ROOT + "/work"
OUT = ROOT + "/out"
MODELS = {"rl2c": ("/workspace/sft_build/hf/hub/models--" + os.environ['HF_ORG_LEGACY'] + "--gemma-3-270m-translate-rl2c/snapshots/80baefff3b16b120b3e37e70afeb51f1fc248d96"), "sft": A2 + "/models/sft"}
CTX = 10240
HUMAN_LANGS = "hau yor ibo swh zul xho amh kin som wol twi mlg".split()
NLLB_ONLY = "kea kam sag nus".split()
LANGS = HUMAN_LANGS + NLLB_ONLY
MADLAD_CLEAN = {"mlg", "som", "swh", "xho", "zul"}      # MADLAD flagged < 2% among our languages (summary.csv)
CORE_BINS = [64, 256, 512, 1024, 2048, 4096]
HUMAN_BINS = ["sm", "st512", "st1024", "st2048"]       # SmolDoc short docs (<=~680 tok), stitched ~512 / ~1024 / ~2048 Gemma tokens


def rd(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wr(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)


def load_core_docs():
    docs = rd(CORE + "/docs.jsonl")
    for d in docs:
        d["set"] = "core"
        d["hbin"] = d["bin"]
    return docs


def human_selection(seed=0, n_unc=None, n_con=None):
    """Deterministic per-language subset of the human-reference documents: {iso: [(doc_id, source_file, contaminated)]}.
    Caps per (language, human bin): uncontaminated up to UNC, contaminated up to CON."""
    UNC = {"sm": 60, "st512": 40, "st1024": 20, "st2048": 10}
    CON = {"sm": 20, "st512": 20, "st1024": 20, "st2048": 25}
    sm = {d["doc_id"]: d for d in rd(CORE + "/human_smoldoc/docs.jsonl")}
    st = {d["doc_id"]: d for d in rd(CORE + "/human_stitched/docs.jsonl")}
    sel = {}
    for iso in HUMAN_LANGS:
        rng = random.Random(f"{seed}-{iso}")
        cand = collections.defaultdict(list)
        for name, docs in (("human_smoldoc", sm), ("human_stitched", st)):
            for r in rd(f"{CORE}/{name}/refs/{iso}.jsonl"):
                d = docs[r["doc_id"]]
                if name == "human_smoldoc":
                    hb = "sm"
                else:
                    if d["bin"] not in (512, 1024, 2048):
                        continue
                    hb = f"st{d['bin']}"
                cand[(hb, bool(r["contaminated"]))].append((r["doc_id"], name))
        rows = []
        for (hb, con), lst in cand.items():
            lst = sorted(lst)
            rng.shuffle(lst)
            cap = (CON if con else UNC)[hb]
            rows += [(i, n, con, hb) for i, n in lst[:cap]]
        sel[iso] = sorted(rows)
    return sel, sm, st


def load_hyps_file(p):
    return rd(p)


# ------------------------------------------------------------------------------------------------ chunking
def piece_list(sent, max_chars=700):
    from gemma_mt.eval.longdoc import split_long_sentence
    return split_long_sentence(sent, max_chars)


def make_windows(sents, para_starts, count_tokens, max_tok=256):
    """Windows of whole sentences, each <= max_tok Gemma tokens (a single longer sentence is its own window). Paragraph-aware: a window is closed
    at a paragraph boundary when the next paragraph does not fit; paragraphs are never merged across a boundary that would overflow.
    Returns [(first_sent_index, last_sent_index_exclusive)]."""
    from gemma_mt.eval.longdoc import join_sentences
    n = len(sents)
    if n == 0:
        return []
    starts = sorted(set(para_starts or [0]) | {0})
    paras = [(starts[k], starts[k + 1] if k + 1 < len(starts) else n) for k in range(len(starts))]
    out = []
    cur = None
    for a, b in paras:
        # whole paragraph into the current window if it fits
        if cur is not None and count_tokens(join_sentences(sents[cur[0]:b], [s - cur[0] for s in starts if cur[0] <= s < b])) <= max_tok:
            cur = (cur[0], b)
            continue
        if cur is not None:
            out.append(cur)
            cur = None
        # new window for this paragraph; split it by sentences if it is too long
        if count_tokens(" ".join(sents[a:b])) <= max_tok:
            cur = (a, b)
        else:
            w0 = a
            for i in range(a, b):
                if i > w0 and count_tokens(" ".join(sents[w0:i + 1])) > max_tok:
                    out.append((w0, i))
                    w0 = i
            cur = (w0, b)
    if cur is not None:
        out.append(cur)
    return out


def window_text(sents, para_starts, a, b):
    from gemma_mt.eval.longdoc import join_sentences
    return join_sentences(sents[a:b], [s - a for s in sorted(set(para_starts or [0]) | {0}) if a <= s < b])
