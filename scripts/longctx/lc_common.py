"""LC1 long-context benchmark: shared helpers (box: /workspace/jobs/lc1; run with venv_gemma, PYTHONPATH from env_lc.sh)."""
import json, os, gzip, random, collections

ROOT = "/workspace/jobs/lc1"
WORK, OUT, HYP, SC, RES = ROOT + "/work", ROOT + "/out", ROOT + "/hyp", ROOT + "/sc", ROOT + "/res"
A2 = "/workspace/jobs/a2/data/hf"
CORE = A2 + "/core"
NLLB_REFS = "/workspace/jobs/a2/data/pseudo/nllb/nllb-200-3.3b"
AFS = "/workspace/jobs/r7/assets/afs_raw/corpus"
TOK_DIR = "/workspace/sft_build/runs/final_1433"          # Gemma-3 tokenizer + our chat template (all four of our models share it)
CTX = 10240

G1 = "hau yor swh amh ibo mlg".split()                       # human-reference languages with >= 240 clean SmolDoc documents
G2 = "zul xho kin som wol twi".split()                       # ... with ~115 clean documents
HUMAN_LANGS = G1 + G2
SCI_LANGS = "amh hau lug nso yor zul".split()
CORE_BINS = [256, 512, 1024, 2048]                           # b256 = control; b4096 optional (CORE_BINS_EXTRA)
CORE_BINS_EXTRA = [4096]
HUM_BINS = ["sm", "st512", "st1024", "st2048"]
NAMES = {"afr": "Afrikaans", "amh": "Amharic", "hau": "Hausa", "ibo": "Igbo", "kin": "Kinyarwanda", "lin": "Lingala", "lug": "Luganda", "mlg": "Malagasy", "nya": "Nyanja",
         "orm": "Oromo", "sna": "Shona", "som": "Somali", "sot": "Southern Sotho", "swh": "Swahili", "tsn": "Tswana", "wol": "Wolof", "xho": "Xhosa", "yor": "Yoruba",
         "zul": "Zulu", "twi": "Twi", "nso": "Northern Sotho"}
NLLB_CODE = {"hau": "hau_Latn", "yor": "yor_Latn", "ibo": "ibo_Latn", "swh": "swh_Latn", "zul": "zul_Latn", "xho": "xho_Latn", "amh": "amh_Ethi", "kin": "kin_Latn",
             "som": "som_Latn", "wol": "wol_Latn", "twi": "twi_Latn", "mlg": "plt_Latn", "lug": "lug_Latn", "nso": "nso_Latn"}


def rd(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wr(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    op = gzip.open if p.endswith(".gz") else open
    with op(p + ".tmp", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)


def load_docs():
    return {d["doc_id"]: d for d in rd(WORK + "/docs.jsonl")}


def load_items():
    return rd(WORK + "/items.jsonl")


def load_refs(kind):
    p = f"{WORK}/refs_{kind}.jsonl"
    return {r["iid"]: r for r in rd(p)} if os.path.exists(p) else {}
