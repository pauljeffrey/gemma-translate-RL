"""Shared helpers for the Ge'ez Step 0 scripts (2026-10-25). Read-only Hub access; the token is read inside this module and never printed."""
import os, json, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reports", "data", "ge_ez_step0_2026-10-25")
CACHE = os.environ.get("GE_EZ_CACHE", os.path.join(OUT, "_cache"))

def load_token():
    for k in ("HF_READ_TOKEN", "HF_API_KEY", "HF_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    p = os.path.join(ROOT, ".env")
    vals = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*$", l)
            if m:
                vals[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    for k in ("HF_READ_TOKEN", "HF_API_KEY", "HF_TOKEN"):
        if vals.get(k):
            return vals[k]
    return None

# FLORES+ codes (script-qualified) for the languages of this study
FLORES = {"eng": "eng_Latn", "amh": "amh_Ethi", "tir": "tir_Ethi", "fon": "fon_Latn", "hau": "hau_Latn",
          "yor": "yor_Latn", "swh": "swh_Latn", "kin": "kin_Latn", "ibo": "ibo_Latn"}
