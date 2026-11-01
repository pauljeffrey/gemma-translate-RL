#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check of the S2 start checkpoint (google/gemma-3-270m-it + 3,000 Ge'ez tokens, scripts/upcycle_s2_start.sh). CPU only; exits non-zero on the first failure.

    python scripts/upcycle_s2_start_check.py --base <gemma-3-270m-it dir> --start <ckpt dir> --swap-map <swap_map.json> [--report out.json] [--latin-text file.txt]

  1. the swap map was built from THIS base tokenizer (sha256 of the original tokenizer.json) and lists n new tokens (default 3,000: 61 letters + 2,939 merges), all
     distinct, all with ids in [256001, 262143] (the unused tail of Gemma-3's 262,144 ids), none of them an id < 256001, none a special token.
  2. tokenizer: vocabulary size and id range unchanged (262,145 entries with the one added token; 262,144 embedding rows); every new id carries its new token; every other id keeps its token; the original merge list is an untouched
     prefix of the new one; the original `<unusedN>` strings of the replaced ids are gone and nothing else is.
  3. weights: embedding matrix [262144, 640] (tied: no lm_head tensor); every row outside the new ids is identical to the base; every new row equals the mean of the base rows of the
     sub-tokens the ORIGINAL tokenizer gives for the token's string (recomputed here independently of the surgery code); every other tensor is identical to the base.
  4. behaviour of the tokenizer: identical ids on Latin-script text (built-in sentences in English, French, Hausa, Yoruba, Igbo, Swahili, Kinyarwanda, Ewe, Fon, Bambara + optional
     --latin-text file) and shorter sequences with new ids (>= 256001) on Amharic / Tigrinya text; decode(encode(x)) == x on those texts.
  5. (if transformers is importable) the chat template of the extended tokenizer is the reference Gemma-3 template (gemma_mt.format.assert_template_matches_reference).
"""
import argparse, glob, hashlib, json, os, sys

LO, HI = 256001, 262143
LATIN = ["The market opens at eight o'clock and the children are playing outside.", "Où est l'hôpital le plus proche ? Merci beaucoup.",
         "Ina son in karanta littafi kowace dare.", "Ẹ kú àárọ̀, báwo ni ẹ ṣe wà lónìí?", "Anyị ga-eje ahịa echi n'ụtụtụ.", "Habari za asubuhi, tunakwenda sokoni leo.",
         "Muraho, murakoze cyane ku bufasha bwanyu.", "Ŋdi nyuie, akpe ɖe ŋutifafa ŋu.", "Mi na yì azɔ̌ ɖò xwé ɔ tɔn mɛ.", "I ni sogoma, i ka kɛnɛ wa?", "Translate the text to Hausa:  Good morning."]
GEEZ = ["ሰላም ለዓለም፤ እንዴት ነህ? ዛሬ ጥሩ ቀን ነው።", "የኢትዮጵያ መንግሥት አዲስ ትምህርት ቤት ከፍቷል።", "ከመይ ኣለኻ? ሎሚ ጽቡቕ መዓልቲ እዩ።", "ትግራይ ኣብ ሰሜን ኢትዮጵያ ትርከብ።"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fail(msg):
    print("FAIL:", msg, flush=True)
    sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--start", required=True); ap.add_argument("--swap-map", required=True)
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--report"); ap.add_argument("--latin-text")
    a = ap.parse_args(argv)
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    smap = json.load(open(a.swap_map, encoding="utf-8"))
    rep = {}
    # 1. swap map
    if smap["orig_tokenizer_sha256"] != sha256_file(os.path.join(a.base, "tokenizer.json")):
        fail("the swap map was built from another original tokenizer.json than --base/tokenizer.json")
    nt = smap["new_tokens"]
    ids = [t["id"] for t in nt]
    if len(nt) != a.n or len(set(ids)) != a.n or len({t["token"] for t in nt}) != a.n:
        fail(f"expected {a.n} distinct new tokens, found {len(nt)} ({len(set(ids))} distinct ids)")
    if min(ids) < LO or max(ids) > HI:
        fail(f"new ids outside [{LO}, {HI}]: min {min(ids)}, max {max(ids)}")
    rep["new_tokens"] = len(nt); rep["id_min"] = min(ids); rep["id_max"] = max(ids)
    print(f"[1] swap map: {len(nt)} new tokens ({sum(1 for t in nt if t['a'] is None)} letters + {sum(1 for t in nt if t['a'] is not None)} merges), ids {min(ids)}..{max(ids)} all inside [{LO}, {HI}]", flush=True)
    # 2. tokenizer
    orig = Tokenizer.from_file(os.path.join(a.base, "tokenizer.json")); new = Tokenizer.from_file(os.path.join(a.start, "tokenizer.json"))
    ov, nv = orig.get_vocab(), new.get_vocab()
    id2o = {i: s for s, i in ov.items()}; id2n = {i: s for s, i in nv.items()}
    if len(ov) != len(nv) or len(id2n) != len(nv) or max(id2o) != max(id2n):
        fail(f"vocabulary sizes {len(ov)} / {len(nv)} (ids up to {max(id2o)} / {max(id2n)})")
    idset = set(ids)
    for t in nt:
        if id2n.get(t["id"]) != t["token"]:
            fail(f"id {t['id']} should hold {t['token']!r}, holds {id2n.get(t['id'])!r}")
        if not id2o[t["id"]].startswith("<unused"):
            fail(f"id {t['id']} was {id2o[t['id']]!r} in the base, not an unused slot")
    changed = [i for i in id2o if i not in idset and id2o[i] != id2n.get(i)]
    if changed:
        fail(f"{len(changed)} ids outside the new set changed their token (first: {changed[:5]})")
    mo = json.loads(orig.to_str())["model"]["merges"]; mn = json.loads(new.to_str())["model"]["merges"]
    if mn[:len(mo)] != mo:
        fail("the original merge list is not an untouched prefix of the new one")
    rep["merges_appended"] = len(mn) - len(mo)
    print(f"[2] tokenizer: vocabulary unchanged in size ({len(nv)} entries); only the {len(nt)} unused slots changed; {len(mn) - len(mo)} merges appended after the {len(mo)} original ones", flush=True)
    # 3. weights
    def tensors(d):
        sd = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            sd.update(load_file(f))
        return sd
    b, s = tensors(a.base), tensors(a.start)
    if set(b) != set(s):
        fail(f"tensor names differ: {sorted(set(b) ^ set(s))[:5]}")
    key = [k for k in b if k.endswith("embed_tokens.weight")][0]
    if any(k.endswith("lm_head.weight") for k in b):
        fail("untied checkpoint: the surgery assumes tied input/output embeddings")
    Eb, Es = b[key].float(), s[key].float()
    if Eb.shape != (262144, 640) or Es.shape != Eb.shape:
        fail(f"embedding shape {tuple(Eb.shape)} / {tuple(Es.shape)}")
    tol = 0.0 if s[key].dtype == torch.float32 else 1e-2
    mask = torch.ones(262144, dtype=torch.bool); mask[list(idset)] = False
    if float((Eb[mask] - Es[mask]).abs().max()) > (0.0 if s[key].dtype == b[key].dtype or tol == 0 else tol):
        fail("embedding rows outside the new ids differ from the base")
    worst = 0.0
    for t in nt:
        sub = orig.encode(t["token"].replace("▁", " "), add_special_tokens=False).ids
        if not sub:
            fail(f"{t['token']!r} has no sub-tokens under the original tokenizer")
        expect = Eb[sub].mean(0)
        worst = max(worst, float((Es[t["id"]] - expect).abs().max()))
    if worst > max(tol, 1e-7):
        fail(f"new rows differ from the mean of their sub-token rows by {worst}")
    for k in b:
        if k != key and not torch.equal(b[k].float(), s[k].float()):
            fail(f"tensor {k} differs from the base")
    rep["max_row_error_vs_mean_of_subtokens"] = worst; rep["start_dtype"] = str(s[key].dtype)
    print(f"[3] weights: {len(b)} tensors; rows outside the new ids and every other tensor identical to the base; the {len(nt)} new rows = mean of their sub-token rows (max abs error {worst:.2e}, dtype {s[key].dtype})", flush=True)
    # 4. behaviour
    latin = list(LATIN)
    if a.latin_text:
        latin += [l.strip() for l in open(a.latin_text, encoding="utf-8") if l.strip()][:20000]
    diff = [x for x in latin if orig.encode(x).ids != new.encode(x).ids]
    if diff:
        fail(f"{len(diff)} Latin-script texts are tokenised differently (first: {diff[0]!r})")
    used = 0; n_o = n_n = 0
    for x in GEEZ:
        eo, en = orig.encode(x).ids, new.encode(x).ids
        used += sum(1 for i in en if i >= LO); n_o += len(eo); n_n += len(en)
        if new.decode(en, skip_special_tokens=False).replace("<bos>", "") != orig.decode(eo, skip_special_tokens=False).replace("<bos>", ""):
            fail(f"decode(encode(x)) differs between the tokenizers for {x!r}")
    if used == 0 or n_n >= n_o:
        fail(f"Ge'ez text does not use the new tokens or is not shorter ({n_o} -> {n_n} tokens, {used} new ids)")
    rep["latin_texts_identical"] = len(latin); rep["geez_tokens_orig"] = n_o; rep["geez_tokens_new"] = n_n
    print(f"[4] {len(latin)} Latin-script texts tokenise identically; {len(GEEZ)} Ge'ez sentences {n_o} -> {n_n} tokens using {used} new ids; round-trip decode equal", flush=True)
    # 5. template
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from transformers import AutoTokenizer
        from gemma_mt import format as fmt
        fmt.assert_template_matches_reference(AutoTokenizer.from_pretrained(a.start))
        print("[5] chat template of the extended tokenizer matches the reference Gemma-3 template", flush=True)
        rep["template_ok"] = True
    except ImportError as e:
        print("[5] skipped (transformers / gemma_mt not importable):", e, flush=True)
    rep["base_model_sha256"] = sha256_file(glob.glob(os.path.join(a.base, "*.safetensors"))[0]); rep["start_model_sha256"] = sha256_file(glob.glob(os.path.join(a.start, "*.safetensors"))[0])
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)
    print("S2 START CHECKPOINT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
