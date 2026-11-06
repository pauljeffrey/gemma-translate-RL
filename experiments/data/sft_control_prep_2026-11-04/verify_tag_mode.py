"""Check that building TranslatePsy samples directly in tag mode `single_token` gives the SAME tokens as the production route (text-tag pack -> retag_packs.py): for N pairs,
tokenise in text mode, replace the leading tag pieces by the tag token, compare with the single_token tokenisation. Also writes tag_mode / tag_map_sha256 into a pack's meta.json
(prepare_sft does not write them; retag_packs.py does for the old packs), so that the combined dataset's meta is consistent with the s6 shard.
    python verify_tag_mode.py TOKENIZER_DIR tp_clean_dir [packed_dir_to_patch]"""
import json, os, sys, glob
sys.path.insert(0, "gemma_mt"); sys.path.insert(0, "data_pipeline/sft_clean/chain")
from transformers import AutoTokenizer
import gemma_mt.format as fmt
import retag_packs as RP
tok = AutoTokenizer.from_pretrained(sys.argv[1]); clean = sys.argv[2]
n_bad = n = 0
for f in sorted(glob.glob(clean + "/en2x_*.jsonl")):
    iso = os.path.basename(f)[5:-6]
    rows = [json.loads(l) for _, l in zip(range(12), open(f, encoding="utf-8"))]
    pieces = [int(i) for i in tok(iso, add_special_tokens=False)["input_ids"]]
    tag_id = int(tok.convert_tokens_to_ids(__import__("gemma_mt.tags", fromlist=["x"]).tag_string(iso, tok)))
    for r in rows:
        with fmt.tag_mode("text"): a = fmt.build_example(r["english"], iso, r["target"], tok)
        with fmt.tag_mode("single_token"): b = fmt.build_example(r["english"], iso, r["target"], tok)
        ia, ib = list(a.input_ids), list(b.input_ids)
        p = a.prompt_len
        assert ia[p:p + len(pieces)] == pieces, (iso, ia[p:p + 4], pieces)
        conv = ia[:p] + [tag_id] + ia[p + len(pieces):]
        n += 1
        if conv != ib or b.prompt_len != p or b.target_len != a.target_len - len(pieces) + 1: n_bad += 1
print(f"{n} samples checked in 19 languages: {n_bad} differ between (text-mode tokens with the tag pieces replaced by the tag token) and single_token mode")
assert n_bad == 0
if len(sys.argv) > 3:
    d = sys.argv[3]; m = json.load(open(d + "/meta.json"))
    pcs, ids = RP.tag_maps(tok, m["iso_table"])
    m["tag_mode"] = "single_token"; m["tag_map_sha256"] = RP.tag_map_sha256(m["iso_table"], pcs, ids)
    json.dump(m, open(d + "/meta.json", "w"), indent=1); print("patched meta:", d, m["tag_map_sha256"][:16])
