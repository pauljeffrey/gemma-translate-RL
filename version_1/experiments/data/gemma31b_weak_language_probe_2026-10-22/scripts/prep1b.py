import json, os, re, numpy as np, collections
from transformers import AutoTokenizer
from gemma_mt.languages import ALL_ISOS
OUT="/workspace/jobs/r15/probe/raw"; os.makedirs(OUT,exist_ok=True)
import sys; L=sys.argv[1:]
tok=AutoTokenizer.from_pretrained("/workspace/sft_build/runs/final_1433")
D="/workspace/jobs/r4/raw/meta/gemma/packed_fwd"
PR=re.compile(r"^<bos><start_of_turn>user\nTranslate the text to [^:\n]*:  (.*)<end_of_turn>\n<start_of_turn>model\n$",re.S)
for sp in ("val",):
    ld=lambda n: np.load(f"{D}/{sp}/{n}.npy",mmap_mode="r")
    iso=np.asarray(ld("iso_id")); off=np.asarray(ld("sample_offset")).astype(np.int64); sl=np.asarray(ld("sample_len")).astype(np.int64)
    pl=np.asarray(ld("prompt_len")).astype(np.int64); tl=np.asarray(ld("target_len")).astype(np.int64)
    tokens=np.memmap(f"{D}/{sp}/tokens.u32.bin",dtype=np.uint32,mode="r")
    cnt=collections.Counter()
    for l in L:
        idx=np.where(iso==ALL_ISOS.index(l))[0]
        with open(f"{OUT}/{sp}_{l}.jsonl","w") as f:
            for i in idx:
                ids=tokens[off[i]:off[i]+sl[i]].tolist()
                m=PR.match(tok.decode(ids[:pl[i]],skip_special_tokens=False))
                if not m: cnt[l+"_nomatch"]+=1; continue
                t=tok.decode(ids[pl[i]:pl[i]+tl[i]],skip_special_tokens=False).replace("<end_of_turn>","")
                t=t[len(l):].strip() if t.startswith(l) else t.strip()
                f.write(json.dumps({"english":m.group(1),"target":t,"split":sp,"idx":int(i)},ensure_ascii=False)+"\n"); cnt[l]+=1
    print(sp,dict(cnt),flush=True)
