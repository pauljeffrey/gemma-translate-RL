import json, os, sys, time, torch
from gemma_mt.eval.eval_translation import generate_translations
from gemma_mt.eval import runtime as RT
from gemma_mt.modeling import load_model
R="/workspace/jobs/r15/probe"; os.makedirs(R+"/preds/final_1433",exist_ok=True)
ref=RT.resolve_model("/workspace/sft_build/runs/final_1433")
tok,stop,pad=RT.load_tokenizer_and_stops(ref.path)
model=load_model(ref.path,attn_implementation="sdpa",dtype=torch.bfloat16,device="cuda")
for l in "ish bin bbj oke tsc dag".split():
    T=[json.loads(x) for x in open(f"{R}/tests/{l}.jsonl")]
    rows=[dict(english=t["english"],iso=l,reference=t["reference"]) for t in T]
    t0=time.time()
    out=generate_translations(model,tok,rows,stop_ids=stop.ids,pad_id=pad,device="cuda",batch_size=16,max_new_tokens=512)
    with open(f"{R}/preds/final_1433/{l}.jsonl","w") as f:
        for t,o in zip(T,out): f.write(json.dumps(dict(id=t["id"],set=t["set"],hyp=o["hyp"],raw=o["raw"],finished=o["finished"],prefix=o["prefix_status"]),ensure_ascii=False)+"\n")
    print(l,len(rows),f"{time.time()-t0:.0f}s",flush=True)
