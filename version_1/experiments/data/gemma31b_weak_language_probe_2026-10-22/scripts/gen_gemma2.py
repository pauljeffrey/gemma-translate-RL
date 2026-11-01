import json, os, sys, time
sys.path.insert(0,"/workspace/jobs/r15/probe")
from probe_common import *
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
R="/workspace/jobs/r15/probe"; M="/workspace/datagen/models/gemma-4-31B-it-qat-w4a16-ct"
tok=AutoTokenizer.from_pretrained(M)
def render(user):
    text=tok.apply_chat_template([{"role":"user","content":user}],tokenize=False,add_generation_prompt=True)
    assert text.count("<bos>")==1 and "<|think|>" not in text and text.endswith("<|turn>model\n<|channel>thought\n<channel|>"), text[-80:]
    return tok(text,add_special_tokens=False)["input_ids"]
SIX="ish bin bbj oke tsc dag".split(); SEV="ibb urh nbl shi kea ach vmw seh nyn her yao sid ndc kri dua bas wes".split()
PLAN=json.loads(os.environ["PLAN_JSON"])
MAXP=2600
llm=LLM(model=M,dtype="bfloat16",max_model_len=3328,max_num_seqs=8,kv_cache_dtype="auto",enable_prefix_caching=True,max_num_batched_tokens=8192,seed=1234,
        language_model_only=True,kv_cache_memory_bytes=int(float(os.environ.get("KV_GIB","5"))*2**30),gpu_memory_utilization=float(os.environ.get("UTIL","0.318")))
SPS={"greedy":SamplingParams(temperature=0.0,max_tokens=600,stop_token_ids=[1,106,50],skip_special_tokens=True),
     "t07":SamplingParams(temperature=0.7,top_p=0.95,top_k=64,seed=7,max_tokens=600,stop_token_ids=[1,106,50],skip_special_tokens=True)}
cfg={"fs8":(8,False,"greedy"),"fs3":(3,False,"greedy"),"zs":(0,False,"greedy"),"fs8note":(8,True,"greedy"),"fs8t07":(8,False,"t07")}
for LANGS,SYS in PLAN:
    LANGS=LANGS.split() if isinstance(LANGS,str) else LANGS
    for s in SYS:
        k,note,spn=cfg[s]; sp=SPS[spn]; os.makedirs(f"{R}/preds/gemma_{s}",exist_ok=True)
        for l in LANGS:
            out=f"{R}/preds/gemma_{s}/{l}.jsonl"
            if os.path.exists(out): continue
            T=[json.loads(x) for x in open(f"{R}/tests/{l}.jsonl")]
            T=[t for t in T if (t["main"] or t["seen40"])] if s=="fs8" else [t for t in T if t["sub"]]
            ids=[];used=[]
            for t in T:
                kk=k
                while True:
                    p=render(build_prompt(l,t["english"],t["shots"],kk,note))
                    if len(p)<=MAXP or kk<=1: break
                    kk-=1
                ids.append(p); used.append(kk)
            t0=time.time()
            res=llm.generate([{"prompt_token_ids":p} for p in ids],sp,use_tqdm=False)
            with open(out+".tmp","w") as f:
                for t,r,p,kk in zip(T,res,ids,used):
                    o=r.outputs[0]
                    f.write(json.dumps(dict(id=t["id"],set=t["set"],hyp=clean_out(o.text,l),raw=o.text,n_gen=len(o.token_ids),finish=str(o.finish_reason),n_prompt=len(p),k_used=kk),ensure_ascii=False)+"\n")
            os.rename(out+".tmp",out)
            ng=sum(len(r.outputs[0].token_ids) for r in res)
            print(f"{s} {l} n={len(T)} gen_tok={ng} {time.time()-t0:.0f}s {ng/(time.time()-t0):.0f} tok/s",flush=True)

print("ALL DONE",flush=True)
