import json, os, sys, time, glob
sys.path.insert(0,"/workspace/jobs/r15/probe")
from probe_common import *
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
S="/workspace/jobs/r15/synth"; M="/workspace/datagen/models/gemma-4-31B-it-qat-w4a16-ct"
ISOS=os.environ["ISOS"].split(); K=int(os.environ.get("K","8")); BATCH=int(os.environ.get("BATCH","240")); MINUTES=float(os.environ.get("MINUTES","90"))
TEMP=float(os.environ.get("TEMP","0.7")); MAXP=2600
tok=AutoTokenizer.from_pretrained(M)
def render(user):
    text=tok.apply_chat_template([{"role":"user","content":user}],tokenize=False,add_generation_prompt=True)
    assert text.count("<bos>")==1 and "<|think|>" not in text and text.endswith("<|turn>model\n<|channel>thought\n<channel|>"), text[-80:]
    return tok(text,add_special_tokens=False)["input_ids"]
llm=LLM(model=M,dtype="bfloat16",max_model_len=3328,max_num_seqs=int(os.environ.get("SEQS","8")),kv_cache_dtype="auto",enable_prefix_caching=True,max_num_batched_tokens=8192,seed=1234,
        language_model_only=True,kv_cache_memory_bytes=int(float(os.environ.get("KV_GIB","6.5"))*2**30),gpu_memory_utilization=float(os.environ.get("UTIL","0.27")))
spg=SamplingParams(temperature=0.0,max_tokens=500,stop_token_ids=[1,106,50],skip_special_tokens=True)
sps=SamplingParams(temperature=TEMP,top_p=0.95,top_k=64,seed=11,max_tokens=500,stop_token_ids=[1,106,50],skip_special_tokens=True)
src={i:[json.loads(x) for x in open(f"{S}/src/{i}.jsonl")] for i in ISOS}
nxt={}
for i in ISOS:
    os.makedirs(f"{S}/out/{i}",exist_ok=True)
    done=sorted(glob.glob(f"{S}/out/{i}/shard_*.jsonl")); nxt[i]=len(done)
t_end=time.time()+MINUTES*60; tot=0
while time.time()<t_end and not os.path.exists("/workspace/jobs/r15/STOP_SYNTH"):
    progressed=False
    for l in ISOS:
        b=nxt[l]; items=src[l][b*BATCH:(b+1)*BATCH]
        if not items: continue
        progressed=True
        ids=[];used=[]
        for t in items:
            kk=K
            while True:
                p=render(build_prompt(l,t["english"],t["shots"],kk,False))
                if len(p)<=MAXP or kk<=1: break
                kk-=1
            ids.append(p); used.append(kk)
        t0=time.time()
        res=llm.generate([{"prompt_token_ids":p} for p in ids]+[{"prompt_token_ids":p} for p in ids],[spg]*len(ids)+[sps]*len(ids),use_tqdm=False)
        n=len(ids); out=f"{S}/out/{l}/shard_{b:05d}.jsonl"
        with open(out+".tmp","w") as f:
            for j,t in enumerate(items):
                g=res[j].outputs[0]; s=res[n+j].outputs[0]
                f.write(json.dumps(dict(sid=t["sid"],english=t["english"],source_dataset=t["source_dataset"],licence=t["licence"],domain=t["domain"],field=t["field"],pool_tag=t["pool_tag"],
                    hyp_g=clean_out(g.text,l),hyp_s=clean_out(s.text,l),fin_g=str(g.finish_reason),fin_s=str(s.finish_reason),n_gen_g=len(g.token_ids),n_gen_s=len(s.token_ids),k_used=used[j],shot_targets=[x["target"] for x in t["shots"][:used[j]]]),ensure_ascii=False)+"\n")
        os.rename(out+".tmp",out); nxt[l]=b+1
        ng=sum(len(r.outputs[0].token_ids) for r in res); tot+=n
        print(f"{l} shard {b} n={n} gen_tok={ng} {time.time()-t0:.0f}s {ng/(time.time()-t0):.0f} tok/s total_sources={tot}",flush=True)
        if time.time()>=t_end or os.path.exists("/workspace/jobs/r15/STOP_SYNTH"): break
    if not progressed: break
print("ALL DONE",flush=True)
