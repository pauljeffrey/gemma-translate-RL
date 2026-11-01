import json, random, glob, os
R="/workspace/jobs/r15/probe"
for f in sorted(glob.glob(f"{R}/tests/*.jsonl")):
    T=[json.loads(x) for x in open(f)]
    ns=[i for i,t in enumerate(T) if t["set"]!="seen"]; se=[i for i,t in enumerate(T) if t["set"]=="seen"]
    rs=random.Random(11); rs.shuffle(ns); main=set(ns[:150]); sub=set(ns[:80]); seen40=set(se[:40])
    for i,t in enumerate(T): t["main"]=int(i in main); t["sub"]=int(i in sub); t["seen40"]=int(i in seen40)
    open(f+".new","w").write("".join(json.dumps(t,ensure_ascii=False)+"\n" for t in T)); os.rename(f+".new",f)
    print(os.path.basename(f),len(T),sum(t["main"] for t in T),sum(t["sub"] for t in T),sum(t["seen40"] for t in T))
