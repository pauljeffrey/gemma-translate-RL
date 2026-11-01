"""T1: build evaluation units (chunks of k consecutive sentences) for every benchmark/language and the table of unique English chunk texts.
Output: /workspace/jobs/t1/units.jsonl.gz  {b,iso,k,uid,ids,ekey}   and   /workspace/jobs/t1/en_texts.jsonl {ekey,text}.
Chunks: non-overlapping windows of k consecutive sentences (stride k) inside one document; tail shorter than k is dropped (k=1: every sentence).
FLORES+: document = url (281 excerpts of 2-6 sentences). GlotStory: document = id prefix before '#'. NTREX / MAFAND: no document ids available, windows over consecutive ids."""
import gzip, json, collections, hashlib, os, re
os.makedirs("/workspace/jobs/t1", exist_ok=True)
rows = collections.defaultdict(dict)       # (b,iso) -> {id: en}
for l in gzip.open("/workspace/jobs/e1/sys_rl.jsonl.gz", "rt"):
    d = json.loads(l)
    rows[(d["b"], d["iso"])][d["id"]] = d["en"]
fl = [json.loads(l) for l in open("/workspace/sft_build/bench/test_sets/flores_plus/devtest/eng_Latn.jsonl")]
doc_of = {str(x["id"]): x["url"] for x in fl}
SEP = " "
def key(t): return hashlib.md5(t.encode()).hexdigest()[:16]
texts = {}
units = []
def add(b, iso, k, uid, ids, en):
    t = SEP.join(en); ek = key(t); texts[ek] = t
    units.append({"b": b, "iso": iso, "k": k, "uid": uid, "ids": ids, "ekey": ek})
def windows(seq, k):
    for s in range(0, len(seq) - k + 1, k): yield seq[s:s + k]
for (b, iso), m in sorted(rows.items()):
    if b not in ("flores_plus", "ntrex", "mafand", "smol_sent", "glot_story", "salt"): continue
    ks = (1, 3, 5) if b in ("flores_plus", "ntrex", "mafand", "glot_story") else (1,)
    if b == "flores_plus":
        docs = collections.defaultdict(list)
        for i in sorted(m, key=int): docs[doc_of[i]].append(i)
        groups = list(docs.values())
    elif b == "glot_story":
        docs = collections.defaultdict(list)
        for i in m:
            mm = re.match(r"(asp_\d+)#(\d+)(@.*)?$", i); d = mm.group(1) + (mm.group(3) or ""); docs[d].append((int(mm.group(2)), i))
        groups = []
        for d, v in sorted(docs.items()):
            v.sort(); run = [v[0]]
            for a in v[1:]:
                if a[0] == run[-1][0] + 1: run.append(a)
                else: groups.append([x[1] for x in run]); run = [a]
            groups.append([x[1] for x in run])
    elif b in ("ntrex", "mafand"):
        ids = sorted(m, key=int); groups = []; run = [ids[0]]
        for i in ids[1:]:
            if int(i) == int(run[-1]) + 1: run.append(i)
            else: groups.append(run); run = [i]
        groups.append(run)
    else:
        groups = [[i] for i in sorted(m, key=lambda x: int(x))]
    for k in ks:
        if b in ("smol_sent", "salt"): groups_k = [[i] for i in sorted(m, key=int)]
        else: groups_k = groups
        for g in groups_k:
            for w in windows(g, k):
                add(b, iso, k, f"{b}|{iso}|{k}|{w[0]}", w, [m[i] for i in w])
with gzip.open("/workspace/jobs/t1/units.jsonl.gz", "wt") as f:
    for u in units: f.write(json.dumps(u) + "\n")
with open("/workspace/jobs/t1/en_texts.jsonl", "w") as f:
    for ek, t in texts.items(): f.write(json.dumps({"ekey": ek, "text": t}, ensure_ascii=False) + "\n")
c = collections.Counter((u["b"], u["k"]) for u in units)
print(len(units), len(texts)); print(sorted(c.items()))
cu = collections.Counter()
for u in units: cu[(u["b"], u["k"])] += 0
nt = collections.Counter(); seen = collections.defaultdict(set)
for u in units: seen[(u["b"], u["k"])].add(u["ekey"])
print({k: len(v) for k, v in sorted(seen.items())})
