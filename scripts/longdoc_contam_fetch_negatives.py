# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetches the NEGATIVE control texts of the contamination validation: 90 Wikinews articles of 2025-08..2026-05 through the MediaWiki API (UA with contact, 1 request/s).
Project Gutenberg novels are fetched by hand with curl from gutenberg.org/cache/epub/<id>/pg<id>.txt (ids in reports/longdoc_benchmark_plan_2026-10-20_parts/contamination.md)."""
import json, time, urllib.request, urllib.parse, os, sys
UA = "SabiYarn-longdoc-benchmark/0.1 (research; contact <email>)"
API = "https://en.wikinews.org/w/api.php"
def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))
titles = []; cont = None
while len(titles) < 90:
    p = dict(action="query", list="categorymembers", cmtitle="Category:Published", cmsort="timestamp", cmdir="desc", cmlimit=50, cmprop="title|timestamp", format="json")
    if cont: p["cmcontinue"] = cont
    d = get(p); time.sleep(1.1)
    titles += [(m["title"], m["timestamp"]) for m in d["query"]["categorymembers"]]
    cont = d.get("continue", {}).get("cmcontinue")
    if not cont: break
out = []
for t, ts in titles[:90]:
    d = get(dict(action="query", prop="extracts", explaintext=1, titles=t, format="json", redirects=1)); time.sleep(1.1)
    pg = next(iter(d["query"]["pages"].values()))
    txt = pg.get("extract", "")
    out.append({"title": t, "timestamp": ts, "text": txt})
os.makedirs("/workspace/sft_build/longdoc_bench/raw/contam_neg", exist_ok=True)
with open("/workspace/sft_build/longdoc_bench/raw/contam_neg/wikinews_2026.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(len(out), sum(len(o["text"].split()) for o in out), out[0]["timestamp"], out[-1]["timestamp"])
print(out[5]["text"][:600])
