"""Nairaland raw HTML (nairaland_cc.py output) -> cleaned, deduplicated text shards -> private HF dataset.

  python nairaland_clean.py build  --src nairaland_cc_data/pages.jsonl.gz --out nairaland_clean
  python nairaland_clean.py push   --out nairaland_clean --repo <HF_ORG_LEGACY>/nairaland-forum-text

build: one row per page = {url, timestamp, text, crawl}; text = thread title + post bodies only
       (no nav/ads/usernames; quotes removed; e-mails and phone numbers masked).
       Dedup: a post id is kept once globally; long posts (>=80 chars) are also kept once by content hash;
       pages left with no new text are dropped.
push:  uploads each parquet shard as its own commit (resumable: shards already on the Hub are skipped).
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from selectolax.lexbor import LexborHTMLParser

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<!\d)(?:\+?234|0)[\s-]?[789][01]\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
JUNK_LINE = re.compile(r"^\s*(?:\d+\s+(?:Likes?|Shares?)|Share|Quote|Re:.*)\s*$", re.I)
MIN_POST_DEDUP = 80
MIN_PAGE_CHARS = 100
SHARD_ROWS = 40_000


def node_text(n):
    for br in n.css("br"):
        br.replace_with("\n")
    return n.text(strip=False)


def drop(tree, sel):
    for n in tree.css(sel):
        n.decompose()


def extract(html):
    """-> (title, [(post_id, text)]) for either site layout, or (title, [])."""
    t = LexborHTMLParser(html)
    drop(t, "script,style,noscript,blockquote,.quote,.quoteheader,.quotefooter,.attachment,.attachments,.signature")
    title = ""
    if t.css_first("title"):
        title = re.sub(r"\s+-\s+Nairaland.*$|\s+-\s+[^-]+-\s+Nigeria$", "", t.css_first("title").text().strip())
    posts = []
    modern = t.css("td[id^=pb] > div.narrow")
    if modern:
        for d in modern:
            posts.append(("m" + d.parent.attributes["id"][2:], node_text(d)))
    else:  # 2005-2009 layout: <a name="msgN"></a> ... <div style="overflow: auto; width: 100%;">body</div>
        for a in t.css("a[name^=msg]"):
            pid = a.attributes["name"][3:]
            body, n = None, a.next
            while n is not None and body is None:
                body = n.css_first("div[style*='overflow: auto']") if n.tag != "-text" else None
                n = n.next
            if body is not None:
                posts.append(("o" + pid, node_text(body)))
    return title, posts


def clean(s):
    s = EMAIL.sub("[EMAIL]", s)
    s = PHONE.sub("[PHONE]", s)
    lines = [re.sub(r"[ \t ]+", " ", l).strip() for l in s.replace("\r", "").split("\n")]
    lines = [l for l in lines if not JUNK_LINE.match(l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def build(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seen_ids, seen_hash = set(), set()
    rows, shard, stats = [], 0, dict(pages=0, no_posts=0, dup_only=0, short=0, kept=0, chars=0)

    def flush():
        nonlocal rows, shard
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), out / f"train-{shard:05d}.parquet", compression="zstd")
            shard += 1
            rows = []

    with gzip.open(src, "rt") as f:
        for line in f:
            r = json.loads(line)
            stats["pages"] += 1
            title, posts = extract(r["html"])
            if not posts:
                stats["no_posts"] += 1
                continue
            texts = []
            for pid, raw in posts:
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                tx = clean(raw)
                if not tx:
                    continue
                if len(tx) >= MIN_POST_DEDUP:
                    h = hashlib.blake2b(re.sub(r"\W+", " ", tx.lower()).encode(), digest_size=12).digest()
                    if h in seen_hash:
                        continue
                    seen_hash.add(h)
                texts.append(tx)
            if not texts:
                stats["dup_only"] += 1
                continue
            text = (clean(title) + "\n\n" if title else "") + "\n\n".join(texts)
            if len(text) < MIN_PAGE_CHARS:
                stats["short"] += 1
                continue
            rows.append({"url": r["url"], "timestamp": r["timestamp"], "text": text, "crawl": r["crawl"]})
            stats["kept"] += 1
            stats["chars"] += len(text)
            if len(rows) >= SHARD_ROWS:
                flush()
            if stats["pages"] % 20000 == 0:
                print(stats, flush=True)
    flush()
    print("DONE", stats, f"shards={shard}")


def push(out, repo):
    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    tok = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    have = set(api.list_repo_files(repo, repo_type="dataset"))
    for p in sorted(Path(out).glob("train-*.parquet")):
        dest = f"data/{p.name}"
        if dest in have:
            continue
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dest, repo_id=repo, repo_type="dataset",
                        commit_message=f"add {p.name}")
        print("pushed", dest, f"{p.stat().st_size / 1e6:.0f} MB", flush=True)
    card = Path(out) / "README.md"
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, repo_type="dataset")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "push"])
    ap.add_argument("--src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo")
    a = ap.parse_args()
    build(a.src, a.out) if a.cmd == "build" else push(a.out, a.repo)
