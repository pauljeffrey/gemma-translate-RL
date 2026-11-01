#!/usr/bin/env python3
"""W1 (2026-10-26): greedy English -> X generation of the final-run Q8_0 GGUF on CPU for the weak-language dev sample (7,700 rows).

    python weak_select_generate.py [--threads 5] [--parallel 8] [--max-new 512]

One llama-server for all languages; resumable (rows already in gen_raw.jsonl are skipped; every finished row is appended and flushed at once).
Prompt = the training format with ONE bos (the server adds it; the literal '<bos>' of cpu_translate/translate_cpu.py gave a double bos).
Stores the raw completion, stop_type, tokens_predicted: scoring is done by weak_select_score.py.
"""
import argparse, concurrent.futures as cf, json, os, shutil, subprocess, sys, threading, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reports", "data", "weak_selection_2026-10-26")
GGUF = os.path.join(ROOT, "cpu_translate", "gemma-science-translate-final2-step401-Q8_0.gguf")
NAMES = json.load(open(os.path.join(ROOT, "cpu_translate", "language_names.json"), encoding="utf-8"))


def post(port, path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=5); ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512); ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N todo rows")
    ap.add_argument("--out", default=os.path.join(OUT, "gen_raw.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(OUT, "dev_sample.jsonl"), encoding="utf-8")]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l); done.add((r["iso"], r["id"]))
            except Exception:
                pass
    todo = [r for r in rows if (r["iso"], r["id"]) not in done]
    if a.limit: todo = todo[:a.limit]
    print(f"{len(rows)} rows, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo: return
    exe = shutil.which("llama-server") or sys.exit("llama-server not found")
    srv = subprocess.Popen([exe, "-m", GGUF, "-t", str(a.threads), "-np", str(a.parallel), "-c", str(a.parallel * (a.max_new + 700)), "--port", str(a.port), "--log-disable"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lock = threading.Lock(); fout = open(a.out, "a", encoding="utf-8"); t0 = time.time(); n = [0]
    try:
        for _ in range(180):
            try: urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2); break
            except Exception: time.sleep(1)
        else:
            sys.exit("llama-server did not become healthy")

        def one(r):
            p = f"<start_of_turn>user\nTranslate the text to {NAMES[r['iso']]}:  {r['english'].strip()}<end_of_turn>\n<start_of_turn>model\n"
            o = post(a.port, "/completion", {"prompt": p, "n_predict": a.max_new, "temperature": 0.0, "top_k": 1, "cache_prompt": False, "stream": False})
            rec = {"iso": r["iso"], "id": r["id"], "raw": o["content"], "stop_type": o.get("stop_type"), "tokens": o.get("tokens_predicted"), "prompt_tokens": o.get("tokens_evaluated")}
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush(); n[0] += 1
                if n[0] % 100 == 0:
                    print(f"{n[0]}/{len(todo)}  {n[0]/(time.time()-t0):.2f}/s  last={r['iso']}", flush=True)

        with cf.ThreadPoolExecutor(a.parallel) as ex:
            list(ex.map(one, todo))
        print(f"finished {n[0]} rows in {time.time()-t0:.0f}s", flush=True)
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
