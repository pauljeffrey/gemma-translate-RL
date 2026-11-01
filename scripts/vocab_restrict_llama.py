"""Vocabulary restriction study, step 3c: llama.cpp with a PHYSICALLY pruned vocabulary.
  prune : vocab_restrict_llama.py prune SRC.gguf KEEP.npy DST.gguf      keep only the listed token ids (sorted; ids 0..106 must be inside so that bos/eos/eot/pad ids do not move):
          token_embd.weight (tied: input embedding AND output head) loses the other rows, tokenizer.ggml.tokens/scores/token_type are cut to the same ids. Works on Q8_0 rows byte-wise (no requantisation).
  bench : vocab_restrict_llama.py bench GGUF KEEP.npy|full SET.json ISO N THREADS BATCH OUTPREFIX [BIN]
          llama-server with B slots, prompts sent as token ids (original ids mapped to the pruned ids; a source token that was pruned away is re-spelled with byte tokens), greedy,
          generated ids mapped back. Writes OUTPREFIX.jsonl (hyp, ids) and OUTPREFIX.timing.json.  Needs PYTHONPATH with `gguf` for prune."""
import sys, os, json, time, subprocess, threading, urllib.request, queue
import numpy as np
mode = sys.argv[1]
if mode == "prune":
    import gguf
    src, keepf, dst = sys.argv[2:5]; keep = np.load(keepf).astype(np.int64); assert (keep[:107] == np.arange(107)).all() and (np.diff(keep) > 0).all()
    r = gguf.GGUFReader(src); arch = r.fields["general.architecture"].contents(); w = gguf.GGUFWriter(dst, arch)
    for k, f in r.fields.items():
        if k.startswith("GGUF.") or k == "general.architecture": continue
        v = f.contents()
        if k in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type"): v = [v[i] for i in keep]
        if f.types[0] == gguf.GGUFValueType.ARRAY: w.add_key_value(k, v, gguf.GGUFValueType.ARRAY, sub_type=f.types[-1])
        else: w.add_key_value(k, v, f.types[0])
    for t in r.tensors:
        data = t.data
        if t.name == "token_embd.weight": data = np.ascontiguousarray(data[keep])  # quantised rows: (rows, bytes per row)
        w.add_tensor(t.name, data, raw_shape=data.shape, raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
    print("wrote", dst, os.path.getsize(dst) / 1e6, "MB, vocab", len(keep))
else:
    from transformers import AutoTokenizer
    gguf_p, keepf, setf, iso, N, threads, B, outp = sys.argv[2:10]; N = int(N); threads = int(threads); B = int(B)
    BIN = sys.argv[10] if len(sys.argv) > 10 else "/workspace/jobs/c1/bin/llama-server"
    tok = AutoTokenizer.from_pretrained("/dev/shm/lc1_models/final2")
    rows = [r for r in json.load(open(setf)) if r["iso"] == iso][:N]
    if keepf == "full": o2n = None; n2o = None
    else:
        keep = np.load(keepf).astype(np.int64); o2n = {int(o): n for n, o in enumerate(keep)}; n2o = keep
    byte_id = {b: tok.convert_tokens_to_ids("<0x%02X>" % b) for b in range(256)}
    nspell = 0; ntot = 0
    def enc(ids):
        global nspell, ntot
        if o2n is None: return ids
        out = []
        for i in ids:
            ntot += 1
            if i in o2n: out.append(o2n[i])
            else:
                nspell += 1; out += [o2n[byte_id[b]] for b in tok.decode([i]).encode("utf-8")]
        return out
    prompts = [enc(r["ids"]) for r in rows]
    port = 19080 + (os.getpid() % 500)
    srv = subprocess.Popen([BIN, "-m", gguf_p, "-t", str(threads), "-tb", str(threads), "-np", str(B), "-c", str(1024 * B), "-b", "2048", "-ub", "512", "--port", str(port), "--host", "127.0.0.1", "-cb", "--no-webui", "--cache-ram", "0", "--no-warmup"],
                           stdout=open(outp + ".server.log", "a"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception:
            time.sleep(0.5)
            if srv.poll() is not None: raise SystemExit("server died: " + outp + ".server.log")
    load_s = time.time() - t0
    def req(ids, n):
        body = json.dumps({"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "cache_prompt": False, "stream": False, "return_tokens": True}).encode()
        return json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"}), timeout=7200).read())
    req(prompts[0], 4)
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i])); q = queue.Queue(); [q.put(i) for i in order]; out = {}
    def worker():
        while True:
            try: i = q.get_nowait()
            except queue.Empty: return
            out[i] = req(prompts[i], 400)
    ts = time.perf_counter(); th = [threading.Thread(target=worker) for _ in range(B)]; [t.start() for t in th]; [t.join() for t in th]; tot = time.perf_counter() - ts
    gen = 0
    with open(outp + ".jsonl", "w") as f:
        for i, r in enumerate(rows):
            ids = out[i].get("tokens") or []
            if n2o is not None: ids = [int(n2o[x]) for x in ids]
            n = len(ids)
            for e in (1, 106):
                if e in ids: n = min(n, ids.index(e) + 1)
            ids = ids[:n]; gen += n
            from gemma_mt import format as fmt
            raw = tok.decode(ids, skip_special_tokens=True)
            f.write(json.dumps({"iso": r["iso"], "id": str(r["id"]), "hyp": fmt.strip_iso_prefix(raw, r["iso"]).text, "ids": ids, "gen_tokens": n, "finished": n < 400}, ensure_ascii=False) + "\n")
    srv.terminate()
    rec = {"gguf": os.path.basename(gguf_p), "size_mb": round(os.path.getsize(gguf_p) / 1e6, 1), "iso": iso, "n": len(rows), "threads": threads, "batch": B, "load_s": round(load_s, 1), "total_s": round(tot, 2), "sent_per_s": round(len(rows) / tot, 3), "gen_tokens": gen, "tok_per_s": round(gen / tot, 1),
           "source_tokens_respelled_with_bytes": nspell, "source_tokens_total": ntot, "load": open("/proc/loadavg").read().split()[0], "ts": time.strftime("%FT%T")}
    open(outp + ".timing.jsonl", "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
