#!/usr/bin/env python3
"""Faster scoring with the SAME model as `model.predict` (AfriCOMET-QE-STL-1.1 = comet ReferencelessRegression: XLM-R encoder, layer "mix", avg pooling, two encoder passes, MLP on
[mt, src, mt*src, |mt-src|]). Nothing about the model changes; only HOW the batches are formed:

  1. the sentence embedding of a text does not depend on its batch (attention mask), so the source side and the translation side are encoded in SEPARATE length-sorted passes
     (comet.predict sorts by the length of the source only: the translation side of a batch is padded to its longest member);
  2. batches are made by a TOKEN budget (default 24,576 padded tokens) instead of a fixed number of sentences: no padding waste, constant GPU memory;
  3. identical English sources are encoded once: unique within the chunk and, in a process-wide cache (fp16 rows in CPU memory, `cache_gb`), across chunks and languages
     (the English side repeats a lot across the parallel languages of JW/Bible/Wikimedia style corpora);
  4. tokenisation is done once per chunk (fast tokenizer, batch call), batches are padded with numpy; CUDA kernels of batch i+1 are queued while batch i runs (no host sync in the loop).

The model weights, the tokenizer, the 510-token truncation, fp16 and the estimator are untouched, so the scores equal comet's up to the rounding of padding differences. `check` proves it:

    python qe_fast.py check --pairs <pair dir> --n 600 --device cuda --half      # Pearson / max |diff| of fast vs comet predict + the speed of both; exit 1 below --min-pearson 0.9999
"""
import argparse, os, sys, time
import numpy as np
import xxhash

MAX_TOKENS_DEFAULT = 0          # 0 = the model's own limit (510); a smaller number truncates (changes the scores of long pairs: not used by default)


def h64(s):
    return xxhash.xxh64_intdigest(s.encode("utf-8", "surrogatepass"))


class FastQE:
    def __init__(self, model, device, half, token_budget=24576, max_batch=1024, cache_gb=8.0, max_tokens=MAX_TOKENS_DEFAULT):
        import torch
        self.torch, self.m, self.dev, self.half = torch, model, torch.device(device), bool(half)
        self.budget, self.max_batch = int(token_budget), int(max_batch)
        self.tok = model.encoder.tokenizer
        self.max_len = (model.encoder.max_positions - 2) if not max_tokens else min(int(max_tokens), model.encoder.max_positions - 2)
        self.dim = model.encoder.output_units
        self.dtype = torch.float16 if self.half else torch.float32
        cap = int(cache_gb * 2 ** 30 / (self.dim * (2 if self.half else 4)))
        self.cache_cap = max(0, cap)
        self.cache_arr = torch.empty((min(self.cache_cap, 1 << 16), self.dim), dtype=self.dtype) if self.cache_cap else None       # grows by doubling up to cache_cap rows
        self.cache_idx = {}
        self.stats = dict(src_total=0, src_unique_chunk=0, src_cache_hits=0, src_encoded=0, mt_encoded=0, tokens_src=0, tokens_mt=0, padded_tokens=0)

    # ------------------------------------------------------------------------------------------------------------------------------------ encoder passes
    def _token_ids(self, texts):
        return self.tok(texts, truncation=True, max_length=self.max_len, add_special_tokens=True, return_attention_mask=False)["input_ids"]

    def _batches(self, lens):
        """index batches over a length-sorted order: (order slice) with padded size <= budget"""
        order = np.argsort(lens, kind="stable")
        out, start = [], 0
        n = len(order)
        while start < n:
            end = start + 1
            lo = int(lens[order[start]])
            while end < n and (end - start + 1) <= self.max_batch and int(lens[order[end]]) * (end - start + 1) <= self.budget and int(lens[order[end]]) <= 1.3 * lo + 8:      # token budget AND at most ~30% padding
                end += 1
            out.append(order[start:end]); start = end
        return out

    def encode(self, texts):
        """sentence embeddings (len(texts) x dim, on the device) of texts in the given order"""
        torch = self.torch
        ids = self._token_ids(texts)
        lens = np.fromiter((len(x) for x in ids), np.int64, len(ids))
        res = torch.empty((len(texts), self.dim), dtype=self.dtype, device=self.dev)
        pad = self.tok.pad_token_id
        for b in self._batches(lens):
            L = int(lens[b].max())
            arr = np.full((len(b), L), pad, np.int64); msk = np.zeros((len(b), L), np.int64)
            for r, i in enumerate(b):
                n = lens[i]; arr[r, :n] = ids[i]; msk[r, :n] = 1
            x = torch.from_numpy(arr).to(self.dev, non_blocking=True); mk = torch.from_numpy(msk).to(self.dev, non_blocking=True)
            with torch.inference_mode():
                e = self.m.get_sentence_embedding(x, mk)
            res[torch.from_numpy(b).to(self.dev)] = e.to(self.dtype)
            self.stats["padded_tokens"] += len(b) * L
        return res, int(lens.sum())

    def _cache_put(self, keys, emb):
        torch = self.torch
        if not self.cache_cap:
            return
        room = self.cache_cap - len(self.cache_idx)
        n = min(room, len(keys))
        if n <= 0:
            return
        need = len(self.cache_idx) + n
        if need > self.cache_arr.shape[0]:
            new = torch.empty((min(self.cache_cap, max(need, 2 * self.cache_arr.shape[0])), self.dim), dtype=self.dtype)
            new[:self.cache_arr.shape[0]] = self.cache_arr; self.cache_arr = new
        base = len(self.cache_idx)
        self.cache_arr[base:base + n] = emb[:n].to("cpu")
        for j in range(n):
            self.cache_idx[keys[j]] = base + j

    # ------------------------------------------------------------------------------------------------------------------------------------ scoring
    def score(self, srcs, mts):
        torch = self.torch
        n = len(srcs)
        hs = [h64(s) for s in srcs]
        uniq, inv, first = {}, np.empty(n, np.int64), []
        for i, k in enumerate(hs):
            j = uniq.get(k)
            if j is None:
                j = uniq[k] = len(uniq); first.append(i)
            inv[i] = j
        ukeys = list(uniq)
        emb_u = torch.empty((len(ukeys), self.dim), dtype=self.dtype, device=self.dev)
        miss = [j for j, k in enumerate(ukeys) if k not in self.cache_idx]
        hit = [j for j, k in enumerate(ukeys) if k in self.cache_idx]
        if hit:
            rows = torch.tensor([self.cache_idx[ukeys[j]] for j in hit])
            emb_u[torch.tensor(hit, device=self.dev)] = self.cache_arr[rows].to(self.dev)
        if miss:
            e, ntok = self.encode([srcs[first[j]] for j in miss])
            emb_u[torch.tensor(miss, device=self.dev)] = e
            self._cache_put([ukeys[j] for j in miss], e)
            self.stats["tokens_src"] += ntok
        self.stats["src_total"] += n; self.stats["src_unique_chunk"] += len(ukeys); self.stats["src_cache_hits"] += len(hit); self.stats["src_encoded"] += len(miss)
        mt_emb, ntok = self.encode(list(mts))
        self.stats["mt_encoded"] += n; self.stats["tokens_mt"] += ntok
        src_emb = emb_u[torch.from_numpy(inv).to(self.dev)]
        out = []
        with torch.inference_mode():
            for s in range(0, n, 8192):
                a, b = mt_emb[s:s + 8192], src_emb[s:s + 8192]
                x = torch.cat((a, b, a * b, torch.abs(a - b)), dim=1)
                out.append(self.m.estimator(x).view(-1).float())
        return torch.cat(out).cpu().numpy().astype(np.float32)

    def summary(self):
        s = self.stats
        return (f"src {s['src_total']:,} pairs -> {s['src_unique_chunk']:,} unique in chunk -> {s['src_encoded']:,} encoded ({100 * s['src_cache_hits'] / max(1, s['src_unique_chunk']):.1f}% cache hits); "
                f"mt {s['mt_encoded']:,} encoded; real tokens {s['tokens_src'] + s['tokens_mt']:,}, padded {s['padded_tokens']:,} "
                f"({100 * (s['tokens_src'] + s['tokens_mt']) / max(1, s['padded_tokens']):.0f}% of the padded tokens are real)")


def comet_scores(model, srcs, mts, batch, gpus, workers=2):
    res = model.predict([{"src": e, "mt": t} for e, t in zip(srcs, mts)], batch_size=batch, gpus=gpus, progress_bar=False, num_workers=workers)
    return np.asarray(res.scores if hasattr(res, "scores") else res, np.float32)


def check(a):
    import torch, pyarrow.parquet as pq, glob
    from comet import download_model, load_from_checkpoint
    t = None
    for f in sorted(glob.glob(os.path.join(a.pairs, "*.parquet"))):
        tb = pq.read_table(f, columns=["english", "target"]).to_pydict()
        t = tb if t is None else {k: t[k] + tb[k] for k in t}
        if len(t["english"]) >= a.n * 3:
            break
    rng = np.random.default_rng(0); idx = rng.choice(len(t["english"]), size=min(a.n, len(t["english"])), replace=False)
    srcs = [t["english"][i] for i in idx]; mts = [t["target"][i] for i in idx]
    srcs = srcs + srcs[: len(srcs) // 4]; mts = mts + [m + " " for m in mts[: len(mts) // 4]]            # repeated English sources exercise the cache; the changed target keeps the pairs distinct
    model = load_from_checkpoint(download_model("masakhane/africomet-qe-stl-1.1")); model.eval()
    dev = a.device
    if dev == "cuda":
        assert torch.cuda.is_available()
        model.to("cuda")
    if a.half and dev == "cuda":
        model.half()
    fq = FastQE(model, dev, a.half and dev == "cuda", token_budget=a.token_budget, cache_gb=a.cache_gb)
    t0 = time.time(); ref = comet_scores(model, srcs, mts, a.batch, 1 if dev == "cuda" else 0); t_ref = time.time() - t0
    if dev == "cuda":
        torch.cuda.synchronize(); model.to("cuda")                 # Lightning moves the module back to the CPU after predict
    t0 = time.time(); fast = fq.score(srcs, mts)
    if dev == "cuda": torch.cuda.synchronize()
    t_fast = time.time() - t0
    t0 = time.time(); fast2 = fq.score(srcs, mts); t_fast2 = time.time() - t0                       # second call: every English source is a cache hit
    pear = float(np.corrcoef(ref, fast)[0, 1]); mx = float(np.abs(ref - fast).max()); pear2 = float(np.corrcoef(ref, fast2)[0, 1])
    print(f"n={len(srcs)} device={dev} half={a.half}: Pearson {pear:.7f} (second call, cached sources {pear2:.7f}), max|diff| {mx:.5f}, mean|diff| {float(np.abs(ref - fast).mean()):.5f}")
    print(f"comet predict {len(srcs) / t_ref:.1f} pairs/s | fast {len(srcs) / t_fast:.1f} pairs/s (x{t_ref / t_fast:.2f}) | fast with all sources cached {len(srcs) / t_fast2:.1f} pairs/s")
    print(fq.summary())
    if pear < a.min_pearson or pear2 < a.min_pearson:
        print(f"FAIL: Pearson below {a.min_pearson}; do not use the fast path"); sys.exit(1)
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--pairs", required=True); c.add_argument("--n", type=int, default=600); c.add_argument("--device", default="cpu"); c.add_argument("--half", action="store_true")
    c.add_argument("--batch", type=int, default=64); c.add_argument("--token-budget", type=int, default=24576); c.add_argument("--cache-gb", type=float, default=1.0); c.add_argument("--min-pearson", type=float, default=0.9999)
    a = ap.parse_args()
    check(a)
