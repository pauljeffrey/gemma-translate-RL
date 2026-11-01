#!/usr/bin/env python3
"""swap_vocab.py -- Gemma-3 vocabulary swap (token-id reuse): free list, new Ge'ez / Fon BPE merges, swapped
tokenizer.json, model surgery.  CPU only.  Written 2026-10-25 (agent T1).  Copy on the box: /workspace/tokswap/swap_vocab.py.

Sub-commands (all deterministic; see the report reports/token_swap_build_2026-10-25.md for the exact commands):

  classify   tokenizer.json -> free_pool.json + token_classes.tsv.gz + class_counts.csv
             Unicode-script classification of the 262,144 ids and the strict "never needed" counts (scenarios S1-S4, REPORTED ONLY,
             nothing is removed).  The ids actually re-used are the 6,143 `<unusedN>` ids 256001..262143 (tail), which no tokenizer
             output ever contained: Ge'ez tokens take them ascending from 256001, Fon tokens descending from 262143.
  train      tokenizer.json + text files -> merges_<kind>.json
             BPE merge learning ON TOP of the original tokenization (continuation of the original merge list).
             kind=geez: result tokens are pure Ethiopic (optional leading U+2581); kind=fon: result tokens are Latin letters /
             combining marks (optional leading U+2581) that contain at least one Fon-specific character.
  build      tokenizer.json + merges + free_pool.json -> swapped tokenizer dir (+ swap_map.json)
  surgery    checkpoint dir + swap_map.json + original tokenizer -> swapped checkpoint (bf16 safetensors)
  simcheck   verifies that the merge-learning simulation equals the real `tokenizers` BPE on a text sample

Why the construction is safe (proved by construction, then TESTED by tokswap_eval.py):
  * New merges are appended AFTER all original merges, so on any text the original merges are applied first (HF BPE applies
    the lowest-rank pair first) and the new ones only act on what is left.
  * Every new token contains at least one Ethiopic character (geez) or one Fon-specific character (fon); one of the two parts of
    its merge then contains it too, so a new merge can only fire on text that contains such a character.  Text without them
    is segmented exactly as before.
  * Re-used ids are `<unusedN>` ids (never produced by the tokenizer, in no merge, no special/template token); `build` asserts this.
    Their entries are deleted from model.vocab and added_tokens, so nothing else in the tokenizer changes.
"""
import argparse, collections, csv, gzip, hashlib, heapq, json, os, re, sys, time, unicodedata

SP = "▁"
UNUSED_RE = re.compile(r"^<unused\d+>$")
POOL_SIZE = 16000          # size of the OPTIONAL extra pool (leaf tokens of freed scripts); reported only, never used by `build`
# Fon-specific characters: \u0256 \u0189 \u025b \u0190 \u0254 \u0186 \u014b \u014a (d-tail, open e, open o, eng, and capitals), the hacek tone vowels and the combining hacek U+030C
FON_TRIGGERS = "\u0256\u0189\u025b\u0190\u0254\u0186\u014b\u014a\u01cd\u01ce\u011a\u011b\u01cf\u01d0\u01d1\u01d2\u01d3\u01d4\u030c"

# ----------------------------------------------------------------------------------------------- script classification
SCRIPT_BY_NAME_PREFIX = [
    ("LATIN", "latin"), ("CYRILLIC", "cyrillic"), ("GREEK", "greek"), ("ARABIC", "arabic"), ("HEBREW", "hebrew"),
    ("DEVANAGARI", "devanagari"), ("BENGALI", "bengali"), ("GURMUKHI", "gurmukhi"), ("GUJARATI", "gujarati"), ("ORIYA", "oriya"),
    ("TAMIL", "tamil"), ("TELUGU", "telugu"), ("KANNADA", "kannada"), ("MALAYALAM", "malayalam"), ("SINHALA", "sinhala"),
    ("THAI", "thai"), ("LAO", "lao"), ("TIBETAN", "tibetan"), ("MYANMAR", "myanmar"), ("GEORGIAN", "georgian"), ("HANGUL", "hangul"),
    ("ETHIOPIC", "ethiopic"), ("KHMER", "khmer"), ("MONGOLIAN", "mongolian"), ("CJK", "cjk"), ("IDEOGRAPHIC", "cjk"),
    ("HIRAGANA", "hiragana"), ("KATAKANA", "katakana"), ("BOPOMOFO", "bopomofo"), ("ARMENIAN", "armenian"), ("THAANA", "thaana"),
    ("SYRIAC", "syriac"), ("COPTIC", "coptic"), ("TIFINAGH", "tifinagh"), ("CHEROKEE", "cherokee"), ("NKO", "nko"), ("OGHAM", "ogham"),
    ("RUNIC", "runic"), ("CANADIAN", "canadian_syllabics"), ("FULLWIDTH LATIN", "latin"), ("HALFWIDTH KATAKANA", "katakana"),
    ("FULLWIDTH", "common"), ("HALFWIDTH", "common"), ("VAI", "vai"), ("ADLAM", "adlam"), ("OSMANYA", "osmanya"), ("BAMUM", "bamum"),
]
# KEEP scripts: needed by the project (English / French / 77 target languages: Latin + Ethiopic; Arabic source items;
# Greek letters as science / LaTeX symbols; Tifinagh / N'Ko / Vai / Adlam / Osmanya / Bamum because African scripts of possible
# target orthographies are never freed even though the current 77 targets do not use them).
KEEP_SCRIPTS = {"latin", "ethiopic", "arabic", "greek", "tifinagh", "nko", "vai", "adlam", "osmanya", "bamum"}
# FREE scripts in CONSUMPTION PRIORITY order (first = freed first).  cyrillic / hebrew are the reserve tail.
FREE_PRIORITY = ["cjk", "hangul", "hiragana", "katakana", "bopomofo", "thai", "lao", "khmer", "myanmar", "tibetan", "sinhala", "tamil",
                 "malayalam", "telugu", "kannada", "gujarati", "gurmukhi", "oriya", "bengali", "devanagari", "georgian", "armenian",
                 "thaana", "mongolian", "cherokee", "syriac", "coptic", "ogham", "runic", "canadian_syllabics", "cyrillic", "hebrew"]
FREE_SCRIPTS = set(FREE_PRIORITY)
BYTE_RE = re.compile(r"^<0x[0-9A-F]{2}>$")
_script_cache = {}


def char_script(c):
    """script key of one character, or None for common/inherited characters (punctuation, digits, symbols, marks, spaces)."""
    r = _script_cache.get(c)
    if r is not None:
        return r or None
    cat = unicodedata.category(c)
    if cat[0] in "NPSZC" or cat[0] == "M":
        res = ""
    else:
        try:
            nm = unicodedata.name(c)
        except ValueError:
            nm = ""
        res = ""
        for pre, key in SCRIPT_BY_NAME_PREFIX:
            if nm.startswith(pre):
                res = key if key != "common" else ""
                break
        else:
            res = "other:" + (nm.split(" ")[0].lower() if nm else "unnamed")
    _script_cache[c] = res
    return res or None


def token_scripts(s):
    return {x for x in map(char_script, s.replace(SP, "")) if x}


def classify_token(s, added):
    if added:
        return "added_special_unused"
    if BYTE_RE.match(s):
        return "byte"
    sc = token_scripts(s)
    if not sc:
        return "common_ascii" if all(ord(c) < 128 for c in s) else "common_nonascii"
    if len(sc) == 1:
        return next(iter(sc))
    return "mixed:" + "+".join(sorted(sc))


def is_free_class(cls):
    """FREE iff every script of the token is a FREE script (mixed tokens containing a KEEP script are kept)."""
    if cls.startswith("mixed:"):
        parts = cls[6:].split("+")
        return all(p in FREE_SCRIPTS for p in parts)
    return cls in FREE_SCRIPTS


def free_priority_of(cls):
    parts = cls[6:].split("+") if cls.startswith("mixed:") else [cls]
    return min(FREE_PRIORITY.index(p) for p in parts)


# ----------------------------------------------------------------------------------------------- helpers
def load_tok(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_pairs(d):
    return [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in d["model"]["merges"]]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reachable(vocab, merges, removed_ids, base_ids):
    """set of token strings reachable: base tokens + outputs of merges whose both inputs are reachable (and not removed)."""
    removed = set(removed_ids)
    ok = {s for s, i in vocab.items() if i in base_ids and i not in removed}
    pend = [(a, b, a + b) for a, b in merges]
    changed = True
    while changed:
        changed = False
        nxt = []
        for a, b, t in pend:
            if t in ok:
                continue
            if a in ok and b in ok and t in vocab and vocab[t] not in removed:
                ok.add(t); changed = True
            else:
                nxt.append((a, b, t))
        pend = nxt
    return ok


# ----------------------------------------------------------------------------------------------- classify
def unused_tail(vocab):
    return sorted(i for s_, i in vocab.items() if UNUSED_RE.match(s_) and i > 105)


def cmd_classify(a):
    d = load_tok(a.tok)
    vocab = d["model"]["vocab"]
    added = {t["id"]: t for t in d["added_tokens"]}
    id2s = {i: s for s, i in vocab.items()}
    merges = merge_pairs(d)
    used_as_input = set()
    for x, y in merges:
        used_as_input.add(x); used_as_input.add(y)
    tail = unused_tail(vocab)
    tail_set = set(tail)
    in_merges = [id2s[i] for i in tail if id2s[i] in used_as_input]
    rows = []
    for i in range(len(vocab)):
        s = id2s[i]
        rows.append((i, s, classify_token(s, i in added)))
    counts = collections.Counter(c for _, _, c in rows)
    # strict "never needed" scenarios (S0 definitions): a token is freeable if it has a letter outside the kept scripts
    def scen(keep):
        n = 0; per = collections.Counter()
        for i, s, c in rows:
            if c in ("added_special_unused", "byte", "common_ascii", "common_nonascii"):
                continue
            sc = set(c[6:].split("+")) if c.startswith("mixed:") else {c}
            if not sc <= keep:
                n += 1; per[c] += 1
        return n, per
    base = {"latin", "ethiopic"}
    scenarios = {}
    for name, keep in (("S1 keep Latin+Ethiopic", base), ("S2 = S1 + Arabic", base | {"arabic"}), ("S3 = S2 + Greek", base | {"arabic", "greek"}),
                       ("S4 = S3 + Cyrillic + Hebrew", base | {"arabic", "greek", "cyrillic", "hebrew"}),
                       ("OUR policy (KEEP_SCRIPTS)", set(KEEP_SCRIPTS))):
        n, per = scen(keep)
        scenarios[name] = {"freeable_letter_tokens": n, "plus_unused_tail": n + len(tail), "share_of_vocab_incl_tail": round((n + len(tail)) / len(vocab), 4)}
    free_rows = [(i, s, c) for i, s, c in rows if is_free_class(c)]
    leaf = [(i, s, c) for i, s, c in free_rows if s not in used_as_input]
    order = sorted(leaf, key=lambda r: (free_priority_of(r[2]), -r[0]))
    pool = order[:POOL_SIZE]
    free_by_class = collections.Counter(c for _, _, c in free_rows)
    leaf_by_class = collections.Counter(c for _, _, c in leaf)
    pool_by_class = collections.Counter(c for _, _, c in pool)
    os.makedirs(a.out, exist_ok=True)
    with gzip.open(os.path.join(a.out, "token_classes.tsv.gz"), "wt", encoding="utf-8") as f:
        f.write("id\ttoken\tclass\tfree_script\tleaf\tunused_tail\n")
        for i, s, c in rows:
            f.write(f"{i}\t{s!r}\t{c}\t{int(is_free_class(c))}\t{int(is_free_class(c) and s not in used_as_input)}\t{int(i in tail_set)}\n")
    with open(os.path.join(a.out, "class_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "tokens", "free_script", "free_script_leaf"])
        for c, n in counts.most_common():
            w.writerow([c, n, free_by_class.get(c, 0), leaf_by_class.get(c, 0)])
    out = {
        "tokenizer_sha256": sha256_file(a.tok), "vocab_size": len(vocab), "n_merges": len(merges),
        "NOTHING_IS_REMOVED": "slots for new tokens are the unused tail only; the freeable counts below are reporting only",
        "unused_tail": {"n": len(tail), "first_id": tail[0], "last_id": tail[-1], "also_in_merges": len(in_merges), "names": f"{id2s[tail[0]]}..{id2s[tail[-1]]}",
                        "all_in_added_tokens": all(i in added for i in tail), "low_unused_block_ids_6_104_kept": True},
        "slots": {"geez": "tail ascending from %d" % tail[0], "fon": "tail descending from %d" % tail[-1]},
        "policy_for_scenarios": {
            "KEEP_never_touched": "all added/special/unused tokens (incl. <start_of_turn>=105, <end_of_turn>=106, <bos>, <pad>, <eos>, <unused0..98>, html markup), byte tokens <0xHH>, "
                                  "tokens with no letter (digits, punctuation, symbols, emoji, whitespace pieces), always kept in every scenario",
            "freeable": "a token with at least one letter of a script outside the kept set (mixed tokens count as freeable); counts exclude the unused tail, "
                        "plus_unused_tail adds the 6,143 tail ids (S0 convention)",
        },
        "scenarios": scenarios,
        "counts_by_class": dict(counts.most_common()),
        "free_script_policy_tokens": len(free_rows), "free_script_leaf_tokens": len(leaf),
        "free_by_script_group": {
            "cjk_ideograph_and_mixed_kana": sum(n for c, n in free_by_class.items() if c.startswith("cjk") or c.startswith("mixed:") and "cjk" in c),
            "hangul": free_by_class.get("hangul", 0), "kana": free_by_class.get("hiragana", 0) + free_by_class.get("katakana", 0),
            "thai": free_by_class.get("thai", 0), "devanagari": free_by_class.get("devanagari", 0), "bengali": free_by_class.get("bengali", 0),
            "cyrillic": free_by_class.get("cyrillic", 0), "tamil": free_by_class.get("tamil", 0),
        },
        "optional_extra_pool_NOT_USED": {"what": "leaf tokens of freed scripts (cjk first, ids descending); needed only for N beyond the 6,143 unused ids (e.g. 8,000) and would require removing those tokens", "size": len(pool), "by_class": dict(pool_by_class), "ids": [i for i, _, _ in pool]},
    }
    json.dump(out, open(os.path.join(a.out, "free_pool.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("optional_extra_pool_NOT_USED", "counts_by_class")}, ensure_ascii=False, indent=1))


# ----------------------------------------------------------------------------------------------- train merges
ETH_RANGES = [(0x1200, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)]


def is_eth(c):
    o = ord(c)
    return any(lo <= o <= hi for lo, hi in ETH_RANGES)


def geez_part_ok(s):
    """token = optional leading SP then Ethiopic characters only"""
    body = s[1:] if s.startswith(SP) else s
    return all(is_eth(c) for c in body)


def geez_result_ok(s):
    body = s[1:] if s.startswith(SP) else s
    return len(body) >= 1 and all(is_eth(c) for c in body)


def fon_char_ok(c):
    cat = unicodedata.category(c)
    return cat[0] == "M" or (cat[0] == "L" and (char_script(c) == "latin"))


def fon_part_ok(s):
    body = s[1:] if s.startswith(SP) else s
    return all(fon_char_ok(c) for c in body)


def fon_result_ok(s):
    body = s[1:] if s.startswith(SP) else s
    return len(body) >= 2 and all(fon_char_ok(c) for c in body) and any(c in FON_TRIGGERS for c in body)


KINDS = {"geez": (geez_part_ok, geez_result_ok), "fon": (fon_part_ok, fon_result_ok)}


def read_lines(path, max_chars):
    n = 0
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                out.append(ln); n += len(ln)
                if n >= max_chars:
                    break
    return out


def runs_from_tokens(toks, id2s, part_ok):
    """split one token sequence into mergeable runs: consecutive part_ok tokens; a token that starts with SP starts a new run."""
    runs, cur = [], []
    for t in toks:
        s = id2s[t]
        if part_ok(s):
            if s.startswith(SP) and cur:
                runs.append(cur); cur = []
            cur.append(t)
        else:
            if cur:
                runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return runs


def cmd_train(a):
    from tokenizers import Tokenizer
    part_ok, result_ok = KINDS[a.kind]
    d = load_tok(a.tok)
    vocab = dict(d["model"]["vocab"])
    tk = Tokenizer.from_str(json.dumps(d))
    id2s = [None] * (max(vocab.values()) + 1)
    for s, i in vocab.items():
        id2s[i] = s
    word_counts = collections.Counter()
    stats = {}
    t0 = time.time()
    chars = []
    if a.kind == "geez":
        # step 0: Ethiopic letters without a single-character token (they are spelled with 3 byte tokens today)
        cc = collections.Counter()
        for path in a.text:
            for ln in read_lines(path, a.max_chars):
                cc.update(c for c in set(ln) if is_eth(c) and unicodedata.category(c) == "Lo")
        chars = [c for c, n in cc.most_common() if c not in vocab and n >= a.char_min_count]
        chars = chars[:a.n]
        base_id = max(vocab.values()) + 1
        d = json.loads(json.dumps(d))
        for k, c in enumerate(chars):
            d["model"]["vocab"][c] = base_id + k
            vocab[c] = base_id + k
        tk = Tokenizer.from_str(json.dumps(d))
        id2s = [None] * (max(vocab.values()) + 1)
        for s_, i_ in vocab.items():
            id2s[i_] = s_
        print(f"step 0: {len(chars)} Ethiopic letters without a token get a single-character token (min count {a.char_min_count})", flush=True)
    for spec in a.text:
        path, _, cap = spec.partition("@")      # optional per-file character cap: path@12000000
        lines = read_lines(path, int(cap) if cap else a.max_chars)
        nchar = sum(map(len, lines))
        ntok = 0
        for k in range(0, len(lines), 20000):
            for e in tk.encode_batch(lines[k:k + 20000], add_special_tokens=False):
                ntok += len(e.ids)
                for r in runs_from_tokens(e.ids, id2s, part_ok):
                    if len(r) >= 2:
                        word_counts[tuple(r)] += 1
        stats[os.path.basename(path)] = {"lines": len(lines), "chars": nchar, "orig_tokens": ntok}
        print(f"[{time.time()-t0:.0f}s] {path}: {len(lines)} lines, {nchar} chars, {ntok} tokens; word types so far {len(word_counts)}", flush=True)
    words = [list(w) for w in word_counts]
    counts = [word_counts[tuple(w)] for w in words]
    # local string table (original ids + new ids >= base)
    S = list(id2s)
    s2i = dict(vocab)
    pair_cnt = collections.Counter()
    where = collections.defaultdict(set)
    valid_cache = {}

    def valid(x, y):
        k = (x, y)
        r = valid_cache.get(k)
        if r is None:
            t = S[x] + S[y]
            r = result_ok(t) and t not in s2i
            # also: the left part may not end the word with SP etc. (parts are part_ok by construction)
            valid_cache[k] = r
        return r

    def add_word(wi):
        w = words[wi]; c = counts[wi]
        for p in range(len(w) - 1):
            x, y = w[p], w[p + 1]
            if valid(x, y):
                pair_cnt[(x, y)] += c
                where[(x, y)].add(wi)

    for wi in range(len(words)):
        add_word(wi)
    heap = [(-c, x, y) for (x, y), c in pair_cnt.items()]
    heapq.heapify(heap)
    merges_out = []
    n_merges_wanted = a.n - len(chars)
    while heap and len(merges_out) < n_merges_wanted:
        negc, x, y = heapq.heappop(heap)
        cur = pair_cnt.get((x, y), 0)
        if cur <= 0:
            continue
        if cur != -negc:
            heapq.heappush(heap, (-cur, x, y)); continue
        if cur < a.min_count:
            break
        t = S[x] + S[y]
        if t in s2i or not result_ok(t):
            pair_cnt[(x, y)] = 0; continue
        nid = len(S); S.append(t); s2i[t] = nid
        merges_out.append({"a": S[x], "b": S[y], "tok": t, "count": cur})
        touched = list(where[(x, y)])
        changed_pairs = set()
        for wi in touched:
            w = words[wi]; c = counts[wi]
            if x not in w:
                continue
            # remove old contributions
            for p in range(len(w) - 1):
                k = (w[p], w[p + 1])
                if k in pair_cnt:
                    pair_cnt[k] -= c
            new = []; p = 0; L = len(w)
            while p < L:
                if p < L - 1 and w[p] == x and w[p + 1] == y:
                    new.append(nid); p += 2
                else:
                    new.append(w[p]); p += 1
            words[wi] = new
            for p in range(len(new) - 1):
                k = (new[p], new[p + 1])
                if valid(*k):
                    pair_cnt[k] += c
                    where[k].add(wi)
                    changed_pairs.add(k)
        pair_cnt[(x, y)] = 0
        where.pop((x, y), None)
        for k in changed_pairs:
            if pair_cnt[k] > 0:
                heapq.heappush(heap, (-pair_cnt[k], k[0], k[1]))
        if len(merges_out) % 500 == 0:
            print(f"[{time.time()-t0:.0f}s] merges {len(merges_out)}  last count {cur}  {t}", flush=True)
    out = {"kind": a.kind, "n_requested_total": a.n, "n_chars": len(chars), "n_learned": len(merges_out), "min_count": a.min_count, "chars": chars,
           "N_semantics": "N = single-letter tokens (chars) + merges, as in S0; first tokens of a variant = chars, then merges in learning order", "tokenizer_sha256": sha256_file(a.tok),
           "corpus": stats, "word_types": len(word_counts), "merges": merges_out}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"learned {len(merges_out)} merges; last count {merges_out[-1]['count'] if merges_out else None}; wrote {a.out}")


# ----------------------------------------------------------------------------------------------- build
def plan_for(kind, path, n, tail):
    m = json.load(open(path, encoding="utf-8"))
    assert m["kind"] == kind
    chars = m.get("chars", [])
    items = [(c, None, None, "geez-letter", 0) for c in chars[:n]]
    nm = n - len(items)
    if nm > len(m["merges"]):
        sys.exit(f"{kind}: asked N={n} (= {len(items)} letters + {nm} merges), file has {len(m['merges'])} merges")
    items += [(mm["tok"], mm["a"], mm["b"], kind, mm["count"]) for mm in m["merges"][:nm]]
    ids = tail[:len(items)] if kind == "geez" else tail[::-1][:len(items)]
    if len(items) > len(tail):
        sys.exit(f"{kind}: N={n} exceeds the {len(tail)} unused ids (beyond that tokens would have to be removed: not done)")
    return [(t, x, y, i, k, c) for (t, x, y, k, c), i in zip(items, ids)]


def cmd_build(a):
    d = load_tok(a.tok)
    vocab = d["model"]["vocab"]
    merges = merge_pairs(d)
    tail = unused_tail(vocab)
    id2s = {i: s for s, i in vocab.items()}
    plan = []
    if a.geez and a.n_geez:
        plan += plan_for("geez", a.geez, a.n_geez, tail)
    if a.fon and a.n_fon:
        plan += plan_for("fon", a.fon, a.n_fon, tail)
    ids = [p[3] for p in plan]
    assert len(set(ids)) == len(ids), "Ge'ez and Fon slots overlap (N too large for a combined variant)"
    removed_ids = set(ids)
    inputs = {x for ab in merges for x in ab}
    for i in removed_ids:
        assert UNUSED_RE.match(id2s[i]) and i > 105, id2s[i]
        assert id2s[i] not in inputs
    for tok, x, y, i, kind, cnt in plan:
        assert tok not in vocab, f"new token {tok!r} already in vocabulary"
    nv = {s: i for s, i in vocab.items() if i not in removed_ids}
    for tok, x, y, i, kind, cnt in plan:
        nv[tok] = i
    assert len(nv) == len(vocab) and sorted(nv.values()) == sorted(vocab.values())
    appended = []
    for tok, x, y, i, kind, cnt in plan:
        if x is not None:
            assert x in nv and y in nv and x + y == tok, (tok, x, y)
            appended.append([x, y])
    d2 = json.loads(json.dumps(d))
    d2["model"]["vocab"] = dict(sorted(nv.items(), key=lambda kv: kv[1]))
    d2["model"]["merges"] = [list(m) for m in merges] + appended
    n_added_before = len(d2["added_tokens"])
    d2["added_tokens"] = [t for t in d2["added_tokens"] if t["id"] not in removed_ids]
    os.makedirs(a.out, exist_ok=True)
    json.dump(d2, open(os.path.join(a.out, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    srcdir = os.path.dirname(a.tok)
    for fn in ("tokenizer_config.json", "chat_template.jinja", "generation_config.json", "special_tokens_map.json"):
        src = os.path.join(srcdir, fn)
        if not os.path.exists(src):
            continue
        if fn == "tokenizer_config.json":
            c = json.load(open(src, encoding="utf-8"))
            if "added_tokens_decoder" in c:
                c["added_tokens_decoder"] = {k: v for k, v in c["added_tokens_decoder"].items() if int(k) not in removed_ids}
            json.dump(c, open(os.path.join(a.out, fn), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        else:
            open(os.path.join(a.out, fn), "wb").write(open(src, "rb").read())
    smap = {"orig_tokenizer_sha256": sha256_file(a.tok), "n_geez": a.n_geez if a.geez else 0, "n_fon": a.n_fon if a.fon else 0,
            "removed_tokens": [{"id": i, "token": id2s[i]} for i in sorted(removed_ids)],
            "removed_merges": 0, "appended_merges": len(appended), "added_tokens_entries_removed": n_added_before - len(d2["added_tokens"]),
            "new_tokens": [{"id": i, "token": t, "a": x, "b": y, "kind": k, "count": c} for t, x, y, i, k, c in plan]}
    json.dump(smap, open(os.path.join(a.out, "swap_map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"built {a.out}: {len(plan)} new tokens in {len(removed_ids)} unused ids (ids {min(ids)}..{max(ids)}), 0 merges removed, {len(appended)} appended; "
          f"vocab size {len(nv)}; sha256 {sha256_file(os.path.join(a.out, 'tokenizer.json'))}")


# ----------------------------------------------------------------------------------------------- surgery
def cmd_surgery(a):
    import torch
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer
    smap = json.load(open(os.path.join(a.swap, "swap_map.json"), encoding="utf-8"))
    orig = Tokenizer.from_file(a.orig_tok)
    sd = load_file(os.path.join(a.model, "model.safetensors"))
    key = [k for k in sd if k.endswith("embed_tokens.weight")]
    assert len(key) == 1, key
    key = key[0]
    E = sd[key].to(torch.float32)
    tied = not any(k.endswith("lm_head.weight") for k in sd)
    assert E.shape[0] == 262144
    newE = E.clone()
    rules = []
    for nt in smap["new_tokens"]:
        txt = nt["token"].replace(SP, " ")
        ids = orig.encode(txt, add_special_tokens=False).ids
        assert ids, nt
        newE[nt["id"]] = E[ids].mean(0)
        rules.append({"id": nt["id"], "token": nt["token"], "orig_piece_ids": ids, "orig_pieces": [orig.id_to_token(i) for i in ids]})
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    out = {}
    for k, v in sd.items():
        out[k] = (newE if k == key else v.to(torch.float32)).to(dt).contiguous()
    if not tied:
        lk = [k for k in sd if k.endswith("lm_head.weight")][0]
        out[lk] = newE.to(dt).contiguous()
    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    for fn in ("config.json", "generation_config.json"):
        src = os.path.join(a.model, fn)
        if os.path.exists(src):
            open(os.path.join(a.out, fn), "wb").write(open(src, "rb").read())
    tokdir = a.orig_tok_dir if a.control else a.swap
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        src = os.path.join(tokdir, fn)
        if os.path.exists(src):
            open(os.path.join(a.out, fn), "wb").write(open(src, "rb").read())
    json.dump({"rule": "new-token embedding row = mean (float32) of the ORIGINAL embedding rows of the sub-tokens that the ORIGINAL tokenizer produces "
                       "for the new token's string (leading U+2581 -> space); input and output embeddings are tied (no lm_head tensor), so the output row is the same",
               "tied_embeddings": tied, "dtype": a.dtype, "control": bool(a.control), "n_rows_reinitialised": len(rules), "rows": rules},
              open(os.path.join(a.out, "surgery.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"wrote {a.out} ({a.dtype}, {len(rules)} rows re-initialised, tied={tied}, control={bool(a.control)}); sha256 {sha256_file(os.path.join(a.out, 'model.safetensors'))}")


# ----------------------------------------------------------------------------------------------- SentencePiece tokenizer.model
def cmd_spm(a):
    """same ids in tokenizer.model (the file llama.cpp's converter reads): piece = new string, type NORMAL, score below every existing piece
    (minscore - 1 - rank), as validated by S0."""
    from sentencepiece import sentencepiece_model_pb2 as pb
    smap = json.load(open(os.path.join(a.swap, "swap_map.json"), encoding="utf-8"))
    proto = pb.ModelProto(); proto.ParseFromString(open(a.spm, "rb").read())
    minscore = min(p.score for p in proto.pieces)
    for k, nt in enumerate(smap["new_tokens"]):
        pc = proto.pieces[nt["id"]]
        assert pc.piece.startswith("<unused") and pc.type == 4, (nt, pc.piece, pc.type)
        pc.piece = nt["token"]; pc.type = 1; pc.score = minscore - 1 - k
    open(os.path.join(a.swap, "tokenizer.model"), "wb").write(proto.SerializeToString())
    print(f"wrote {os.path.join(a.swap, 'tokenizer.model')} sha256 {sha256_file(os.path.join(a.swap, 'tokenizer.model'))}")


# ----------------------------------------------------------------------------------------------- simcheck
def cmd_simcheck(a):
    """apply the learned merges (as the trainer does, per run, in order, left to right) to the original segmentation of a text sample and
    compare with the real swapped tokenizer."""
    from tokenizers import Tokenizer
    part_ok, _ = KINDS[a.kind]
    od = load_tok(a.orig); sd = load_tok(os.path.join(a.swapped, "tokenizer.json"))
    ot = Tokenizer.from_str(json.dumps(od)); st = Tokenizer.from_str(json.dumps(sd))
    ov = od["model"]["vocab"]; id2s = {i: s for s, i in ov.items()}
    mj = json.load(open(a.merges, encoding="utf-8"))
    m = mj["merges"][:max(0, a.n - len(mj.get("chars", [])))] if a.n_is_total else mj["merges"][:a.n]
    new_letters = set(mj.get("chars", [])[:a.n])   # lines containing these are not simulated (the original tokenizer spells them with 3 byte tokens)
    rank = {(mm["a"], mm["b"]): r for r, mm in enumerate(m)}
    lines = read_lines(a.text, a.max_chars)
    n_all = len(lines)
    lines = [l for l in lines if not (new_letters & set(l))]
    skipped = n_all - len(lines)
    bad = tot = 0
    for k in range(0, len(lines), 5000):
        batch = lines[k:k + 5000]
        for ln, eo, es in zip(batch, ot.encode_batch(batch, add_special_tokens=False), st.encode_batch(batch, add_special_tokens=False)):
            toks = [id2s[i] for i in eo.ids]
            # simulate on runs (rank-based BPE restricted to the new merges: lowest rank first, leftmost occurrence first)
            out = []; cur = []
            def flush():
                nonlocal cur
                if cur:
                    w = cur
                    while len(w) > 1:
                        best = None
                        for p in range(len(w) - 1):
                            r = rank.get((w[p], w[p + 1]))
                            if r is not None and (best is None or r < best[0]):
                                best = (r, p)
                        if best is None:
                            break
                        r, p = best
                        w = w[:p] + [m[r]["tok"]] + w[p + 2:]
                    out.extend(w); cur = []
            for s in toks:
                if part_ok(s):
                    if s.startswith(SP) and cur:
                        flush()
                    cur.append(s)
                else:
                    flush(); out.append(s)
            flush()
            real = list(es.tokens)
            tot += 1
            if out != real:
                bad += 1
    print(f"simcheck {a.kind}: {tot} lines simulated ({skipped} further lines skipped because they contain one of the {len(new_letters)} new single-letter tokens), {bad} mismatches between simulation and the real swapped tokenizer")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("classify"); p.add_argument("--tok", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_classify)
    p = sp.add_parser("train"); p.add_argument("--tok", required=True); p.add_argument("--kind", choices=list(KINDS), required=True)
    p.add_argument("--text", nargs="+", required=True, help="text files, one segment per line; each capped at --max-chars or at its own path@CHARS cap"); p.add_argument("--n", type=int, default=3000, help="total new tokens N (geez: letters + merges)")
    p.add_argument("--max-chars", type=int, default=25_000_000, help="per file"); p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--char-min-count", type=int, default=3, help="geez: min number of lines containing a missing Ethiopic letter"); p.add_argument("--out", required=True); p.set_defaults(f=cmd_train)
    p = sp.add_parser("build"); p.add_argument("--tok", required=True)
    p.add_argument("--geez"); p.add_argument("--n-geez", type=int, default=0); p.add_argument("--fon"); p.add_argument("--n-fon", type=int, default=0)
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_build)
    p = sp.add_parser("surgery"); p.add_argument("--model", required=True); p.add_argument("--swap", required=True, help="dir written by `build`")
    p.add_argument("--orig-tok", required=True, help="original tokenizer.json"); p.add_argument("--orig-tok-dir", help="dir of original tokenizer files (for --control)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16"); p.add_argument("--control", action="store_true", help="same weights, ORIGINAL tokenizer files")
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_surgery)
    p = sp.add_parser("spm"); p.add_argument("--spm", required=True, help="original tokenizer.model"); p.add_argument("--swap", required=True, help="dir written by `build` (tokenizer.model is written into it)"); p.set_defaults(f=cmd_spm)
    p = sp.add_parser("simcheck"); p.add_argument("--orig", required=True); p.add_argument("--swapped", required=True); p.add_argument("--merges", required=True)
    p.add_argument("--kind", choices=list(KINDS), required=True); p.add_argument("--n", type=int, required=True); p.add_argument("--text", required=True)
    p.add_argument("--max-chars", type=int, default=2_000_000)
    p.add_argument("--n-is-total", action="store_true", help="--n counts new letters + merges (as in `build`); default: --n = number of merges")
    p.set_defaults(f=cmd_simcheck)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
