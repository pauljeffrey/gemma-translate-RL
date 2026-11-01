# SPDX-License-Identifier: AGPL-3.0-or-later
"""The TOTAL RL REWARD of a system's document translations, computed with the real reward code (D-44 addendum).

    python scripts/longdoc_score_reward.py --preds preds.jsonl --docs docs.jsonl --config /workspace/sft_build/gemma_mt_2c/configs/rl_rtxpro_phase2c.yaml \\
        --calibration /workspace/sft_build/gemma_mt_2c/calibration/shaping_flores_dev_v6.json --out reward_scores.jsonl

For every (document, language, system) output the text is turned back into what the policy produced - `"{iso} {translation}"` + the end-of-turn id (or no
stop id when the generation did not finish) - and scored by `gemma_mt.rl.reward.TranslationReward` with the reward block of the RL yaml: AfriCOMET-QE per
chunk (sentence chunking with the language-aware omission detector, D-30), the calibrated shaping curve of the language, its content mode (qe / blend /
consensus), failure classes (no_end_of_turn, copy, degenerate, length ratio, wrong language ...), length / lexical / diacritic penalties. Nothing is re-implemented:
the numbers are the numbers the trainer would have given the same completion.

Consensus / blend languages need a GROUP of completions of the same prompt. Two modes: (default) every output alone, exactly like the greedy evaluation of the
trainer (blend -> QE only, consensus languages -> the rubric-only value `consensus_skip_value`); `--group-systems`: the outputs of the several SYSTEMS for the same
(document, language) form the group, so consensus is the mean chrF++ against the other systems' outputs (needs >= `consensus_min_valid` systems).

Library (importable without torch / transformers): `TokensAdapter`, `completion_ids`, `build_reward`, `score_rows`, `summarise`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from gemma_mt.languages import canonical_iso


class TokensAdapter:
    """The few methods of a Hugging Face tokenizer that the reward uses, on top of a `tokenizers.Tokenizer` (tokenizer.json): no transformers, no torch.
    `encode(text, add_special_tokens=False)`, `decode(ids, skip_special_tokens)`, `pad_token_id`, `token_id(str)`."""

    def __init__(self, tokenizer, pad_token_id: int = 0):
        self.tk = tokenizer
        self.pad_token_id = pad_token_id

    @classmethod
    def from_file(cls, path: str) -> "TokensAdapter":
        import glob

        from tokenizers import Tokenizer

        hits = sorted(glob.glob(path))
        t = Tokenizer.from_file(hits[0] if hits else path)
        return cls(t, t.token_to_id("<pad>") or 0)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(self.tk.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_id(self, token: str) -> Optional[int]:
        return self.tk.token_to_id(token)


def stop_ids_of(tok: Any) -> List[int]:
    """[<end_of_turn>, <eos>] of a Gemma tokenizer (adapter or the offline fake)."""
    ids = []
    for s in ("<end_of_turn>", "<eos>"):
        i = tok.token_id(s) if hasattr(tok, "token_id") else tok.convert_tokens_to_ids(s)
        if i is not None and i >= 0:
            ids.append(int(i))
    return ids


def completion_ids(tok: Any, iso: str, translation: str, finished: Optional[bool], stop_id: int, assume_finished: bool = True) -> List[int]:
    """The completion the policy would have sampled for `translation`: tag + space + text, then the end-of-turn id when the generation finished.
    `finished` None = unknown: `assume_finished` decides."""
    tag = canonical_iso(iso) or iso
    ids = list(tok.encode(f"{tag} {translation.strip()}", add_special_tokens=False))
    fin = assume_finished if finished is None else bool(finished)
    return ids + [int(stop_id)] if fin else ids


def file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_reward(config_path: Optional[str], *, calibration: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None, scorer: Any = None,
                 scorer_kind: str = "config", tokenizer: Any = None, tokenizer_json: Optional[str] = None, lid: Any = None,
                 python: Optional[str] = None, pythonpath: Optional[str] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """(TranslationReward, tokenizer, meta). `config_path`: an RL yaml (its `reward:` and `scorer:` blocks are used; None = RewardConfig defaults). `calibration`
    replaces `reward.calibration_path`. `scorer_kind`: config (the yaml's scorer block) | worker (AfriCOMET-QE in a child process) | fake (FakeScorer). `overrides`:
    extra RewardConfig fields. The tokenizer is a `tokenizer.json` (GEMMA tokenizer of the policy) unless one is passed."""
    from gemma_mt.rl.reward import RewardConfig, TranslationReward
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    cfg_d = load_yaml(config_path) if config_path else {}
    rd = dict(cfg_d.get("reward") or {})
    if calibration:
        rd["calibration_path"] = calibration
    rd.update(overrides or {})
    rcfg = dataclass_from_dict(RewardConfig, rd)
    if tokenizer is None:
        tokenizer = TokensAdapter.from_file(tokenizer_json or "/workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json")
    sp = dict(cfg_d.get("scorer") or {})
    if scorer is None:
        kind = scorer_kind if scorer_kind != "config" else sp.get("type", "worker")
        if kind == "fake":
            from gemma_mt.rl.comet import FakeScorer

            scorer = FakeScorer()
        elif kind == "worker":
            from gemma_mt.rl.comet import WorkerScorer

            scorer = WorkerScorer(python=python or sp.get("python", sys.executable), model_id=sp.get("model_id") or "masakhane/africomet-qe-stl-1.1",
                                  batch_size=sp.get("batch_size", 32), pythonpath=pythonpath or sp.get("pythonpath"))
        else:
            from gemma_mt.rl.comet import CometScorer

            scorer = CometScorer(sp.get("model_id", "masakhane/africomet-qe-stl-1.1"), batch_size=sp.get("batch_size", 32))
    stops = stop_ids_of(tokenizer)
    reward = TranslationReward(rcfg, scorer, tokenizer, stops, lid=lid, pad_id=getattr(tokenizer, "pad_token_id", None))
    meta = {"config_path": config_path, "config_sha256": file_sha256(config_path), "calibration_path": rcfg.calibration_path, "calibration_sha256": file_sha256(rcfg.calibration_path),
            "reward_range": list(reward.reward_range), "preset": rcfg.preset, "chunking": rcfg.chunking, "omission_detector": rcfg.omission_detector, "consensus": rcfg.consensus,
            "lid_enabled": rcfg.lid_enabled, "scorer": type(scorer).__name__, "stop_ids": stops}
    return reward, tokenizer, meta


def _row_key(r: Dict[str, Any]) -> Tuple[str, str]:
    return canonical_iso(r["iso"]) or r["iso"], r["doc_id"]


def score_rows(reward: Any, tok: Any, rows: Sequence[Dict[str, Any]], sources: Dict[str, str], *, group_systems: bool = False, batch_size: int = 64,
               assume_finished: bool = True, log: Callable[[str], None] = lambda m: None) -> List[Dict[str, Any]]:
    """rows: [{doc_id, iso, system, hyp (the translation), finished?}]; sources: doc_id -> English text. Returns one dict per input row (same order) with
    reward, status, qe, content_mode, n_chunks, failed_chunks, weight, lid_label, consensus_chrf, diacritic_deficit.
    Rows are scored per language in batches of `batch_size` documents (a batch = `batch_size` x n_systems completions in group mode)."""
    from gemma_mt.rl.reward import Sample

    stop_id = stop_ids_of(tok)[0]
    out: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    by_lang: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_lang[canonical_iso(r["iso"]) or r["iso"]].append(i)
    for iso, idx in sorted(by_lang.items()):
        if group_systems:
            groups: Dict[str, List[int]] = defaultdict(list)
            for i in idx:
                groups[rows[i]["doc_id"]].append(i)
            sizes = {len(v) for v in groups.values()}
            if len(sizes) != 1:
                raise ValueError(f"{iso}: --group-systems needs every document translated by the same set of systems (group sizes {sorted(sizes)})")
            g = sizes.pop()
            units = [sorted(v, key=lambda i: rows[i].get("system", "")) for v in groups.values()]
            per_batch = max(1, batch_size)
            batches = [[i for u in units[b: b + per_batch] for i in u] for b in range(0, len(units), per_batch)]
        else:
            g = None
            batches = [idx[b: b + batch_size] for b in range(0, len(idx), batch_size)]
        for batch in batches:
            samples = [Sample(completion_ids(tok, iso, rows[i]["hyp"], rows[i].get("finished"), stop_id, assume_finished), iso, sources[rows[i]["doc_id"]]) for i in batch]
            res = reward(samples, group_size=g)
            for k, i in enumerate(batch):
                out[i] = {"doc_id": rows[i]["doc_id"], "iso": iso, "system": rows[i].get("system", ""), "reward": float(res.rewards[k]), "status": res.status[k],
                          "qe": res.qe[k], "content_mode": res.content_mode[k], "n_chunks": res.n_chunks[k], "failed_chunks": res.failed_chunks[k], "weight": float(res.weights[k]),
                          "lid_label": res.lid_label[k], "consensus_chrf": res.consensus_chrf[k], "diacritic_deficit": res.diacritic_deficit[k]}
        log(f"[reward] {iso}: {len(idx)} outputs scored")
    return [o for o in out]  # type: ignore[misc]


def summarise(scores: Sequence[Dict[str, Any]], reward_range: Sequence[float]) -> Dict[str, Any]:
    """Per (system, language) and per system: mean reward, the share of failure statuses, mean QE of the scored chunks."""
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for s in scores:
        cells[(s["system"], s["iso"])].append(s)
    res: Dict[str, Any] = {"reward_range": list(reward_range), "systems": {}}
    for (system, iso), ss in sorted(cells.items()):
        st: Dict[str, int] = defaultdict(int)
        for s in ss:
            st[s["status"]] += 1
        qes = [s["qe"] for s in ss if s["qe"] is not None]
        res["systems"].setdefault(system, {"languages": {}})["languages"][iso] = {
            "n": len(ss), "reward_mean": sum(s["reward"] for s in ss) / len(ss), "ok_rate": st.get("ok", 0) / len(ss), "status": dict(st),
            "qe_mean": (sum(qes) / len(qes)) if qes else None, "content_modes": sorted({s["content_mode"] for s in ss})}
    for system, d in res["systems"].items():
        ls = d["languages"].values()
        d["macro_reward_mean"] = sum(v["reward_mean"] for v in ls) / max(len(ls), 1)
        d["documents"] = sum(v["n"] for v in ls)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="jsonl rows {doc_id, iso, system, hyp (or text), finished?}")
    ap.add_argument("--docs", required=True, help="docs.jsonl (doc_id, text = the English source); several files allowed after a comma")
    ap.add_argument("--config", default=None, help="RL yaml whose reward: and scorer: blocks are used")
    ap.add_argument("--calibration", default=None, help="shaping calibration json (replaces reward.calibration_path)")
    ap.add_argument("--reward-override", nargs="*", default=[], help="key=value overrides of RewardConfig fields")
    ap.add_argument("--scorer", default="config", choices=["config", "worker", "fake"])
    ap.add_argument("--python", default=None)
    ap.add_argument("--pythonpath", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--group-systems", action="store_true", help="outputs of the different systems for one (document, language) form the consensus group")
    ap.add_argument("--batch-docs", type=int, default=64)
    ap.add_argument("--assume-unfinished", action="store_true", help="rows without `finished` count as not finished (no_end_of_turn)")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--out", required=True, help="per-output scores (jsonl); a summary json is written next to it")
    a = ap.parse_args(argv)
    from gemma_mt.eval import longdoc as L
    from gemma_mt.utils import parse_overrides

    sources: Dict[str, str] = {}
    for p in a.docs.split(","):
        for d in L.read_jsonl(p):
            sources[d["doc_id"]] = d["text"]
    rows = []
    for r in L.read_jsonl(a.preds):
        if a.langs and (canonical_iso(r["iso"]) not in a.langs):
            continue
        if r["doc_id"] in sources:
            rows.append({"doc_id": r["doc_id"], "iso": r["iso"], "system": r.get("system", ""), "hyp": r.get("hyp", r.get("text", "")), "finished": r.get("finished")})
    fake_tok = None
    if a.tokenizer == "fake":                                     # offline tests: the byte-level stand-in of the Gemma tokenizer
        from gemma_mt.fake_tokenizer import FakeGemmaTokenizer

        fake_tok = FakeGemmaTokenizer()
    reward, tok, meta = build_reward(a.config, calibration=a.calibration, overrides=parse_overrides(a.reward_override), scorer_kind=a.scorer, tokenizer=fake_tok,
                                     tokenizer_json=None if fake_tok else a.tokenizer, python=a.python, pythonpath=a.pythonpath)
    scores = score_rows(reward, tok, rows, sources, group_systems=a.group_systems, batch_size=a.batch_docs, assume_finished=not a.assume_unfinished, log=lambda m: print(m, flush=True))
    L.write_jsonl(a.out, scores)
    summ = summarise(scores, reward.reward_range)
    summ["meta"] = {**meta, "group_systems": a.group_systems, "assume_finished": not a.assume_unfinished}
    with open(os.path.splitext(a.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps({s: {"macro_reward_mean": d["macro_reward_mean"], "documents": d["documents"]} for s, d in summ["systems"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
