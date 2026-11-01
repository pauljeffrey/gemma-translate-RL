# SPDX-License-Identifier: AGPL-3.0-or-later
"""CTranslate2 wrapper for facebook/nllb-200-distilled-600M (R14). Sentence level, English source."""
import time
from typing import List, Optional, Sequence

CONFIGS = {
    "beam4": dict(beam_size=4),
    "greedy": dict(beam_size=1),
    "beam4nr4": dict(beam_size=4, no_repeat_ngram_size=4),           # beam 4 + 4-gram repeat guard: the 'teacher' decoding
}


class NLLB:
    def __init__(self, ct2_dir: str, hf_dir: str, device: str = "cuda", compute_type: str = "float16", src_lang: str = "eng_Latn", intra_threads: int = 4):
        import ctranslate2
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(hf_dir, src_lang=src_lang)
        self.tr = ctranslate2.Translator(ct2_dir, device=device, compute_type=compute_type, intra_threads=intra_threads)

    def encode(self, texts: Sequence[str], max_len: int = 256) -> List[List[str]]:
        out = []
        for t in texts:
            ids = self.tok.encode(t, truncation=True, max_length=max_len)
            out.append(self.tok.convert_ids_to_tokens(ids))
        return out

    def translate(self, texts: Sequence[str], tgt: str, config: str = "beam4", max_batch_tokens: int = 4096, max_decoding_length: int = 256,
                  return_scores: bool = False):
        src = self.encode(texts)
        kw = dict(CONFIGS[config])
        res = self.tr.translate_batch(src, target_prefix=[[tgt]] * len(src), max_batch_size=max_batch_tokens, batch_type="tokens",
                                      max_decoding_length=max_decoding_length, return_scores=return_scores, **kw)
        hyps, scores = [], []
        for r in res:
            toks = r.hypotheses[0]
            toks = toks[1:] if toks and toks[0] == tgt else toks
            hyps.append(self.tok.decode(self.tok.convert_tokens_to_ids(toks), skip_special_tokens=True).strip())
            scores.append(r.scores[0] if return_scores and r.scores else None)
        return (hyps, scores) if return_scores else hyps
