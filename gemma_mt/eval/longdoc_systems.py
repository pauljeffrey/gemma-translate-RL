# SPDX-License-Identifier: AGPL-3.0-or-later
"""Candidate open-weight English -> African-language translators for the long-document benchmark.

Pure-Python, stdlib-only data registry (no network access at import time).  Every entry was read
from the public model card / paper / config on 2026-10-20 (see reports/longdoc_benchmark_plan_2026-10-20_parts/
system_cards.json for the evidence: card URL, licence string as written, language-list source, pinned revision).

Terminology used throughout
    codes        our iso -> the system's own code, for languages the system EXPLICITLY names in its
                 language list (model card / paper / tokenizer).  Naming a language is NOT a quality claim.
    proxy_codes  our iso -> the system's code of a CLOSE variety (documented in ``notes``).  NOT counted
                 as support; only a hint for human review.
    prompt_style nllb      tokenizer src_lang + forced_bos_token_id (codes are FLORES-200 codes)
                 madlad    "<2xx> " prefix on the input text
                 chat      chat template carrying ``template`` as the user message
                 t5prompt  plain prompt ``template`` (also used for Alpaca-style prompts)
                 seamless  SeamlessM4Tv2ForTextToText with tgt_lang=<code>
                 marian    hf_id_template.format(code=...) = one model per language; or, when
                           hf_id_template == "", a multi-target Marian model hf_id whose ``template``
                           (">>{tgt_code}<< {text}") carries the target label
    template placeholders: {src_lang} (English name of the source, "English"), {tgt_lang} (English name of
                 the target, e.g. LANGUAGES[iso].name), {tgt_code} (the value from ``codes``), {text}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SystemSpec:
    name: str            # short key, e.g. "madlad400-7b-mt"
    hf_id: str           # Hub repo id ("" if none / per-pair Marian)
    revision: str        # pinned commit sha seen on the Hub ("" if unknown / per-pair)
    family: str          # nllb | madlad | gemma | aya | seamless | mt0 | marian | qwen | other
    prompt_style: str    # nllb | madlad | chat | t5prompt | seamless | marian
    template: str        # user message / prompt ("" when the style needs none)
    hf_id_template: str  # per-language Marian repo template ("" otherwise)
    backends: Tuple[str, ...]   # subset of ("hf", "vllm", "ctranslate2") known to work for the architecture
    params_b: float
    vram_gb_fp16: float
    vram_gb_int8: float
    max_input_tokens: int
    weights_licence: str
    licence_class: str   # permissive | gemma_terms | llama | noncommercial | other
    licence_url: str
    output_note: str
    card_url: str
    checked: str
    src_code: str        # the system's code for English
    codes: Dict[str, str]
    proxy_codes: Dict[str, str]
    notes: str


PROMPT_STYLES = ("nllb", "madlad", "chat", "t5prompt", "seamless", "marian")
FAMILIES = ("nllb", "madlad", "gemma", "aya", "seamless", "mt0", "marian", "qwen", "other")
LICENCE_CLASSES = ("permissive", "gemma_terms", "llama", "noncommercial", "other")
BACKENDS = ("hf", "vllm", "ctranslate2")

# The 77 target languages (identical to gemma_mt.languages.SUPPORTED_RL; a unit test checks this).
ISO77: Tuple[str, ...] = (
    'yor', 'hau', 'ibo', 'pcm', 'efi', 'twi', 'ewe', 'fon', 'wol', 'swh',
    'kin', 'som', 'orm', 'amh', 'tir', 'lug', 'lin', 'zul', 'xho', 'afr',
    'sna', 'nya', 'sot', 'tsn', 'mlg', 'ibb', 'urh', 'bem', 'tum', 'run',
    'ven', 'tso', 'nso', 'ssw', 'nbl', 'sag', 'tiv', 'umb', 'kmb', 'mos',
    'kik', 'luo', 'kam', 'bin', 'dag', 'oke', 'shi', 'dyu', 'bam', 'kbp',
    'bbj', 'kon', 'kea', 'kab', 'nus', 'swc', 'lua', 'kqn', 'lun', 'iso',
    'loz', 'wal', 'ach', 'tsc', 'vmw', 'kua', 'seh', 'nyn', 'her', 'yao',
    'sid', 'ndc', 'kri', 'ish', 'dua', 'bas', 'wes',
)


# FLORES-200 code of each of our languages that FLORES-200 contains (official list, 204 codes:
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md, retrieved 2026-10-20).
# Differs from LANGUAGES[iso].flores (FLORES+ column) for: twi (FLORES+ 'twi_Latn_akua1239'), kon (column empty there);
# vmw has a FLORES+ code (vmw_Latn) but is NOT in FLORES-200.
FLORES200_CODE: Dict[str, str] = {
    'yor': 'yor_Latn', 'hau': 'hau_Latn', 'ibo': 'ibo_Latn', 'twi': 'twi_Latn', 'ewe': 'ewe_Latn',
    'fon': 'fon_Latn', 'wol': 'wol_Latn', 'swh': 'swh_Latn', 'kin': 'kin_Latn', 'som': 'som_Latn',
    'orm': 'gaz_Latn', 'amh': 'amh_Ethi', 'tir': 'tir_Ethi', 'lug': 'lug_Latn', 'lin': 'lin_Latn',
    'zul': 'zul_Latn', 'xho': 'xho_Latn', 'afr': 'afr_Latn', 'sna': 'sna_Latn', 'nya': 'nya_Latn',
    'sot': 'sot_Latn', 'tsn': 'tsn_Latn', 'mlg': 'plt_Latn', 'bem': 'bem_Latn', 'tum': 'tum_Latn',
    'run': 'run_Latn', 'tso': 'tso_Latn', 'nso': 'nso_Latn', 'ssw': 'ssw_Latn', 'sag': 'sag_Latn',
    'umb': 'umb_Latn', 'kmb': 'kmb_Latn', 'mos': 'mos_Latn', 'kik': 'kik_Latn', 'luo': 'luo_Latn',
    'kam': 'kam_Latn', 'dyu': 'dyu_Latn', 'bam': 'bam_Latn', 'kbp': 'kbp_Latn', 'kon': 'kon_Latn',
    'kea': 'kea_Latn', 'kab': 'kab_Latn', 'nus': 'nus_Latn', 'lua': 'lua_Latn',
}


SYSTEMS: Dict[str, SystemSpec] = {
    'nllb-200-distilled-600m': SystemSpec(
        name='nllb-200-distilled-600m',
        hf_id='facebook/nllb-200-distilled-600M',
        revision='f8d333a098d19b4fd9a8b18f94170487ad3f821d',
        family='nllb',
        prompt_style='nllb',
        template='',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=0.615,
        vram_gb_fp16=1.2,
        vram_gb_int8=0.6,
        max_input_tokens=512,
        weights_licence='CC-BY-NC-4.0',
        licence_class='noncommercial',
        licence_url='https://creativecommons.org/licenses/by-nc/4.0/legalcode',
        output_note=(
            'Weights are CC-BY-NC-4.0; the card calls NLLB-200 a research model not released for '
            'production deployment, not intended for document translation, trained on inputs of at '
            'most 512 tokens. The licence text does not mention generated translations; whether the '
            'NC condition reaches model outputs is untested, so releasing NLLB outputs as '
            'pseudo-references is a legal risk to clear first.'
        ),
        card_url='https://huggingface.co/facebook/nllb-200-distilled-600M',
        checked='2026-10-20',
        src_code='eng_Latn',
        codes={
            'yor': 'yor_Latn', 'hau': 'hau_Latn', 'ibo': 'ibo_Latn', 'twi': 'twi_Latn', 'ewe': 'ewe_Latn',
            'fon': 'fon_Latn', 'wol': 'wol_Latn', 'swh': 'swh_Latn', 'kin': 'kin_Latn', 'som': 'som_Latn',
            'orm': 'gaz_Latn', 'amh': 'amh_Ethi', 'tir': 'tir_Ethi', 'lug': 'lug_Latn', 'lin': 'lin_Latn',
            'zul': 'zul_Latn', 'xho': 'xho_Latn', 'afr': 'afr_Latn', 'sna': 'sna_Latn', 'nya': 'nya_Latn',
            'sot': 'sot_Latn', 'tsn': 'tsn_Latn', 'mlg': 'plt_Latn', 'bem': 'bem_Latn', 'tum': 'tum_Latn',
            'run': 'run_Latn', 'tso': 'tso_Latn', 'nso': 'nso_Latn', 'ssw': 'ssw_Latn', 'sag': 'sag_Latn',
            'umb': 'umb_Latn', 'kmb': 'kmb_Latn', 'mos': 'mos_Latn', 'kik': 'kik_Latn', 'luo': 'luo_Latn',
            'kam': 'kam_Latn', 'dyu': 'dyu_Latn', 'bam': 'bam_Latn', 'kbp': 'kbp_Latn', 'kon': 'kon_Latn',
            'kea': 'kea_Latn', 'kab': 'kab_Latn', 'nus': 'nus_Latn', 'lua': 'lua_Latn',
        },
        proxy_codes={
            'swc': 'swh_Latn', 'nbl': 'zul_Latn', 'loz': 'sot_Latn', 'ndc': 'sna_Latn', 'tsc': 'tso_Latn',
            'shi': 'tzm_Tfng',
        },
        notes=(
            'Codes = FLORES-200 codes = the 202 language tokens in the tokenizer special_tokens_map '
            '(identical in the 600M, distilled-1.3B and 3.3B cards checked; the dense 1.3B card was '
            "assumed identical). Use tokenizer src_lang='eng_Latn' and forced_bos_token_id=tgt code. "
            'Proxies: shi->tzm_Tfng (Central Atlas Tamazight, Tifinagh; Tachelhit is a related but '
            'distinct variety), swc->swh, nbl->zul, loz->sot, ndc->sna, tsc->tso are close-variety '
            "guesses, NOT counted as support. aka_Latn (Akan) also exists; our 'twi' uses twi_Latn. "
            'gaz_Latn = West Central Oromo, plt_Latn = Plateau Malagasy. Published quality numbers '
            'for NLLB-54B-MoE (not these checkpoints) are in the MADLAD-400 paper Table 17. hf: class'
            ' present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20); vllm: '
            'M2M100ForConditionalGeneration is NOT in the vllm 0.31.0 ModelRegistry (venv_vllm import'
            ' check, no GPU); vLLM dropped T5-style encoder-decoder support (PR #11901 closed '
            'unmerged 2025-10-02, per research notes); ctranslate2: M2M100/NLLB converter listed in '
            'CTranslate2 docs/guides/transformers.md (master, retrieved 2026-10-20); package NOT '
            'installed on the rental and not tested (CT2 build for sm_120 Blackwell unconfirmed) VRAM'
            ' = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV '
            'cache at sentence level.'
        ),
    ),
    'nllb-200-distilled-1.3b': SystemSpec(
        name='nllb-200-distilled-1.3b',
        hf_id='facebook/nllb-200-distilled-1.3B',
        revision='7be3e24664b38ce1cac29b8aeed6911aa0cf0576',
        family='nllb',
        prompt_style='nllb',
        template='',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=1.37,
        vram_gb_fp16=2.7,
        vram_gb_int8=1.4,
        max_input_tokens=512,
        weights_licence='CC-BY-NC-4.0',
        licence_class='noncommercial',
        licence_url='https://creativecommons.org/licenses/by-nc/4.0/legalcode',
        output_note=(
            'Weights are CC-BY-NC-4.0; the card calls NLLB-200 a research model not released for '
            'production deployment, not intended for document translation, trained on inputs of at '
            'most 512 tokens. The licence text does not mention generated translations; whether the '
            'NC condition reaches model outputs is untested, so releasing NLLB outputs as '
            'pseudo-references is a legal risk to clear first.'
        ),
        card_url='https://huggingface.co/facebook/nllb-200-distilled-1.3B',
        checked='2026-10-20',
        src_code='eng_Latn',
        codes={
            'yor': 'yor_Latn', 'hau': 'hau_Latn', 'ibo': 'ibo_Latn', 'twi': 'twi_Latn', 'ewe': 'ewe_Latn',
            'fon': 'fon_Latn', 'wol': 'wol_Latn', 'swh': 'swh_Latn', 'kin': 'kin_Latn', 'som': 'som_Latn',
            'orm': 'gaz_Latn', 'amh': 'amh_Ethi', 'tir': 'tir_Ethi', 'lug': 'lug_Latn', 'lin': 'lin_Latn',
            'zul': 'zul_Latn', 'xho': 'xho_Latn', 'afr': 'afr_Latn', 'sna': 'sna_Latn', 'nya': 'nya_Latn',
            'sot': 'sot_Latn', 'tsn': 'tsn_Latn', 'mlg': 'plt_Latn', 'bem': 'bem_Latn', 'tum': 'tum_Latn',
            'run': 'run_Latn', 'tso': 'tso_Latn', 'nso': 'nso_Latn', 'ssw': 'ssw_Latn', 'sag': 'sag_Latn',
            'umb': 'umb_Latn', 'kmb': 'kmb_Latn', 'mos': 'mos_Latn', 'kik': 'kik_Latn', 'luo': 'luo_Latn',
            'kam': 'kam_Latn', 'dyu': 'dyu_Latn', 'bam': 'bam_Latn', 'kbp': 'kbp_Latn', 'kon': 'kon_Latn',
            'kea': 'kea_Latn', 'kab': 'kab_Latn', 'nus': 'nus_Latn', 'lua': 'lua_Latn',
        },
        proxy_codes={
            'swc': 'swh_Latn', 'nbl': 'zul_Latn', 'loz': 'sot_Latn', 'ndc': 'sna_Latn', 'tsc': 'tso_Latn',
            'shi': 'tzm_Tfng',
        },
        notes=(
            'Codes = FLORES-200 codes = the 202 language tokens in the tokenizer special_tokens_map '
            '(identical in the 600M, distilled-1.3B and 3.3B cards checked; the dense 1.3B card was '
            "assumed identical). Use tokenizer src_lang='eng_Latn' and forced_bos_token_id=tgt code. "
            'Proxies: shi->tzm_Tfng (Central Atlas Tamazight, Tifinagh; Tachelhit is a related but '
            'distinct variety), swc->swh, nbl->zul, loz->sot, ndc->sna, tsc->tso are close-variety '
            "guesses, NOT counted as support. aka_Latn (Akan) also exists; our 'twi' uses twi_Latn. "
            'gaz_Latn = West Central Oromo, plt_Latn = Plateau Malagasy. Published quality numbers '
            'for NLLB-54B-MoE (not these checkpoints) are in the MADLAD-400 paper Table 17. hf: class'
            ' present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20); vllm: '
            'M2M100ForConditionalGeneration is NOT in the vllm 0.31.0 ModelRegistry (venv_vllm import'
            ' check, no GPU); vLLM dropped T5-style encoder-decoder support (PR #11901 closed '
            'unmerged 2025-10-02, per research notes); ctranslate2: M2M100/NLLB converter listed in '
            'CTranslate2 docs/guides/transformers.md (master, retrieved 2026-10-20); package NOT '
            'installed on the rental and not tested (CT2 build for sm_120 Blackwell unconfirmed) VRAM'
            ' = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV '
            'cache at sentence level.'
        ),
    ),
    'nllb-200-1.3b': SystemSpec(
        name='nllb-200-1.3b',
        hf_id='facebook/nllb-200-1.3B',
        revision='b0de46b488af0cf31749cd8da5ed3171e11b2309',
        family='nllb',
        prompt_style='nllb',
        template='',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=1.37,
        vram_gb_fp16=2.7,
        vram_gb_int8=1.4,
        max_input_tokens=512,
        weights_licence='CC-BY-NC-4.0',
        licence_class='noncommercial',
        licence_url='https://creativecommons.org/licenses/by-nc/4.0/legalcode',
        output_note=(
            'Weights are CC-BY-NC-4.0; the card calls NLLB-200 a research model not released for '
            'production deployment, not intended for document translation, trained on inputs of at '
            'most 512 tokens. The licence text does not mention generated translations; whether the '
            'NC condition reaches model outputs is untested, so releasing NLLB outputs as '
            'pseudo-references is a legal risk to clear first.'
        ),
        card_url='https://huggingface.co/facebook/nllb-200-1.3B',
        checked='2026-10-20',
        src_code='eng_Latn',
        codes={
            'yor': 'yor_Latn', 'hau': 'hau_Latn', 'ibo': 'ibo_Latn', 'twi': 'twi_Latn', 'ewe': 'ewe_Latn',
            'fon': 'fon_Latn', 'wol': 'wol_Latn', 'swh': 'swh_Latn', 'kin': 'kin_Latn', 'som': 'som_Latn',
            'orm': 'gaz_Latn', 'amh': 'amh_Ethi', 'tir': 'tir_Ethi', 'lug': 'lug_Latn', 'lin': 'lin_Latn',
            'zul': 'zul_Latn', 'xho': 'xho_Latn', 'afr': 'afr_Latn', 'sna': 'sna_Latn', 'nya': 'nya_Latn',
            'sot': 'sot_Latn', 'tsn': 'tsn_Latn', 'mlg': 'plt_Latn', 'bem': 'bem_Latn', 'tum': 'tum_Latn',
            'run': 'run_Latn', 'tso': 'tso_Latn', 'nso': 'nso_Latn', 'ssw': 'ssw_Latn', 'sag': 'sag_Latn',
            'umb': 'umb_Latn', 'kmb': 'kmb_Latn', 'mos': 'mos_Latn', 'kik': 'kik_Latn', 'luo': 'luo_Latn',
            'kam': 'kam_Latn', 'dyu': 'dyu_Latn', 'bam': 'bam_Latn', 'kbp': 'kbp_Latn', 'kon': 'kon_Latn',
            'kea': 'kea_Latn', 'kab': 'kab_Latn', 'nus': 'nus_Latn', 'lua': 'lua_Latn',
        },
        proxy_codes={
            'swc': 'swh_Latn', 'nbl': 'zul_Latn', 'loz': 'sot_Latn', 'ndc': 'sna_Latn', 'tsc': 'tso_Latn',
            'shi': 'tzm_Tfng',
        },
        notes=(
            'Codes = FLORES-200 codes = the 202 language tokens in the tokenizer special_tokens_map '
            '(identical in the 600M, distilled-1.3B and 3.3B cards checked; the dense 1.3B card was '
            "assumed identical). Use tokenizer src_lang='eng_Latn' and forced_bos_token_id=tgt code. "
            'Proxies: shi->tzm_Tfng (Central Atlas Tamazight, Tifinagh; Tachelhit is a related but '
            'distinct variety), swc->swh, nbl->zul, loz->sot, ndc->sna, tsc->tso are close-variety '
            "guesses, NOT counted as support. aka_Latn (Akan) also exists; our 'twi' uses twi_Latn. "
            'gaz_Latn = West Central Oromo, plt_Latn = Plateau Malagasy. Published quality numbers '
            'for NLLB-54B-MoE (not these checkpoints) are in the MADLAD-400 paper Table 17. hf: class'
            ' present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20); vllm: '
            'M2M100ForConditionalGeneration is NOT in the vllm 0.31.0 ModelRegistry (venv_vllm import'
            ' check, no GPU); vLLM dropped T5-style encoder-decoder support (PR #11901 closed '
            'unmerged 2025-10-02, per research notes); ctranslate2: M2M100/NLLB converter listed in '
            'CTranslate2 docs/guides/transformers.md (master, retrieved 2026-10-20); package NOT '
            'installed on the rental and not tested (CT2 build for sm_120 Blackwell unconfirmed) VRAM'
            ' = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV '
            'cache at sentence level.'
        ),
    ),
    'nllb-200-3.3b': SystemSpec(
        name='nllb-200-3.3b',
        hf_id='facebook/nllb-200-3.3B',
        revision='1a07f7d195896b2114afcb79b7b57ab512e7b43e',
        family='nllb',
        prompt_style='nllb',
        template='',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=3.34,
        vram_gb_fp16=6.7,
        vram_gb_int8=3.3,
        max_input_tokens=512,
        weights_licence='CC-BY-NC-4.0',
        licence_class='noncommercial',
        licence_url='https://creativecommons.org/licenses/by-nc/4.0/legalcode',
        output_note=(
            'Weights are CC-BY-NC-4.0; the card calls NLLB-200 a research model not released for '
            'production deployment, not intended for document translation, trained on inputs of at '
            'most 512 tokens. The licence text does not mention generated translations; whether the '
            'NC condition reaches model outputs is untested, so releasing NLLB outputs as '
            'pseudo-references is a legal risk to clear first.'
        ),
        card_url='https://huggingface.co/facebook/nllb-200-3.3B',
        checked='2026-10-20',
        src_code='eng_Latn',
        codes={
            'yor': 'yor_Latn', 'hau': 'hau_Latn', 'ibo': 'ibo_Latn', 'twi': 'twi_Latn', 'ewe': 'ewe_Latn',
            'fon': 'fon_Latn', 'wol': 'wol_Latn', 'swh': 'swh_Latn', 'kin': 'kin_Latn', 'som': 'som_Latn',
            'orm': 'gaz_Latn', 'amh': 'amh_Ethi', 'tir': 'tir_Ethi', 'lug': 'lug_Latn', 'lin': 'lin_Latn',
            'zul': 'zul_Latn', 'xho': 'xho_Latn', 'afr': 'afr_Latn', 'sna': 'sna_Latn', 'nya': 'nya_Latn',
            'sot': 'sot_Latn', 'tsn': 'tsn_Latn', 'mlg': 'plt_Latn', 'bem': 'bem_Latn', 'tum': 'tum_Latn',
            'run': 'run_Latn', 'tso': 'tso_Latn', 'nso': 'nso_Latn', 'ssw': 'ssw_Latn', 'sag': 'sag_Latn',
            'umb': 'umb_Latn', 'kmb': 'kmb_Latn', 'mos': 'mos_Latn', 'kik': 'kik_Latn', 'luo': 'luo_Latn',
            'kam': 'kam_Latn', 'dyu': 'dyu_Latn', 'bam': 'bam_Latn', 'kbp': 'kbp_Latn', 'kon': 'kon_Latn',
            'kea': 'kea_Latn', 'kab': 'kab_Latn', 'nus': 'nus_Latn', 'lua': 'lua_Latn',
        },
        proxy_codes={
            'swc': 'swh_Latn', 'nbl': 'zul_Latn', 'loz': 'sot_Latn', 'ndc': 'sna_Latn', 'tsc': 'tso_Latn',
            'shi': 'tzm_Tfng',
        },
        notes=(
            'Codes = FLORES-200 codes = the 202 language tokens in the tokenizer special_tokens_map '
            '(identical in the 600M, distilled-1.3B and 3.3B cards checked; the dense 1.3B card was '
            "assumed identical). Use tokenizer src_lang='eng_Latn' and forced_bos_token_id=tgt code. "
            'Proxies: shi->tzm_Tfng (Central Atlas Tamazight, Tifinagh; Tachelhit is a related but '
            'distinct variety), swc->swh, nbl->zul, loz->sot, ndc->sna, tsc->tso are close-variety '
            "guesses, NOT counted as support. aka_Latn (Akan) also exists; our 'twi' uses twi_Latn. "
            'gaz_Latn = West Central Oromo, plt_Latn = Plateau Malagasy. Published quality numbers '
            'for NLLB-54B-MoE (not these checkpoints) are in the MADLAD-400 paper Table 17. hf: class'
            ' present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20); vllm: '
            'M2M100ForConditionalGeneration is NOT in the vllm 0.31.0 ModelRegistry (venv_vllm import'
            ' check, no GPU); vLLM dropped T5-style encoder-decoder support (PR #11901 closed '
            'unmerged 2025-10-02, per research notes); ctranslate2: M2M100/NLLB converter listed in '
            'CTranslate2 docs/guides/transformers.md (master, retrieved 2026-10-20); package NOT '
            'installed on the rental and not tested (CT2 build for sm_120 Blackwell unconfirmed) VRAM'
            ' = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV '
            'cache at sentence level.'
        ),
    ),
    'madlad400-3b-mt': SystemSpec(
        name='madlad400-3b-mt',
        hf_id='google/madlad400-3b-mt',
        revision='fa184c675da0b5c9e1c8694fccd4e12e2d422094',
        family='madlad',
        prompt_style='madlad',
        template='',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=2.94,
        vram_gb_fp16=5.9,
        vram_gb_int8=2.9,
        max_input_tokens=256,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            'Apache-2.0: the licence places no restriction on generated text; the card names the '
            'research community as primary users and notes that only 204 of the supported languages '
            'were evaluated.'
        ),
        card_url='https://huggingface.co/google/madlad400-3b-mt',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'twi': 'ak', 'ewe': 'ee', 'fon': 'fon', 'wol': 'wo',
            'swh': 'sw', 'kin': 'rw', 'som': 'so', 'orm': 'om', 'amh': 'am', 'tir': 'ti', 'lug': 'lg',
            'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'afr': 'af', 'sna': 'sn', 'nya': 'ny', 'sot': 'st',
            'tsn': 'tn', 'mlg': 'mg', 'ibb': 'ibb', 'run': 'rn', 'ven': 've', 'tso': 'ts', 'nso': 'nso',
            'ssw': 'ss', 'nbl': 'nr', 'sag': 'sg', 'tiv': 'tiv', 'kmb': 'kmb', 'dyu': 'dyu', 'bam': 'bm',
            'kbp': 'kbp', 'kon': 'kg', 'iso': 'iso', 'wal': 'wal', 'tsc': 'tsc', 'kua': 'kj', 'seh': 'seh',
            'kri': 'kri', 'bas': 'bas',
        },
        proxy_codes={
            'swc': 'sw', 'loz': 'st', 'ndc': 'sn', 'efi': 'ibb', 'shi': 'ber', 'kab': 'ber_Latn',
            'lua': 'lu',
        },
        notes=(
            "Prompt = '<2xx> ' + English source (xx from the codes column, e.g. '<2sw> Hello'). "
            'Explicit = language is in BOTH the model-card YAML language list (419 codes) and the '
            "sentencepiece vocab '<2xx>' tokens (492 pieces, read from spiece.model). Edge cases NOT "
            "counted: 'ach' is in the README list but has no <2ach> vocab token; 'ndc_ZW' and "
            "'ber_Latn' have vocab tokens but are not in the README list; 'pcm' and 'nyn' appear in "
            "the paper's language table without data. 'ak' is named 'Twi' in the paper table (Akuapem"
            " vs Asante not specified). 'kj' = Kuanyama, 'om' = Oromo (macro), 'mg' = Malagasy "
            "(macro). 'lu' = Luba-Katanga is NOT Luba-Kasai (lua) -> proxy only; 'ber' = Tamazight "
            "(Tfng), 'ber_Latn' = Tamazight (Latn). Many FLORES languages absent from the vocab (bem,"
            " tum, umb, mos, ki, luo, kam, kea, kab, lua) score ~2 chrF for 3B/10B in the paper's "
            'Table 17 (7.2B 10-18): not supported. Max training sequence length 256 tokens (paper '
            'table: MT models). hf: class present in transformers 5.19.0 (venv_gemma import check, '
            'CPU only, 2026-10-20); vllm: T5ForConditionalGeneration is NOT in the vllm 0.31.0 '
            'ModelRegistry (venv_vllm import check, no GPU); vLLM dropped T5-style encoder-decoder '
            'support (PR #11901 closed unmerged 2025-10-02, per research notes); ctranslate2: T5 '
            'converter listed in CTranslate2 docs/guides/transformers.md (master, retrieved '
            '2026-10-20); package NOT installed on the rental and not tested (CT2 build for sm_120 '
            'Blackwell unconfirmed) (community CT2 conversions exist: santhosh/madlad400-3b-ct2, '
            'Nextcloud-AI/madlad400-3b-mt-ct2-int8, LifeAi-dev/madlad400-10b-ct2-float32) VRAM = '
            'weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache '
            'at sentence level.'
        ),
    ),
    'madlad400-7b-mt': SystemSpec(
        name='madlad400-7b-mt',
        hf_id='google/madlad400-7b-mt',
        revision='1ff63ce7ddd571a69d45cad9672c553428419671',
        family='madlad',
        prompt_style='madlad',
        template='',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=8.3,
        vram_gb_fp16=16.6,
        vram_gb_int8=8.3,
        max_input_tokens=256,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            'Apache-2.0: the licence places no restriction on generated text; the card names the '
            'research community as primary users and notes that only 204 of the supported languages '
            'were evaluated.'
        ),
        card_url='https://huggingface.co/google/madlad400-7b-mt',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'twi': 'ak', 'ewe': 'ee', 'fon': 'fon', 'wol': 'wo',
            'swh': 'sw', 'kin': 'rw', 'som': 'so', 'orm': 'om', 'amh': 'am', 'tir': 'ti', 'lug': 'lg',
            'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'afr': 'af', 'sna': 'sn', 'nya': 'ny', 'sot': 'st',
            'tsn': 'tn', 'mlg': 'mg', 'ibb': 'ibb', 'run': 'rn', 'ven': 've', 'tso': 'ts', 'nso': 'nso',
            'ssw': 'ss', 'nbl': 'nr', 'sag': 'sg', 'tiv': 'tiv', 'kmb': 'kmb', 'dyu': 'dyu', 'bam': 'bm',
            'kbp': 'kbp', 'kon': 'kg', 'iso': 'iso', 'wal': 'wal', 'tsc': 'tsc', 'kua': 'kj', 'seh': 'seh',
            'kri': 'kri', 'bas': 'bas',
        },
        proxy_codes={
            'swc': 'sw', 'loz': 'st', 'ndc': 'sn', 'efi': 'ibb', 'shi': 'ber', 'kab': 'ber_Latn',
            'lua': 'lu',
        },
        notes=(
            "Prompt = '<2xx> ' + English source (xx from the codes column, e.g. '<2sw> Hello'). "
            'Explicit = language is in BOTH the model-card YAML language list (419 codes) and the '
            "sentencepiece vocab '<2xx>' tokens (492 pieces, read from spiece.model). Edge cases NOT "
            "counted: 'ach' is in the README list but has no <2ach> vocab token; 'ndc_ZW' and "
            "'ber_Latn' have vocab tokens but are not in the README list; 'pcm' and 'nyn' appear in "
            "the paper's language table without data. 'ak' is named 'Twi' in the paper table (Akuapem"
            " vs Asante not specified). 'kj' = Kuanyama, 'om' = Oromo (macro), 'mg' = Malagasy "
            "(macro). 'lu' = Luba-Katanga is NOT Luba-Kasai (lua) -> proxy only; 'ber' = Tamazight "
            "(Tfng), 'ber_Latn' = Tamazight (Latn). Many FLORES languages absent from the vocab (bem,"
            " tum, umb, mos, ki, luo, kam, kea, kab, lua) score ~2 chrF for 3B/10B in the paper's "
            'Table 17 (7.2B 10-18): not supported. Max training sequence length 256 tokens (paper '
            'table: MT models). hf: class present in transformers 5.19.0 (venv_gemma import check, '
            'CPU only, 2026-10-20); vllm: T5ForConditionalGeneration is NOT in the vllm 0.31.0 '
            'ModelRegistry (venv_vllm import check, no GPU); vLLM dropped T5-style encoder-decoder '
            'support (PR #11901 closed unmerged 2025-10-02, per research notes); ctranslate2: T5 '
            'converter listed in CTranslate2 docs/guides/transformers.md (master, retrieved '
            '2026-10-20); package NOT installed on the rental and not tested (CT2 build for sm_120 '
            'Blackwell unconfirmed) (community CT2 conversions exist: santhosh/madlad400-3b-ct2, '
            'Nextcloud-AI/madlad400-3b-mt-ct2-int8, LifeAi-dev/madlad400-10b-ct2-float32) VRAM = '
            'weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache '
            'at sentence level.'
        ),
    ),
    'madlad400-7b-mt-bt': SystemSpec(
        name='madlad400-7b-mt-bt',
        hf_id='google/madlad400-7b-mt-bt',
        revision='1259294dcefa2a4d8e0167ad418fdb47cfaeeca3',
        family='madlad',
        prompt_style='madlad',
        template='',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=8.3,
        vram_gb_fp16=16.6,
        vram_gb_int8=8.3,
        max_input_tokens=256,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            'Apache-2.0: the licence places no restriction on generated text; the card names the '
            'research community as primary users and notes that only 204 of the supported languages '
            'were evaluated.'
        ),
        card_url='https://huggingface.co/google/madlad400-7b-mt-bt',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'twi': 'ak', 'ewe': 'ee', 'fon': 'fon', 'wol': 'wo',
            'swh': 'sw', 'kin': 'rw', 'som': 'so', 'orm': 'om', 'amh': 'am', 'tir': 'ti', 'lug': 'lg',
            'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'afr': 'af', 'sna': 'sn', 'nya': 'ny', 'sot': 'st',
            'tsn': 'tn', 'mlg': 'mg', 'ibb': 'ibb', 'run': 'rn', 'ven': 've', 'tso': 'ts', 'nso': 'nso',
            'ssw': 'ss', 'nbl': 'nr', 'sag': 'sg', 'tiv': 'tiv', 'kmb': 'kmb', 'dyu': 'dyu', 'bam': 'bm',
            'kbp': 'kbp', 'kon': 'kg', 'iso': 'iso', 'wal': 'wal', 'tsc': 'tsc', 'kua': 'kj', 'seh': 'seh',
            'kri': 'kri', 'bas': 'bas',
        },
        proxy_codes={
            'swc': 'sw', 'loz': 'st', 'ndc': 'sn', 'efi': 'ibb', 'shi': 'ber', 'kab': 'ber_Latn',
            'lua': 'lu',
        },
        notes=(
            "Prompt = '<2xx> ' + English source (xx from the codes column, e.g. '<2sw> Hello'). "
            'Explicit = language is in BOTH the model-card YAML language list (419 codes) and the '
            "sentencepiece vocab '<2xx>' tokens (492 pieces, read from spiece.model). Edge cases NOT "
            "counted: 'ach' is in the README list but has no <2ach> vocab token; 'ndc_ZW' and "
            "'ber_Latn' have vocab tokens but are not in the README list; 'pcm' and 'nyn' appear in "
            "the paper's language table without data. 'ak' is named 'Twi' in the paper table (Akuapem"
            " vs Asante not specified). 'kj' = Kuanyama, 'om' = Oromo (macro), 'mg' = Malagasy "
            "(macro). 'lu' = Luba-Katanga is NOT Luba-Kasai (lua) -> proxy only; 'ber' = Tamazight "
            "(Tfng), 'ber_Latn' = Tamazight (Latn). Many FLORES languages absent from the vocab (bem,"
            " tum, umb, mos, ki, luo, kam, kea, kab, lua) score ~2 chrF for 3B/10B in the paper's "
            'Table 17 (7.2B 10-18): not supported. Max training sequence length 256 tokens (paper '
            'table: MT models). hf: class present in transformers 5.19.0 (venv_gemma import check, '
            'CPU only, 2026-10-20); vllm: T5ForConditionalGeneration is NOT in the vllm 0.31.0 '
            'ModelRegistry (venv_vllm import check, no GPU); vLLM dropped T5-style encoder-decoder '
            'support (PR #11901 closed unmerged 2025-10-02, per research notes); ctranslate2: T5 '
            'converter listed in CTranslate2 docs/guides/transformers.md (master, retrieved '
            '2026-10-20); package NOT installed on the rental and not tested (CT2 build for sm_120 '
            'Blackwell unconfirmed) (community CT2 conversions exist: santhosh/madlad400-3b-ct2, '
            "Nextcloud-AI/madlad400-3b-mt-ct2-int8, LifeAi-dev/madlad400-10b-ct2-float32) '-bt' = 7B "
            'fine-tuned on back-translated data; the paper reports +3.0 chrF on en->xx FLORES-200 for'
            ' it vs 7B-mt (published, not measured by us). VRAM = weights only (params x 2 B '
            'fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'madlad400-10b-mt': SystemSpec(
        name='madlad400-10b-mt',
        hf_id='google/madlad400-10b-mt',
        revision='9f2797629c31e69617186dbe5f0ca43bf662f36d',
        family='madlad',
        prompt_style='madlad',
        template='',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=10.71,
        vram_gb_fp16=21.4,
        vram_gb_int8=10.7,
        max_input_tokens=256,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            'Apache-2.0: the licence places no restriction on generated text; the card names the '
            'research community as primary users and notes that only 204 of the supported languages '
            'were evaluated.'
        ),
        card_url='https://huggingface.co/google/madlad400-10b-mt',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'twi': 'ak', 'ewe': 'ee', 'fon': 'fon', 'wol': 'wo',
            'swh': 'sw', 'kin': 'rw', 'som': 'so', 'orm': 'om', 'amh': 'am', 'tir': 'ti', 'lug': 'lg',
            'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'afr': 'af', 'sna': 'sn', 'nya': 'ny', 'sot': 'st',
            'tsn': 'tn', 'mlg': 'mg', 'ibb': 'ibb', 'run': 'rn', 'ven': 've', 'tso': 'ts', 'nso': 'nso',
            'ssw': 'ss', 'nbl': 'nr', 'sag': 'sg', 'tiv': 'tiv', 'kmb': 'kmb', 'dyu': 'dyu', 'bam': 'bm',
            'kbp': 'kbp', 'kon': 'kg', 'iso': 'iso', 'wal': 'wal', 'tsc': 'tsc', 'kua': 'kj', 'seh': 'seh',
            'kri': 'kri', 'bas': 'bas',
        },
        proxy_codes={
            'swc': 'sw', 'loz': 'st', 'ndc': 'sn', 'efi': 'ibb', 'shi': 'ber', 'kab': 'ber_Latn',
            'lua': 'lu',
        },
        notes=(
            "Prompt = '<2xx> ' + English source (xx from the codes column, e.g. '<2sw> Hello'). "
            'Explicit = language is in BOTH the model-card YAML language list (419 codes) and the '
            "sentencepiece vocab '<2xx>' tokens (492 pieces, read from spiece.model). Edge cases NOT "
            "counted: 'ach' is in the README list but has no <2ach> vocab token; 'ndc_ZW' and "
            "'ber_Latn' have vocab tokens but are not in the README list; 'pcm' and 'nyn' appear in "
            "the paper's language table without data. 'ak' is named 'Twi' in the paper table (Akuapem"
            " vs Asante not specified). 'kj' = Kuanyama, 'om' = Oromo (macro), 'mg' = Malagasy "
            "(macro). 'lu' = Luba-Katanga is NOT Luba-Kasai (lua) -> proxy only; 'ber' = Tamazight "
            "(Tfng), 'ber_Latn' = Tamazight (Latn). Many FLORES languages absent from the vocab (bem,"
            " tum, umb, mos, ki, luo, kam, kea, kab, lua) score ~2 chrF for 3B/10B in the paper's "
            'Table 17 (7.2B 10-18): not supported. Max training sequence length 256 tokens (paper '
            'table: MT models). hf: class present in transformers 5.19.0 (venv_gemma import check, '
            'CPU only, 2026-10-20); vllm: T5ForConditionalGeneration is NOT in the vllm 0.31.0 '
            'ModelRegistry (venv_vllm import check, no GPU); vLLM dropped T5-style encoder-decoder '
            'support (PR #11901 closed unmerged 2025-10-02, per research notes); ctranslate2: T5 '
            'converter listed in CTranslate2 docs/guides/transformers.md (master, retrieved '
            '2026-10-20); package NOT installed on the rental and not tested (CT2 build for sm_120 '
            'Blackwell unconfirmed) (community CT2 conversions exist: santhosh/madlad400-3b-ct2, '
            'Nextcloud-AI/madlad400-3b-mt-ct2-int8, LifeAi-dev/madlad400-10b-ct2-float32) VRAM = '
            'weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache '
            'at sentence level.'
        ),
    ),
    'translategemma-4b-it': SystemSpec(
        name='translategemma-4b-it',
        hf_id='google/translategemma-4b-it',
        revision='10042cb0e6e7fdce748996a71dc3dc432a4e0c89',
        family='gemma',
        prompt_style='chat',
        template=(
            'You are a professional {src_lang} (en) to {tgt_lang} ({tgt_code}) translator. Your goal '
            'is to accurately convey the meaning and nuances of the original {src_lang} text while '
            'adhering to {tgt_lang} grammar, vocabulary, and cultural sensitivities.\nProduce only the'
            ' {tgt_lang} translation, without any additional explanations or commentary. Please '
            'translate the following {src_lang} text into {tgt_lang}:\n\n\n{text}'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=4.97,
        vram_gb_fp16=9.9,
        vram_gb_int8=5.0,
        max_input_tokens=2048,
        weights_licence='gemma',
        licence_class='gemma_terms',
        licence_url='https://ai.google.dev/gemma/terms',
        output_note=(
            'Gemma Terms of Use (page last modified 2026-04-01): Google claims no rights in Outputs '
            "(s3.3), but a model trained on synthetic Outputs of Gemma is a 'Model Derivative' "
            '(s1.1(e)) and inherits the use restrictions; Outputs themselves are not Model '
            'Derivatives, and s3.1 pass-through duties are not clearly triggered by a pure output '
            'dataset (ambiguous; get legal review before release).'
        ),
        card_url='https://huggingface.co/google/translategemma-4b-it',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'pcm': 'pcm', 'efi': 'efi', 'twi': 'ak', 'ewe': 'ee',
            'fon': 'fon', 'wol': 'wo', 'swh': 'sw', 'kin': 'rw', 'som': 'so', 'orm': 'om', 'amh': 'am',
            'tir': 'ti', 'lug': 'lg', 'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'afr': 'af', 'sna': 'sn',
            'nya': 'ny', 'sot': 'st', 'tsn': 'tn', 'mlg': 'mg', 'bem': 'bem', 'tum': 'tum', 'run': 'rn',
            'ven': 've', 'tso': 'ts', 'nso': 'nso', 'ssw': 'ss', 'nbl': 'nr', 'sag': 'sg', 'tiv': 'tiv',
            'mos': 'mos', 'kik': 'ki', 'luo': 'luo', 'dyu': 'dyu', 'bam': 'bm', 'kon': 'kg', 'nus': 'nus',
            'lua': 'lua', 'iso': 'iso', 'ach': 'ach', 'ndc': 'ndc-ZW', 'kri': 'kri',
        },
        proxy_codes={
            'swc': 'sw', 'loz': 'st', 'tsc': 'ts', 'wes': 'pcm', 'ibb': 'efi', 'kab': 'ber-Latn',
            'shi': 'ber', 'nyn': 'cgg',
        },
        notes=(
            'Codes = those named in the TranslateGemma technical report (arXiv:2601.09012) Appendix C'
            ' tables 5/6 (SFT data: human + Gemini-synthetic parallel data), plus zu evaluated on '
            'WMT24++. IMPORTANT: the model card says it is designed for 55 languages (WMT24++); of '
            'our languages only sw and zu were EVALUATED by the authors (MetricX en->sw_KE '
            '4.45/5.36/10.65, en->zu_ZA 6.99/10.73/18.29 for 27B/12B/4B; published, lower is better);'
            " all other codes here are 'named in the SFT data list', quality unmeasured. Template = "
            'the hard-coded prompt in the official chat_template.jinja (read from the ungated '
            'mlx-community/translategemma-4b-it-4bit copy; the google repos are gated and the '
            'template file could not be read directly - verify against the gated copy once access is '
            'granted). The template raises/blanks for codes missing from its built-in name table; '
            'TEMPLATE-INVALID codes among ours: pcm efi fon bem tum nso tiv mos luo dyu kg nus lua '
            'iso ach ndc-ZW kri -> build the Gemma turn string by hand '
            "('<bos><start_of_turn>user\\n'+prompt+'<end_of_turn>\\n<start_of_turn>model\\n') with the "
            'English language name from LANGUAGES[iso].name. For vLLM use offline generate() with the'
            ' hand-built prompt string (the official template needs list-typed content, which '
            'OpenAI-style chat endpoints do not send). Proxies: kab->ber-Latn and shi->ber '
            '(Tamazight, Latin/Tifinagh), nyn->cgg (Kiga), swc->sw, wes->pcm. TG text-only CT2 '
            'conversion not verified (model is Gemma3ForConditionalGeneration incl. SigLIP vision '
            'tower; params shown include it, ~0.4B). Input context stated on card: 2K tokens. hf: '
            'class present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20); '
            'vllm: Gemma3ForConditionalGeneration registered in vllm 0.31.0 ModelRegistry (import '
            'check only, model not run) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add'
            ' ~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'translategemma-12b-it': SystemSpec(
        name='translategemma-12b-it',
        hf_id='google/translategemma-12b-it',
        revision='d1b225e1caa17f1ddc7e62065d8637d0923f34e2',
        family='gemma',
        prompt_style='chat',
        template=(
            'You are a professional {src_lang} (en) to {tgt_lang} ({tgt_code}) translator. Your goal '
            'is to accurately convey the meaning and nuances of the original {src_lang} text while '
            'adhering to {tgt_lang} grammar, vocabulary, and cultural sensitivities.\nProduce only the'
            ' {tgt_lang} translation, without any additional explanations or commentary. Please '
            'translate the following {src_lang} text into {tgt_lang}:\n\n\n{text}'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=13.19,
        vram_gb_fp16=26.4,
        vram_gb_int8=13.2,
        max_input_tokens=2048,
        weights_licence='gemma',
        licence_class='gemma_terms',
        licence_url='https://ai.google.dev/gemma/terms',
        output_note=(
            'Gemma Terms of Use (page last modified 2026-04-01): Google claims no rights in Outputs '
            "(s3.3), but a model trained on synthetic Outputs of Gemma is a 'Model Derivative' "
            '(s1.1(e)) and inherits the use restrictions; Outputs themselves are not Model '
            'Derivatives, and s3.1 pass-through duties are not clearly triggered by a pure output '
            'dataset (ambiguous; get legal review before release).'
        ),
        card_url='https://huggingface.co/google/translategemma-12b-it',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'pcm': 'pcm', 'efi': 'efi', 'twi': 'ak', 'ewe': 'ee',
            'fon': 'fon', 'wol': 'wo', 'swh': 'sw', 'kin': 'rw', 'som': 'so', 'orm': 'om', 'amh': 'am',
            'tir': 'ti', 'lug': 'lg', 'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'afr': 'af', 'sna': 'sn',
            'nya': 'ny', 'sot': 'st', 'tsn': 'tn', 'mlg': 'mg', 'bem': 'bem', 'tum': 'tum', 'run': 'rn',
            'ven': 've', 'tso': 'ts', 'nso': 'nso', 'ssw': 'ss', 'nbl': 'nr', 'sag': 'sg', 'tiv': 'tiv',
            'mos': 'mos', 'kik': 'ki', 'luo': 'luo', 'dyu': 'dyu', 'bam': 'bm', 'kon': 'kg', 'nus': 'nus',
            'lua': 'lua', 'iso': 'iso', 'ach': 'ach', 'ndc': 'ndc-ZW', 'kri': 'kri',
        },
        proxy_codes={
            'swc': 'sw', 'loz': 'st', 'tsc': 'ts', 'wes': 'pcm', 'ibb': 'efi', 'kab': 'ber-Latn',
            'shi': 'ber', 'nyn': 'cgg',
        },
        notes=(
            'Codes = those named in the TranslateGemma technical report (arXiv:2601.09012) Appendix C'
            ' tables 5/6 (SFT data: human + Gemini-synthetic parallel data), plus zu evaluated on '
            'WMT24++. IMPORTANT: the model card says it is designed for 55 languages (WMT24++); of '
            'our languages only sw and zu were EVALUATED by the authors (MetricX en->sw_KE '
            '4.45/5.36/10.65, en->zu_ZA 6.99/10.73/18.29 for 27B/12B/4B; published, lower is better);'
            " all other codes here are 'named in the SFT data list', quality unmeasured. Template = "
            'the hard-coded prompt in the official chat_template.jinja (read from the ungated '
            'mlx-community/translategemma-4b-it-4bit copy; the google repos are gated and the '
            'template file could not be read directly - verify against the gated copy once access is '
            'granted). The template raises/blanks for codes missing from its built-in name table; '
            'TEMPLATE-INVALID codes among ours: pcm efi fon bem tum nso tiv mos luo dyu kg nus lua '
            'iso ach ndc-ZW kri -> build the Gemma turn string by hand '
            "('<bos><start_of_turn>user\\n'+prompt+'<end_of_turn>\\n<start_of_turn>model\\n') with the "
            'English language name from LANGUAGES[iso].name. For vLLM use offline generate() with the'
            ' hand-built prompt string (the official template needs list-typed content, which '
            'OpenAI-style chat endpoints do not send). Proxies: kab->ber-Latn and shi->ber '
            '(Tamazight, Latin/Tifinagh), nyn->cgg (Kiga), swc->sw, wes->pcm. TG text-only CT2 '
            'conversion not verified (model is Gemma3ForConditionalGeneration incl. SigLIP vision '
            'tower; params shown include it, ~0.4B). Input context stated on card: 2K tokens. hf: '
            'class present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20); '
            'vllm: Gemma3ForConditionalGeneration registered in vllm 0.31.0 ModelRegistry (import '
            'check only, model not run) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add'
            ' ~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'translategemma-27b-it': SystemSpec(
        name='translategemma-27b-it',
        hf_id='google/translategemma-27b-it',
        revision='7d10f0b72f89a2d0f268cea30727d8b77c0d25c2',
        family='gemma',
        prompt_style='chat',
        template=(
            'You are a professional {src_lang} (en) to {tgt_lang} ({tgt_code}) translator. Your goal '
            'is to accurately convey the meaning and nuances of the original {src_lang} text while '
            'adhering to {tgt_lang} grammar, vocabulary, and cultural sensitivities.\nProduce only the'
            ' {tgt_lang} translation, without any additional explanations or commentary. Please '
            'translate the following {src_lang} text into {tgt_lang}:\n\n\n{text}'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=28.84,
        vram_gb_fp16=57.7,
        vram_gb_int8=28.8,
        max_input_tokens=2048,
        weights_licence='gemma',
        licence_class='gemma_terms',
        licence_url='https://ai.google.dev/gemma/terms',
        output_note=(
            'Gemma Terms of Use (page last modified 2026-04-01): Google claims no rights in Outputs '
            "(s3.3), but a model trained on synthetic Outputs of Gemma is a 'Model Derivative' "
            '(s1.1(e)) and inherits the use restrictions; Outputs themselves are not Model '
            'Derivatives, and s3.1 pass-through duties are not clearly triggered by a pure output '
            'dataset (ambiguous; get legal review before release).'
        ),
        card_url='https://huggingface.co/google/translategemma-27b-it',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'pcm': 'pcm', 'efi': 'efi', 'twi': 'ak', 'ewe': 'ee',
            'fon': 'fon', 'wol': 'wo', 'swh': 'sw', 'kin': 'rw', 'som': 'so', 'orm': 'om', 'amh': 'am',
            'tir': 'ti', 'lug': 'lg', 'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'afr': 'af', 'sna': 'sn',
            'nya': 'ny', 'sot': 'st', 'tsn': 'tn', 'mlg': 'mg', 'bem': 'bem', 'tum': 'tum', 'run': 'rn',
            'ven': 've', 'tso': 'ts', 'nso': 'nso', 'ssw': 'ss', 'nbl': 'nr', 'sag': 'sg', 'tiv': 'tiv',
            'mos': 'mos', 'kik': 'ki', 'luo': 'luo', 'dyu': 'dyu', 'bam': 'bm', 'kon': 'kg', 'nus': 'nus',
            'lua': 'lua', 'iso': 'iso', 'ach': 'ach', 'ndc': 'ndc-ZW', 'kri': 'kri',
        },
        proxy_codes={
            'swc': 'sw', 'loz': 'st', 'tsc': 'ts', 'wes': 'pcm', 'ibb': 'efi', 'kab': 'ber-Latn',
            'shi': 'ber', 'nyn': 'cgg',
        },
        notes=(
            'Codes = those named in the TranslateGemma technical report (arXiv:2601.09012) Appendix C'
            ' tables 5/6 (SFT data: human + Gemini-synthetic parallel data), plus zu evaluated on '
            'WMT24++. IMPORTANT: the model card says it is designed for 55 languages (WMT24++); of '
            'our languages only sw and zu were EVALUATED by the authors (MetricX en->sw_KE '
            '4.45/5.36/10.65, en->zu_ZA 6.99/10.73/18.29 for 27B/12B/4B; published, lower is better);'
            " all other codes here are 'named in the SFT data list', quality unmeasured. Template = "
            'the hard-coded prompt in the official chat_template.jinja (read from the ungated '
            'mlx-community/translategemma-4b-it-4bit copy; the google repos are gated and the '
            'template file could not be read directly - verify against the gated copy once access is '
            'granted). The template raises/blanks for codes missing from its built-in name table; '
            'TEMPLATE-INVALID codes among ours: pcm efi fon bem tum nso tiv mos luo dyu kg nus lua '
            'iso ach ndc-ZW kri -> build the Gemma turn string by hand '
            "('<bos><start_of_turn>user\\n'+prompt+'<end_of_turn>\\n<start_of_turn>model\\n') with the "
            'English language name from LANGUAGES[iso].name. For vLLM use offline generate() with the'
            ' hand-built prompt string (the official template needs list-typed content, which '
            'OpenAI-style chat endpoints do not send). Proxies: kab->ber-Latn and shi->ber '
            '(Tamazight, Latin/Tifinagh), nyn->cgg (Kiga), swc->sw, wes->pcm. TG text-only CT2 '
            'conversion not verified (model is Gemma3ForConditionalGeneration incl. SigLIP vision '
            'tower; params shown include it, ~0.4B). Input context stated on card: 2K tokens. hf: '
            'class present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20); '
            'vllm: Gemma3ForConditionalGeneration registered in vllm 0.31.0 ModelRegistry (import '
            'check only, model not run) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add'
            ' ~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'gemma-3-27b-it': SystemSpec(
        name='gemma-3-27b-it',
        hf_id='google/gemma-3-27b-it',
        revision='005ad3404e59d6023443cb575daa05336842228a',
        family='gemma',
        prompt_style='chat',
        template=(
            'Translate the following {src_lang} text into {tgt_lang}. Produce only the translation, '
            'without explanations.\n\n{text}'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=27.43,
        vram_gb_fp16=54.9,
        vram_gb_int8=27.4,
        max_input_tokens=2048,
        weights_licence='gemma',
        licence_class='gemma_terms',
        licence_url='https://ai.google.dev/gemma/terms',
        output_note=(
            'Gemma Terms of Use (page last modified 2026-04-01): Google claims no rights in Outputs '
            "(s3.3), but a model trained on synthetic Outputs of Gemma is a 'Model Derivative' "
            '(s1.1(e)) and inherits the use restrictions; Outputs themselves are not Model '
            'Derivatives, and s3.1 pass-through duties are not clearly triggered by a pure output '
            'dataset (ambiguous; get legal review before release).'
        ),
        card_url='https://huggingface.co/google/gemma-3-27b-it',
        checked='2026-10-20',
        src_code='en',
        codes={
            'swh': 'sw', 'zul': 'zu',
        },
        proxy_codes={},
        notes=(
            "Card claims 'over 140 languages' without a list; codes = only the African languages with"
            ' a PUBLISHED Gemma-3 translation score (WMT24++ en->sw_KE MetricX 5.92, en->zu_ZA 9.05 '
            'for 27B, from the TranslateGemma report Table 4; published, not measured by us). All '
            'other languages: unlisted, test empirically. Template is generic (no recommended MT '
            'prompt on the card). hf: class present in transformers 5.19.0 (venv_gemma import check, '
            'CPU only, 2026-10-20); vllm: Gemma3ForConditionalGeneration registered in vllm 0.31.0 '
            'ModelRegistry (import check only, model not run); CTranslate2 docs list Gemma 3 (text '
            'only). VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for '
            'activations/KV cache at sentence level.'
        ),
    ),
    'gemma-4-31b-it': SystemSpec(
        name='gemma-4-31b-it',
        hf_id='google/gemma-4-31B-it',
        revision='842da3794eaa0b77d5f08bae87a17459d91ff475',
        family='gemma',
        prompt_style='chat',
        template=(
            'Translate the following {src_lang} text into {tgt_lang}. Produce only the translation, '
            'without explanations.\n\n{text}'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=31.27,
        vram_gb_fp16=62.5,
        vram_gb_int8=31.3,
        max_input_tokens=2048,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            "Apache-2.0 (card YAML license plus license_link to the Gemma 4 'Apache License 2.0' page"
            ' on ai.google.dev, retrieved 2026-10-20): no output restrictions and no Model-Derivative'
            ' clause, unlike Gemma 1-3.'
        ),
        card_url='https://huggingface.co/google/gemma-4-31B-it',
        checked='2026-10-20',
        src_code='en',
        codes={},
        proxy_codes={},
        notes=(
            "NO LANGUAGE LIST: the card says 'out-of-the-box support for 35+ languages, pre-trained "
            "on 140+ languages' without naming them; the technical report (arXiv:2607.02770) names no"
            ' African language. Included as a candidate permissive generalist; coverage must be '
            'established empirically in the bake-off. Template is generic. Apache-2.0 differs from '
            'the Gemma Terms used by Gemma 1-3 and TranslateGemma. hf: class present in transformers '
            '5.19.0 (venv_gemma import check, CPU only, 2026-10-20); vllm: '
            'Gemma4ForConditionalGeneration registered in vllm 0.31.0 ModelRegistry (import check '
            'only, model not run); CTranslate2 docs list Gemma 4 (text only). VRAM = weights only '
            '(params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache at sentence '
            'level.'
        ),
    ),
    'qwen3-32b': SystemSpec(
        name='qwen3-32b',
        hf_id='Qwen/Qwen3-32B',
        revision='9216db5781bf21249d130ec9da846c4624c16137',
        family='qwen',
        prompt_style='chat',
        template=(
            'Translate the following {src_lang} text into {tgt_lang}. Produce only the translation, '
            'without explanations.\n\n{text}'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=32.76,
        vram_gb_fp16=65.5,
        vram_gb_int8=32.8,
        max_input_tokens=2048,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note='Apache-2.0: no restriction on generated text.',
        card_url='https://huggingface.co/Qwen/Qwen3-32B',
        checked='2026-10-20',
        src_code='en',
        codes={
            'afr': 'afr', 'swh': 'swh', 'kea': 'kea',
        },
        proxy_codes={},
        notes=(
            'Card/report claim 119 languages without listing them; codes = African languages that '
            "appear in the Qwen3 report's Belebele language table (Table 36: afr, swh, kea) i.e. "
            'languages the authors evaluated, not necessarily translation-tuned. Use '
            'enable_thinking=False. Template generic. hf: class present in transformers 5.19.0 '
            '(venv_gemma import check, CPU only, 2026-10-20); vllm: Qwen3ForCausalLM registered in '
            'vllm 0.31.0 ModelRegistry (import check only, model not run); CTranslate2 docs list Qwen'
            ' 3. VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for '
            'activations/KV cache at sentence level.'
        ),
    ),
    'aya-101': SystemSpec(
        name='aya-101',
        hf_id='CohereLabs/aya-101',
        revision='e7dad472b9de8a30a00cf08a05c19003bf59028d',
        family='aya',
        prompt_style='t5prompt',
        template='Translate to {tgt_lang}: {text}',
        hf_id_template='',
        backends=('hf',),
        params_b=12.92,
        vram_gb_fp16=25.8,
        vram_gb_int8=12.9,
        max_input_tokens=1024,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note='Apache-2.0 per card (YAML license: apache-2.0); no output restriction stated.',
        card_url='https://huggingface.co/CohereLabs/aya-101',
        checked='2026-10-20',
        src_code='eng',
        codes={
            'yor': 'yor', 'hau': 'hau', 'ibo': 'ibo', 'twi': 'twi', 'swh': 'swa', 'som': 'som', 'amh': 'amh',
            'zul': 'zul', 'xho': 'xho', 'afr': 'afr', 'sna': 'sna', 'nya': 'nya', 'sot': 'sot', 'mlg': 'mlg',
            'nso': 'nso',
        },
        proxy_codes={
            'swc': 'swa', 'nbl': 'zul', 'loz': 'sot', 'ndc': 'sna',
        },
        notes=(
            'mT5-XXL based (13B), 101 languages (3-letter ISO codes in the card). Prompt follows the '
            "card example ('Translate to English: ...'); the target language is given by its English "
            'name. Coverage here = named in the 101-language list only (quality unmeasured by us). '
            'hf: class present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20)'
            ' (T5ForConditionalGeneration); vllm: T5ForConditionalGeneration is NOT in the vllm '
            '0.31.0 ModelRegistry (venv_vllm import check, no GPU); vLLM dropped T5-style '
            'encoder-decoder support (PR #11901 closed unmerged 2025-10-02, per research notes); '
            'CTranslate2 lists T5/UMT5 but mT5 conversion not verified. VRAM = weights only (params x'
            ' 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'tiny-aya-earth': SystemSpec(
        name='tiny-aya-earth',
        hf_id='CohereLabs/tiny-aya-earth',
        revision='0624fafbe200f930b89566fc4ec4da314ba48d89',
        family='aya',
        prompt_style='chat',
        template=(
            'Translate the following {src_lang} text into {tgt_lang}. Produce only the translation.\n\n'
            '{text}'
        ),
        hf_id_template='',
        backends=('hf',),
        params_b=3.35,
        vram_gb_fp16=6.7,
        vram_gb_int8=3.4,
        max_input_tokens=2048,
        weights_licence='CC-BY-NC-4.0 + Cohere Labs Acceptable Use Policy',
        licence_class='noncommercial',
        licence_url='https://cohere.com/cohere-labs-cc-by-nc-license',
        output_note=(
            "CC-BY-NC-4.0 with Cohere's Acceptable Use Policy and a non-commercial-use "
            'acknowledgement at download; the card does not address generated outputs.'
        ),
        card_url='https://huggingface.co/CohereLabs/tiny-aya-earth',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'wol': 'wo', 'swh': 'sw', 'amh': 'am', 'zul': 'zu',
            'xho': 'xh', 'sna': 'sn', 'mlg': 'mg',
        },
        proxy_codes={
            'swc': 'sw', 'nbl': 'zu', 'ndc': 'sn',
        },
        notes=(
            '3.35B, 70+ languages (10 of ours: amh hau ibo mlg sna swh wol xho yor zul). Weights are '
            "gated (accept form) and require non-commercial use. 'earth' is the African/West-Asian "
            "regional variant, 'global' the generalist; same language list in both cards. Prompt: "
            'card gives only chat-template usage, no MT prompt; template here is generic. config.json'
            ' is gated so the architecture could not be read (Cohere2? unverified) -> only hf listed.'
            ' Context 8K. Published (by McGill-NLP, AfriqueQwen card): tiny-aya-earth FLORES '
            'SSA-COMET 42.0 vs Gemma-3-4B 33.5 (not measured by us). hf: class present in '
            'transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20). VRAM = weights only'
            ' (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache at sentence '
            'level.'
        ),
    ),
    'tiny-aya-global': SystemSpec(
        name='tiny-aya-global',
        hf_id='CohereLabs/tiny-aya-global',
        revision='af89d219b53ed9b13b8a4645f9c8028973510324',
        family='aya',
        prompt_style='chat',
        template=(
            'Translate the following {src_lang} text into {tgt_lang}. Produce only the translation.\n\n'
            '{text}'
        ),
        hf_id_template='',
        backends=('hf',),
        params_b=3.35,
        vram_gb_fp16=6.7,
        vram_gb_int8=3.4,
        max_input_tokens=2048,
        weights_licence='CC-BY-NC-4.0 + Cohere Labs Acceptable Use Policy',
        licence_class='noncommercial',
        licence_url='https://cohere.com/cohere-labs-cc-by-nc-license',
        output_note=(
            "CC-BY-NC-4.0 with Cohere's Acceptable Use Policy and a non-commercial-use "
            'acknowledgement at download; the card does not address generated outputs.'
        ),
        card_url='https://huggingface.co/CohereLabs/tiny-aya-global',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'wol': 'wo', 'swh': 'sw', 'amh': 'am', 'zul': 'zu',
            'xho': 'xh', 'sna': 'sn', 'mlg': 'mg',
        },
        proxy_codes={
            'swc': 'sw', 'nbl': 'zu', 'ndc': 'sn',
        },
        notes=(
            '3.35B, 70+ languages (10 of ours: amh hau ibo mlg sna swh wol xho yor zul). Weights are '
            "gated (accept form) and require non-commercial use. 'earth' is the African/West-Asian "
            "regional variant, 'global' the generalist; same language list in both cards. Prompt: "
            'card gives only chat-template usage, no MT prompt; template here is generic. config.json'
            ' is gated so the architecture could not be read (Cohere2? unverified) -> only hf listed.'
            ' Context 8K. Published (by McGill-NLP, AfriqueQwen card): tiny-aya-earth FLORES '
            'SSA-COMET 42.0 vs Gemma-3-4B 33.5 (not measured by us). hf: class present in '
            'transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20). VRAM = weights only'
            ' (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache at sentence '
            'level.'
        ),
    ),
    'seamless-m4t-v2-large': SystemSpec(
        name='seamless-m4t-v2-large',
        hf_id='facebook/seamless-m4t-v2-large',
        revision='5f8cc790b19fc3f67a61c105133b20b34e3dcb76',
        family='seamless',
        prompt_style='seamless',
        template='',
        hf_id_template='',
        backends=('hf',),
        params_b=2.31,
        vram_gb_fp16=4.6,
        vram_gb_int8=2.3,
        max_input_tokens=256,
        weights_licence='CC-BY-NC-4.0',
        licence_class='noncommercial',
        licence_url='https://creativecommons.org/licenses/by-nc/4.0/legalcode',
        output_note='CC-BY-NC-4.0; the card is silent on generated outputs (research release).',
        card_url='https://huggingface.co/facebook/seamless-m4t-v2-large',
        checked='2026-10-20',
        src_code='eng',
        codes={
            'yor': 'yor', 'ibo': 'ibo', 'swh': 'swh', 'som': 'som', 'orm': 'gaz', 'amh': 'amh', 'lug': 'lug',
            'zul': 'zul', 'afr': 'afr', 'sna': 'sna', 'nya': 'nya', 'luo': 'luo',
        },
        proxy_codes={
            'swc': 'swh', 'nbl': 'zul', 'ndc': 'sna',
        },
        notes=(
            "Use SeamlessM4Tv2ForTextToText with tgt_lang=<3-letter code>, src_lang='eng'. Codes = "
            "African entries with Tx (text) TARGET support in the card's language table (afr amh "
            'gaz(West Central Oromo) ibo lug luo nya sna som swh yor zul; also arz ary fuv not ours);'
            ' the card YAML also lists ha, ln, wo, xh, kea but the table has no text-target entry for'
            ' them (kea: speech source only) so they are NOT counted. Params shown are for the full '
            '2.3B multitask checkpoint (safetensors metadata); the text-to-text submodel loads only '
            'the text encoder/decoder (smaller; size not measured). hf: class present in transformers'
            ' 5.19.0 (venv_gemma import check, CPU only, 2026-10-20) (SeamlessM4Tv2ForTextToText); '
            'vllm: SeamlessM4Tv2ForTextToText is NOT in the vllm 0.31.0 ModelRegistry (venv_vllm '
            'import check, no GPU); vLLM dropped T5-style encoder-decoder support (PR #11901 closed '
            'unmerged 2025-10-02, per research notes); CTranslate2 has no SeamlessM4T converter in '
            'the docs. VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for '
            'activations/KV cache at sentence level.'
        ),
    ),
    'mt0-xxl-mt': SystemSpec(
        name='mt0-xxl-mt',
        hf_id='bigscience/mt0-xxl-mt',
        revision='db491aacb6a2601e3e8aac52214dbf6af7cd3f49',
        family='mt0',
        prompt_style='t5prompt',
        template='Translate to {tgt_lang}: {text}',
        hf_id_template='',
        backends=('hf',),
        params_b=13.95,
        vram_gb_fp16=27.9,
        vram_gb_int8=13.9,
        max_input_tokens=1024,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note='Apache-2.0 per card (YAML); no output restriction stated.',
        card_url='https://huggingface.co/bigscience/mt0-xxl-mt',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'swh': 'sw', 'som': 'so', 'amh': 'am', 'zul': 'zu',
            'xho': 'xh', 'afr': 'af', 'sna': 'sn', 'nya': 'ny', 'sot': 'st', 'mlg': 'mg',
        },
        proxy_codes={
            'swc': 'sw', 'nbl': 'zu', 'loz': 'st', 'ndc': 'sn',
        },
        notes=(
            'mT5-XXL fine-tuned on xP3mt. Card YAML lists the 101 mC4 pretraining languages (African '
            'ones: af am ha ig mg ny sn so st sw xh yo zu); only the 46 xP3 languages were '
            'fine-tuned, so translation INTO most of these is expected to be poor. Prompt follows the'
            " xP3 style of the card examples ('Translate to English: ...'). Weak baseline, kept for "
            'completeness. hf: class present in transformers 5.19.0 (venv_gemma import check, CPU '
            'only, 2026-10-20) (MT5ForConditionalGeneration); vllm: MT5ForConditionalGeneration is '
            'NOT in the vllm 0.31.0 ModelRegistry (venv_vllm import check, no GPU); vLLM dropped '
            'T5-style encoder-decoder support (PR #11901 closed unmerged 2025-10-02, per research '
            'notes) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for '
            'activations/KV cache at sentence level.'
        ),
    ),
    'bloomz-7b1-mt': SystemSpec(
        name='bloomz-7b1-mt',
        hf_id='bigscience/bloomz-7b1-mt',
        revision='fba4ea6dc5719f12df4f36b4236aa042f6998f24',
        family='mt0',
        prompt_style='t5prompt',
        template='Translate to {tgt_lang}: {text}',
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=7.07,
        vram_gb_fp16=14.1,
        vram_gb_int8=7.1,
        max_input_tokens=1024,
        weights_licence='bigscience-bloom-rail-1.0',
        licence_class='other',
        licence_url='https://huggingface.co/spaces/bigscience/license',
        output_note=(
            'BigScience BLOOM RAIL 1.0: use-based restrictions apply to the model and its derivatives'
            ' (licence text not re-read by me today; check before relying on it).'
        ),
        card_url='https://huggingface.co/bigscience/bloomz-7b1-mt',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yo', 'ibo': 'ig', 'twi': 'tw', 'fon': 'fon', 'wol': 'wo', 'swh': 'sw', 'kin': 'rw',
            'lug': 'lg', 'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'sna': 'sn', 'nya': 'ny', 'sot': 'st',
            'tsn': 'tn', 'tum': 'tum', 'run': 'rn', 'tso': 'ts', 'nso': 'nso', 'kik': 'ki', 'bam': 'bm',
        },
        proxy_codes={
            'swc': 'sw', 'nbl': 'zu', 'loz': 'st', 'ndc': 'sn', 'tsc': 'ts',
        },
        notes=(
            'BLOOM pretraining languages from the card YAML (Niger-Congo: ak bm fon ig ki lg ln nso '
            'ny rn rw sn st sw tn ts tum tw wo xh yo zu). Fine-tuned on xP3mt. Weak baseline for '
            'English->African MT. hf: class present in transformers 5.19.0 (venv_gemma import check, '
            'CPU only, 2026-10-20); vllm: BloomForCausalLM registered in vllm 0.31.0 ModelRegistry '
            '(import check only, model not run) VRAM = weights only (params x 2 B fp16/bf16, x 1 B '
            'int8); add ~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'opus-mt-en-x': SystemSpec(
        name='opus-mt-en-x',
        hf_id='',
        revision='',
        family='marian',
        prompt_style='marian',
        template='',
        hf_id_template='Helsinki-NLP/opus-mt-en-{code}',
        backends=('hf', 'ctranslate2'),
        params_b=0.077,
        vram_gb_fp16=0.2,
        vram_gb_int8=0.1,
        max_input_tokens=512,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            'Apache-2.0 per Hub card metadata (OPUS-MT cards); no output restriction. Training data '
            'are OPUS corpora (JW300/Bible/GlobalVoices etc.) of mixed provenance, so the NMT outputs'
            ' may mirror religious-domain text.'
        ),
        card_url='https://huggingface.co/Helsinki-NLP',
        checked='2026-10-20',
        src_code='en',
        codes={
            'hau': 'ha', 'ibo': 'ig', 'efi': 'efi', 'twi': 'tw', 'ewe': 'ee', 'swh': 'sw', 'kin': 'rw',
            'orm': 'om', 'tir': 'ti', 'lug': 'lg', 'lin': 'ln', 'xho': 'xh', 'afr': 'af', 'sna': 'sn',
            'nya': 'ny', 'sot': 'st', 'tsn': 'tn', 'mlg': 'mg', 'bem': 'bem', 'run': 'rn', 'tso': 'ts',
            'nso': 'nso', 'ssw': 'ss', 'sag': 'sg', 'tiv': 'tiv', 'umb': 'umb', 'mos': 'mos', 'luo': 'luo',
            'kon': 'kg', 'swc': 'swc', 'lua': 'lua', 'kqn': 'kqn', 'lun': 'lun', 'iso': 'iso', 'loz': 'loz',
        },
        proxy_codes={
            'ndc': 'sn', 'tsc': 'ts', 'ibb': 'efi',
        },
        notes=(
            'One Marian transformer-align model per target (2020 OPUS-MT, ~75M params inferred from '
            'the 0.3 GB checkpoint, sentence-level, SentencePiece). hf_id_template.format(code=<codes'
            ' value>); existing repos verified on the Hub 2026-10-20 (per-repo SHAs in '
            "system_cards.json). 'rn' and 'run' (Kirundi) both exist, 'rn' used. 'lu' (Luba-Katanga),"
            " 'lue' (Luvale), 'toi' also exist but are not ours. NO en-zu, en-yo, en-so, en-wo, "
            'en-tum, en-ven, en-nbl, en-ak models exist. Benchmarks on cards are old (e.g. en-sw '
            'GlobalVoices BLEU 24.2 / chrF 0.527, published). Only languages where this is the sole '
            'open option are really interesting: efi, iso, tiv, umb, mos, luo, loz, lun, kqn, lua, '
            'swc. hf: class present in transformers 5.19.0 (venv_gemma import check, CPU only, '
            '2026-10-20) (MarianMTModel); vllm: MarianMTModel is NOT in the vllm 0.31.0 ModelRegistry'
            ' (venv_vllm import check, no GPU); vLLM dropped T5-style encoder-decoder support (PR '
            '#11901 closed unmerged 2025-10-02, per research notes); ctranslate2: MarianMT converter '
            'listed in CTranslate2 docs/guides/transformers.md (master, retrieved 2026-10-20); '
            'package NOT installed on the rental and not tested (CT2 build for sm_120 Blackwell '
            'unconfirmed) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for '
            'activations/KV cache at sentence level.'
        ),
    ),
    'opus-mt-tc-bible-big-eng-mul': SystemSpec(
        name='opus-mt-tc-bible-big-eng-mul',
        hf_id='Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-mul',
        revision='160f64269da14c836871f8acf97553b23c63b979',
        family='marian',
        prompt_style='marian',
        template='>>{tgt_code}<< {text}',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=0.241,
        vram_gb_fp16=0.5,
        vram_gb_int8=0.2,
        max_input_tokens=512,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            'Apache-2.0 per Hub card metadata (OPUS-MT cards); no output restriction. Training data '
            'are OPUS corpora (JW300/Bible/GlobalVoices etc.) of mixed provenance, so the NMT outputs'
            ' may mirror religious-domain text.'
        ),
        card_url='https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-mul',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yor', 'hau': 'hau', 'ibo': 'ibo', 'pcm': 'pcm', 'efi': 'efi', 'twi': 'twi', 'ewe': 'ewe',
            'fon': 'fon', 'wol': 'wol', 'swh': 'swh', 'kin': 'kin', 'som': 'som', 'orm': 'orm', 'amh': 'amh',
            'tir': 'tir', 'lug': 'lug', 'lin': 'lin', 'zul': 'zul', 'xho': 'xho', 'afr': 'afr', 'sna': 'sna',
            'nya': 'nya', 'sot': 'sot', 'tsn': 'tsn', 'mlg': 'mlg', 'bem': 'bem', 'tum': 'tum', 'run': 'run',
            'ven': 'ven', 'tso': 'tso', 'nso': 'nso', 'ssw': 'ssw', 'nbl': 'nbl', 'sag': 'sag', 'umb': 'umb',
            'kmb': 'kmb', 'mos': 'mos', 'kik': 'kik', 'luo': 'luo', 'kam': 'kam', 'shi': 'shi', 'dyu': 'dyu',
            'bam': 'bam', 'kbp': 'kbp', 'kon': 'kon', 'kea': 'kea', 'kab': 'kab', 'nus': 'nus', 'swc': 'swc',
            'lua': 'lua', 'wal': 'wal', 'ach': 'ach', 'vmw': 'vmw', 'kua': 'kua', 'seh': 'seh', 'nyn': 'nyn',
            'her': 'her', 'kri': 'kri', 'bas': 'bas',
        },
        proxy_codes={
            'loz': 'sot', 'ndc': 'sna', 'tsc': 'tso', 'wes': 'pcm', 'ibb': 'efi',
        },
        notes=(
            'OPUS-MT-tc-bible-big (2024, transformer-big, trained on OPUS data plus the JHU Bible '
            "corpus - 'jhubc' in the checkpoint name - source side deu+eng+fra+por+spa). Prompt = "
            "'>>xxx<< ' + English text (xxx = ISO-639-3 target label from the codes column; "
            "prompt_style 'marian' with a fixed hf_id and a template). 'Target Language(s)' line of "
            'the card is used as the explicit list; presence in the list means only that some '
            '(possibly tiny, Bible-only) data existed - NOT that quality is usable. 1196 target '
            'labels in total. The repo has NO per-language FLORES scores (benchmark_results.txt only '
            'has a multi-multi Tatoeba average chrF 0.55); use the Bantu/Afro-Asiatic specialist '
            'cards for published numbers. hf: class present in transformers 5.19.0 (venv_gemma import'
            ' check, CPU only, 2026-10-20); vllm: MarianMTModel is NOT in the vllm 0.31.0 '
            'ModelRegistry (venv_vllm import check, no GPU); vLLM dropped T5-style encoder-decoder '
            'support (PR #11901 closed unmerged 2025-10-02, per research notes); ctranslate2: '
            'MarianMT converter listed in CTranslate2 docs/guides/transformers.md (master, retrieved '
            '2026-10-20); package NOT installed on the rental and not tested (CT2 build for sm_120 '
            'Blackwell unconfirmed) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add '
            '~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'opus-mt-tc-bible-big-eng-bnt': SystemSpec(
        name='opus-mt-tc-bible-big-eng-bnt',
        hf_id='Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-bnt',
        revision='ad19d20784f2a0213872851d3d998590f4889e45',
        family='marian',
        prompt_style='marian',
        template='>>{tgt_code}<< {text}',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=0.24,
        vram_gb_fp16=0.5,
        vram_gb_int8=0.2,
        max_input_tokens=512,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            'Apache-2.0 per Hub card metadata (OPUS-MT cards); no output restriction. Training data '
            'are OPUS corpora (JW300/Bible/GlobalVoices etc.) of mixed provenance, so the NMT outputs'
            ' may mirror religious-domain text.'
        ),
        card_url='https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-bnt',
        checked='2026-10-20',
        src_code='en',
        codes={
            'swh': 'swh', 'kin': 'kin', 'lug': 'lug', 'lin': 'lin', 'zul': 'zul', 'xho': 'xho', 'sna': 'sna',
            'nya': 'nya', 'sot': 'sot', 'tsn': 'tsn', 'bem': 'bem', 'tum': 'tum', 'run': 'run', 'ven': 'ven',
            'tso': 'tso', 'nso': 'nso', 'ssw': 'ssw', 'nbl': 'nbl', 'umb': 'umb', 'kmb': 'kmb', 'kik': 'kik',
            'kam': 'kam', 'kon': 'kon', 'swc': 'swc', 'lua': 'lua', 'vmw': 'vmw', 'kua': 'kua', 'seh': 'seh',
            'nyn': 'nyn', 'her': 'her', 'bas': 'bas',
        },
        proxy_codes={
            'loz': 'sot', 'ndc': 'sna', 'tsc': 'tso',
        },
        notes=(
            'OPUS-MT-tc-bible-big (2024, transformer-big, trained on OPUS data plus the JHU Bible '
            "corpus - 'jhubc' in the checkpoint name - source side deu+eng+fra+por+spa). Prompt = "
            "'>>xxx<< ' + English text (xxx = ISO-639-3 target label from the codes column; "
            "prompt_style 'marian' with a fixed hf_id and a template). 'Target Language(s)' line of "
            'the card is used as the explicit list; presence in the list means only that some '
            '(possibly tiny, Bible-only) data existed - NOT that quality is usable. Bantu specialist,'
            ' 78 targets. Published FLORES-200 devtest chrF (card benchmark_results.txt; chrF x100, '
            'sacrebleu chrF not chrF++): eng-kin 41.5, lin 45.6, nso 48.6, nya 45.1, sna 45.6, sot '
            '45.3, tsn 45.2, tso 48.5, xho 52.0, zul 53.3, bem 36.4, run 38.8, ssw 43.6, kon 37.8, '
            'lug 34.2, kik 25.5, kmb 22.7, kam 21.1, umb 25.2, lua 18.2, tum 12.0, swh 24.5 (very low'
            ' for a high-resource language; cause unexplained, possibly swa/swh/swc label confusion).'
            ' hf: class present in transformers 5.19.0 (venv_gemma import check, CPU only, '
            '2026-10-20); vllm: MarianMTModel is NOT in the vllm 0.31.0 ModelRegistry (venv_vllm '
            'import check, no GPU); vLLM dropped T5-style encoder-decoder support (PR #11901 closed '
            'unmerged 2025-10-02, per research notes); ctranslate2: MarianMT converter listed in '
            'CTranslate2 docs/guides/transformers.md (master, retrieved 2026-10-20); package NOT '
            'installed on the rental and not tested (CT2 build for sm_120 Blackwell unconfirmed) VRAM'
            ' = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV '
            'cache at sentence level.'
        ),
    ),
    'opus-mt-tc-bible-big-eng-afa': SystemSpec(
        name='opus-mt-tc-bible-big-eng-afa',
        hf_id='Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-afa',
        revision='90100f6582781668d640eca8e3cbfe36a9e6d156',
        family='marian',
        prompt_style='marian',
        template='>>{tgt_code}<< {text}',
        hf_id_template='',
        backends=('hf', 'ctranslate2'),
        params_b=0.24,
        vram_gb_fp16=0.5,
        vram_gb_int8=0.2,
        max_input_tokens=512,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            'Apache-2.0 per Hub card metadata (OPUS-MT cards); no output restriction. Training data '
            'are OPUS corpora (JW300/Bible/GlobalVoices etc.) of mixed provenance, so the NMT outputs'
            ' may mirror religious-domain text.'
        ),
        card_url='https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-afa',
        checked='2026-10-20',
        src_code='en',
        codes={
            'hau': 'hau', 'som': 'som', 'orm': 'orm', 'amh': 'amh', 'tir': 'tir', 'shi': 'shi', 'kab': 'kab',
            'wal': 'wal',
        },
        proxy_codes={},
        notes=(
            'OPUS-MT-tc-bible-big (2024, transformer-big, trained on OPUS data plus the JHU Bible '
            "corpus - 'jhubc' in the checkpoint name - source side deu+eng+fra+por+spa). Prompt = "
            "'>>xxx<< ' + English text (xxx = ISO-639-3 target label from the codes column; "
            "prompt_style 'marian' with a fixed hf_id and a template). 'Target Language(s)' line of "
            'the card is used as the explicit list; presence in the list means only that some '
            '(possibly tiny, Bible-only) data existed - NOT that quality is usable. Afro-Asiatic '
            'specialist, 62 targets. Published FLORES-200 devtest chrF (card): eng-hau 49.0, som '
            '42.8, amh 32.2, kab 24.8, tir 15.6; orm: FLORES-101 devtest 27.5, NTREX-128 28.5, '
            'TICO-19 33.0. hf: class present in transformers 5.19.0 (venv_gemma import check, CPU '
            'only, 2026-10-20); vllm: MarianMTModel is NOT in the vllm 0.31.0 ModelRegistry '
            '(venv_vllm import check, no GPU); vLLM dropped T5-style encoder-decoder support (PR '
            '#11901 closed unmerged 2025-10-02, per research notes); ctranslate2: MarianMT converter '
            'listed in CTranslate2 docs/guides/transformers.md (master, retrieved 2026-10-20); '
            'package NOT installed on the rental and not tested (CT2 build for sm_120 Blackwell '
            'unconfirmed) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for '
            'activations/KV cache at sentence level.'
        ),
    ),
    'sunflower-qwen3.8-27b': SystemSpec(
        name='sunflower-qwen3.8-27b',
        hf_id='Sunbird/Sunflower-Qwen3.8-27B',
        revision='42774a09a1893c5c83c5cb1cb521386eaa104fcd',
        family='qwen',
        prompt_style='chat',
        template='Translate to {tgt_lang}: {text}',
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=27.36,
        vram_gb_fp16=54.7,
        vram_gb_int8=27.4,
        max_input_tokens=2048,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            "Apache-2.0 per card ('License: Apache 2.0'); no output restriction stated. The card does"
            ' not document training data, so teacher-model terms (if any) are unknown.'
        ),
        card_url='https://huggingface.co/Sunbird/Sunflower-Qwen3.8-27B',
        checked='2026-10-20',
        src_code='English',
        codes={
            'yor': 'yor', 'hau': 'hau', 'ibo': 'ibo', 'pcm': 'pcm', 'ewe': 'ewe', 'wol': 'wol', 'swh': 'swa',
            'kin': 'kin', 'som': 'som', 'orm': 'orm', 'amh': 'amh', 'lug': 'lug', 'lin': 'lin', 'zul': 'zul',
            'xho': 'xho', 'afr': 'afr', 'sna': 'sna', 'nya': 'nya', 'sot': 'sot', 'tsn': 'tsn', 'mlg': 'mlg',
            'bem': 'bem', 'run': 'run', 'nbl': 'nbl', 'kik': 'kik', 'luo': 'luo', 'dag': 'dag', 'bam': 'bam',
            'kab': 'kab', 'ach': 'ach', 'nyn': 'nyn',
        },
        proxy_codes={
            'swc': 'swa', 'loz': 'sot', 'ndc': 'sna', 'wes': 'pcm', 'twi': 'aka', 'shi': 'ber',
        },
        notes=(
            'Gated on HF (auto-approve form; accept the terms with the account whose token is used). '
            "68 African languages (ISO-639-3 list in the card YAML, 'swa'/'mlg'/'orm'/'aka' are "
            "macro-language codes). Prompt on card: 'Translate to {language}: {text}' with the "
            'English language NAME; use enable_thinking=False and greedy decoding for bulk '
            'translation. Published by Sunbird (README, salt-69 test set, 100 sentences/language, '
            'chrF eng->xx, Sunflower-Qwen3.8-27B): afr 82.9, zul 68.4, nya 69.4, swa 68.1, xho 65.1, '
            'som 59.7, pcm 57.1, lin 60.8, sot 61.4, mlg 61.1, ibo 61.1, amh 50.3, hau 57.7, yor '
            '51.5, kin 59.6, tsn 57.2, lug 58.5, sna 52.5, nbl 47.4, run 50.6, ewe 52.0, luo 52.2, '
            'ach 51.2, orm 46.5, nyn 49.5, kik 40.2, aka 41.9, bem 46.3, bam 39.3, wol 34.4, kab '
            '33.4, dag 32.4. Mean eng->xx over the 68 languages 41.6 (Gemini 3.1 Pro 41.8); '
            'FLORES-200 (14 languages) mean both directions 53.5. All published, evaluated by Sunbird'
            ' with its own code, not measured by us. Proxies: twi->aka (Akan macro-language), '
            'shi->ber (generic Berber), wes->pcm. Training data and any synthetic-data teachers are '
            'not documented on the card. Base Qwen3.8-27B (Qwen3_5ForConditionalGeneration, ~55 GB '
            "bf16; card suggests vllm quantization='fp8' ~28 GB, max_model_len=2048). hf: class "
            'present in transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20); vllm: '
            'Qwen3_5ForConditionalGeneration registered in vllm 0.31.0 ModelRegistry (import check '
            'only, model not run) (card has a working vLLM example). VRAM = weights only (params x 2 '
            'B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'sunflower-qwen3.5-9b': SystemSpec(
        name='sunflower-qwen3.5-9b',
        hf_id='Sunbird/Sunflower-Qwen3.5-9B',
        revision='c56081c5024b1c11186a69898f5f6a7a2dabe454',
        family='qwen',
        prompt_style='chat',
        template='Translate to {tgt_lang}: {text}',
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=9.41,
        vram_gb_fp16=18.8,
        vram_gb_int8=9.4,
        max_input_tokens=2048,
        weights_licence='Apache-2.0',
        licence_class='permissive',
        licence_url='https://www.apache.org/licenses/LICENSE-2.0',
        output_note=(
            "Apache-2.0 per card ('License: Apache 2.0'); no output restriction stated. The card does"
            ' not document training data, so teacher-model terms (if any) are unknown.'
        ),
        card_url='https://huggingface.co/Sunbird/Sunflower-Qwen3.5-9B',
        checked='2026-10-20',
        src_code='English',
        codes={
            'yor': 'yor', 'hau': 'hau', 'ibo': 'ibo', 'pcm': 'pcm', 'ewe': 'ewe', 'wol': 'wol', 'swh': 'swa',
            'kin': 'kin', 'som': 'som', 'orm': 'orm', 'amh': 'amh', 'lug': 'lug', 'lin': 'lin', 'zul': 'zul',
            'xho': 'xho', 'afr': 'afr', 'sna': 'sna', 'nya': 'nya', 'sot': 'sot', 'tsn': 'tsn', 'mlg': 'mlg',
            'bem': 'bem', 'run': 'run', 'nbl': 'nbl', 'kik': 'kik', 'luo': 'luo', 'dag': 'dag', 'bam': 'bam',
            'kab': 'kab', 'ach': 'ach', 'nyn': 'nyn',
        },
        proxy_codes={
            'swc': 'swa', 'loz': 'sot', 'ndc': 'sna', 'wes': 'pcm', 'twi': 'aka', 'shi': 'ber',
        },
        notes=(
            'Gated on HF (auto-approve form; accept the terms with the account whose token is used). '
            "68 African languages (ISO-639-3 list in the card YAML, 'swa'/'mlg'/'orm'/'aka' are "
            "macro-language codes). Prompt on card: 'Translate to {language}: {text}' with the "
            'English language NAME; use enable_thinking=False and greedy decoding for bulk '
            'translation. Published by Sunbird (README, salt-69 test set, 100 sentences/language, '
            'chrF eng->xx, Sunflower-Qwen3.8-27B): afr 82.9, zul 68.4, nya 69.4, swa 68.1, xho 65.1, '
            'som 59.7, pcm 57.1, lin 60.8, sot 61.4, mlg 61.1, ibo 61.1, amh 50.3, hau 57.7, yor '
            '51.5, kin 59.6, tsn 57.2, lug 58.5, sna 52.5, nbl 47.4, run 50.6, ewe 52.0, luo 52.2, '
            'ach 51.2, orm 46.5, nyn 49.5, kik 40.2, aka 41.9, bem 46.3, bam 39.3, wol 34.4, kab '
            '33.4, dag 32.4. Mean eng->xx over the 68 languages 41.6 (Gemini 3.1 Pro 41.8); '
            'FLORES-200 (14 languages) mean both directions 53.5. All published, evaluated by Sunbird'
            ' with its own code, not measured by us. Proxies: twi->aka (Akan macro-language), '
            'shi->ber (generic Berber), wes->pcm. Training data and any synthetic-data teachers are '
            'not documented on the card. 9B variant (Qwen3_5ForConditionalGeneration): per the 27B '
            'card chrF 45.8 vs 46.2 on the 68-language set (published). Same language list as the 27B'
            ' (YAML lists 67/70 entries incl. eng/fra). hf: class present in transformers 5.19.0 '
            '(venv_gemma import check, CPU only, 2026-10-20); vllm: Qwen3_5ForConditionalGeneration '
            'registered in vllm 0.31.0 ModelRegistry (import check only, model not run) VRAM = '
            'weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache '
            'at sentence level.'
        ),
    ),
    'afriqueqwen3.5-9b-50langs-instruct': SystemSpec(
        name='afriqueqwen3.5-9b-50langs-instruct',
        hf_id='McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1',
        revision='693f12e444a30971ca53889128d7bade9491ecb3',
        family='qwen',
        prompt_style='chat',
        template=(
            'Translate the following sentence from {src_lang} to {tgt_lang}.\n\n{src_lang}: {text}\n'
            '{tgt_lang}:'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=8.95,
        vram_gb_fp16=17.9,
        vram_gb_int8=8.9,
        max_input_tokens=2048,
        weights_licence='CC-BY-4.0',
        licence_class='permissive',
        licence_url='https://creativecommons.org/licenses/by/4.0/legalcode',
        output_note=(
            'CC-BY-4.0 per card (attribution required); base Qwen3.5-9B-Base is Apache-2.0. The card '
            'says the continued-pretraining mix includes ~324M tokens of synthetic data translated by'
            " GPT-4.1 (OpenAI terms govern that training data, not this model's outputs) plus "
            'FineWeb2/WURA/MADLAD-400 text; no output restriction is stated.'
        ),
        card_url='https://huggingface.co/McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1',
        checked='2026-10-20',
        src_code='English',
        codes={
            'yor': 'yor_Latn', 'hau': 'hau_Latn', 'ibo': 'ibo_Latn', 'twi': 'twi_Latn', 'ewe': 'ewe_Latn',
            'fon': 'fon_Latn', 'wol': 'wol_Latn', 'swh': 'swh_Latn', 'kin': 'kin_Latn', 'som': 'som_Latn',
            'orm': 'gaz_Latn', 'amh': 'amh_Ethi', 'tir': 'tir_Ethi', 'lug': 'lug_Latn', 'lin': 'lin_Latn',
            'zul': 'zul_Latn', 'xho': 'xho_Latn', 'afr': 'afr_Latn', 'sna': 'sna_Latn', 'nya': 'nya_Latn',
            'sot': 'sot_Latn', 'tsn': 'tsn_Latn', 'mlg': 'plt_Latn', 'bem': 'bem_Latn', 'run': 'run_Latn',
            'tso': 'tso_Latn', 'nso': 'nso_Latn', 'ssw': 'ssw_Latn', 'sag': 'sag_Latn', 'kmb': 'kmb_Latn',
            'mos': 'mos_Latn', 'kik': 'kik_Latn', 'luo': 'luo_Latn', 'kam': 'kam_Latn', 'dyu': 'dyu_Latn',
            'bam': 'bam_Latn', 'kbp': 'kbp_Latn', 'kon': 'kon_Latn', 'kea': 'kea_Latn', 'kab': 'kab_Latn',
            'lua': 'lua_Latn',
        },
        proxy_codes={
            'swc': 'swh_Latn', 'nbl': 'zul_Latn', 'loz': 'sot_Latn', 'ndc': 'sna_Latn', 'tsc': 'tso_Latn',
            'shi': 'tzm_Tfng',
        },
        notes=(
            'McGill-NLP AfriqueLLM, post-trained (Instruct v1) from AfriqueQwen3.5-9B-50Langs '
            '(continued pre-training on 50 African languages, ~35.5B tokens). Supported-languages '
            'table in the base-model card gives FLORES-style codes (used here). Prompt from card '
            "('Translation' under Recommended Prompting): 'Translate the following sentence from "
            '{source language} to {target language}.\\n\\n{source language}: {sentence}\\n{target '
            "language}:' with enable_thinking=false, temperature 0, max_tokens 1024. Published (card,"
            ' by McGill-NLP): FLORES SSA-COMET 63.8 for this model vs Sunflower-Qwen3.5-9B 62.3, '
            'tiny-aya-earth 42.0, Gemma-3-4B 33.5, Qwen3.5-4B 44.4 (direction not stated; not '
            "measured by us). Sunbird's own README reports it far below Sunflower on its salt-69 set "
            "(chrF 28.8 eng->xx vs 41.6). twi: card lists 'Akan/Twi aka_Latn / twi_Latn' (explicit). "
            'Card says vLLM needs --language-model-only (checkpoint declares '
            'Qwen3_5ForConditionalGeneration but has no vision weights). 4B sibling: '
            'McGill-NLP/AfriqueQwen3.5-4B-50Langs-Instruct-v1. hf: class present in transformers '
            '5.19.0 (venv_gemma import check, CPU only, 2026-10-20); vllm: '
            'Qwen3_5ForConditionalGeneration registered in vllm 0.31.0 ModelRegistry (import check '
            'only, model not run) (card has vLLM serve command) VRAM = weights only (params x 2 B '
            'fp16/bf16, x 1 B int8); add ~10-25% for activations/KV cache at sentence level.'
        ),
    ),
    'llamax3-8b-alpaca': SystemSpec(
        name='llamax3-8b-alpaca',
        hf_id='LLaMAX/LLaMAX3-8B-Alpaca',
        revision='111f2fa166ff8bd3413757cbda9900374ce38f9e',
        family='other',
        prompt_style='t5prompt',
        template=(
            'Below is an instruction that describes a task, paired with an input that provides '
            'further context. Write a response that appropriately completes the request.\n### '
            'Instruction:\nTranslate the following sentences from {src_lang} to {tgt_lang}.\n### Input:'
            '\n{text}\n### Response:'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=8.03,
        vram_gb_fp16=16.1,
        vram_gb_int8=8.0,
        max_input_tokens=2048,
        weights_licence='Llama-3 Community License (Hub tag: MIT)',
        licence_class='llama',
        licence_url='https://github.com/meta-llama/llama3/blob/main/LICENSE',
        output_note=(
            'Hub tag says MIT but the weights are a Llama-3-8B derivative: the Llama 3 Community '
            'License (2024-04-18, read from github.com/meta-llama/llama3) applies, including clause '
            "1.b.v (no using Llama Materials or any output/results to improve another LLM) and 'Built"
            " with Meta Llama 3' attribution; pseudo-references from this model would carry that "
            'restriction.'
        ),
        card_url='https://huggingface.co/LLaMAX/LLaMAX3-8B-Alpaca',
        checked='2026-10-20',
        src_code='English',
        codes={
            'yor': 'yo', 'hau': 'ha', 'ibo': 'ig', 'wol': 'wo', 'swh': 'sw', 'som': 'so', 'orm': 'om',
            'amh': 'am', 'lug': 'lg', 'lin': 'ln', 'zul': 'zu', 'xho': 'xh', 'afr': 'af', 'sna': 'sn',
            'nya': 'ny', 'nso': 'ns', 'umb': 'umb', 'luo': 'luo', 'kam': 'kam', 'kea': 'kea',
        },
        proxy_codes={
            'swc': 'sw', 'nbl': 'zu', 'ndc': 'sn',
        },
        notes=(
            'Llama-3-8B continued-trained for translation on 100+ languages (Flores-101-style list, '
            "'ns' = Northern Sotho). Prompt = Alpaca template from the card. English-centric Alpaca "
            'fine-tune; quality for the rare languages unknown. hf: class present in transformers '
            '5.19.0 (venv_gemma import check, CPU only, 2026-10-20) (LlamaForCausalLM); vllm: '
            'LlamaForCausalLM registered in vllm 0.31.0 ModelRegistry (import check only, model not '
            'run) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for '
            'activations/KV cache at sentence level.'
        ),
    ),
    'n-atlas-llm': SystemSpec(
        name='n-atlas-llm',
        hf_id='NCAIR1/N-ATLaS',
        revision='e294476928aca9030e924ca27bb8e085e8581273',
        family='other',
        prompt_style='chat',
        template=(
            'Translate the following {src_lang} text into {tgt_lang}. Produce only the translation, '
            'without explanations.\n\n{text}'
        ),
        hf_id_template='',
        backends=('hf', 'vllm'),
        params_b=8.03,
        vram_gb_fp16=16.1,
        vram_gb_int8=8.0,
        max_input_tokens=2048,
        weights_licence='N-ATLaS Open-Source Research and Innovation License (custom, 1000-user cap)',
        licence_class='other',
        licence_url='https://huggingface.co/NCAIR1/N-ATLaS',
        output_note=(
            "README licence: custom 'Open-Source Research and Innovation License' - free for research"
            ' and organisations with at most 1000 active end-users, a commercial licence above that, '
            'derivative works under the same licence; Llama-3-8B base, so Llama 3 terms probably also'
            ' bind. Not suitable for a public permissive benchmark.'
        ),
        card_url='https://huggingface.co/NCAIR1/N-ATLaS',
        checked='2026-10-20',
        src_code='en',
        codes={
            'yor': 'yor', 'hau': 'hau', 'ibo': 'ibo',
        },
        proxy_codes={},
        notes=(
            'Nigerian NCAIR/Awarri Llama-3-8B fine-tune: Yoruba, Hausa, Igbo (+ Nigerian English) '
            'only; SFT data partly translated with Google Translate and OpenAI GPT. Gated (auto '
            'form). Template generic (README shows chat usage, no MT prompt). hf: class present in '
            'transformers 5.19.0 (venv_gemma import check, CPU only, 2026-10-20) (LlamaForCausalLM); '
            'vllm: LlamaForCausalLM registered in vllm 0.31.0 ModelRegistry (import check only, model'
            ' not run) VRAM = weights only (params x 2 B fp16/bf16, x 1 B int8); add ~10-25% for '
            'activations/KV cache at sentence level.'
        ),
    ),
}


# --------------------------------------------------------------------------- helpers
_LICENCE_RANK = {"permissive": 0, "gemma_terms": 1, "other": 2, "llama": 3, "noncommercial": 4}


def languages_covered(system_name: str) -> List[str]:
    """Our isos EXPLICITLY supported by the system, in ISO77 order."""
    s = SYSTEMS[system_name]
    return [i for i in ISO77 if i in s.codes]


def proxy_languages(system_name: str) -> List[str]:
    """Our isos covered only by a close variety (not counted as support)."""
    s = SYSTEMS[system_name]
    return [i for i in ISO77 if i in s.proxy_codes]


def coverage_counts() -> Dict[str, int]:
    """system name -> number of our 77 languages explicitly supported."""
    return {n: len(s.codes) for n, s in SYSTEMS.items()}


def proxy_counts() -> Dict[str, int]:
    return {n: len(s.proxy_codes) for n, s in SYSTEMS.items()}


def covering_systems(iso: str) -> List[str]:
    """Names of all systems that explicitly support ``iso``."""
    return [n for n, s in SYSTEMS.items() if iso in s.codes]


def best_licence_systems_for(iso: str) -> List[str]:
    """Systems that explicitly support ``iso`` and sit in the best licence class available for it
    (permissive < gemma_terms < other < llama < noncommercial).  [] if no system names the language."""
    cov = covering_systems(iso)
    if not cov:
        return []
    best = min(_LICENCE_RANK[SYSTEMS[n].licence_class] for n in cov)
    return [n for n in cov if _LICENCE_RANK[SYSTEMS[n].licence_class] == best]


def uncovered_languages() -> List[str]:
    """Isos named by NO system (explicit support only)."""
    return [i for i in ISO77 if not covering_systems(i)]


def only_nonpermissive_languages() -> List[str]:
    """Isos covered explicitly, but by no system of licence_class 'permissive'."""
    out = []
    for i in ISO77:
        cov = covering_systems(i)
        if cov and not any(SYSTEMS[n].licence_class == "permissive" for n in cov):
            out.append(i)
    return out


def coverage_matrix_markdown(systems: List[str] | None = None) -> str:
    """Markdown table: rows = our 77 languages, columns = systems; cell = the system's code,
    '<code>(p)' for a proxy, '-' for none.  Names come from the in-repo language table when importable."""
    names: Dict[str, str] = {}
    try:  # optional; the registry itself never needs it
        from gemma_mt.languages import LANGUAGES  # type: ignore

        names = {i: LANGUAGES[i].name for i in ISO77}
    except Exception:  # pragma: no cover - stdlib-only fallback
        names = {i: i for i in ISO77}
    cols = list(systems) if systems is not None else list(SYSTEMS)
    lines = ["| iso | language | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for iso in ISO77:
        cells = []
        for n in cols:
            s = SYSTEMS[n]
            if iso in s.codes:
                cells.append(s.codes[iso])
            elif iso in s.proxy_codes:
                cells.append(s.proxy_codes[iso] + "(p)")
            else:
                cells.append("-")
        lines.append(f"| {iso} | {names.get(iso, iso)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
