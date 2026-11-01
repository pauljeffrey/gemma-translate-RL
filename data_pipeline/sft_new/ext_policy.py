"""Which sources are NOT part of the SFT mix, and why (one place, read by combine.plan() and by the final merge).

HELD      built or buildable, but kept out of the mix until the owner decides (an explicit instruction conflicts, or a licence needs an override)
EXCLUDED  never trained on: evaluation sets, non-commercial licences (see license_scan.json), repos that are not usable
Every name here is reported at the end of the build with its size (docs/DATA_INVENTORY.md)."""
from __future__ import annotations
import os

# build steps (converter names, or `converter@family` prefixes) that must not enter the final mix
HELD = {
    "mmmlu_yoruba": "owner 2026-10-10: keep out (evaluation set MMMLU_yoruba)",
    "global_mmlu_dev": "benchmark data (Global-MMLU dev split): evaluation contamination",
    "mmlu_prox_validation": "benchmark data (MMLU-ProX validation split): evaluation contamination",
    "ghananlp_pristine": "licence cc-by-nc-4.0; the owner's 'use regardless of licence' names ghananlpcommunity/pristine-twi-english, not this -multilen repo: confirm",
    # owner 2026-10-10: remove the evaluation-derived sets
    "xsim_nli": "evaluation benchmark (xsim++ dev/devtest): removed on the owner's instruction",
    "gsm_symbolic_african": "evaluation benchmark (GSM-Symbolic test splits): removed on the owner's instruction",
    # owner 2026-10-10: machine translation by NLLB (weights CC-BY-NC-4.0, the licence of the non-commercial sets we already skip)
    "aya_language_split": "Aya collection translated data = NLLB-3.3B output (card): NLLB is CC-BY-NC-4.0",
    "aya_collection": "Aya collection (Hausa split of the same translated data): NLLB-3.3B output (card)",
    "aya_translate": "derived from the Aya collection's NLLB-3.3B translations",
    "aya_interlanguage": "derived from the Aya collection's NLLB-3.3B translations",
    "wori_wolof": "card: Wolof/French translated with galsenai/wolofToFrenchTranslator_nllb (NLLB fine-tune)",
    # owner 2026-10-11
    "wmt_human_mt": "human-scored MT outputs are quality-estimation / reward material: skipped for now (no RL-style data)",
    # owner 2026-10-13: dropped (2,910,664 rows). No dataset card; 35-62% of sampled pairs match michsethowusu/*_sentence-pairs exactly = the NLLB-mined
    # bitext class the catalog treats as NLLB-derived (standing rule: skip NLLB-derived); similarity-scored mined pairs, unverified, no licence
    "african_languages_translation": ("owner 2026-10-13: " + os.environ['HF_ORG'] + "/african_languages_translation is a re-pack of michsethowusu NLLB-mined *_sentence-pairs (standing rule: skip NLLB-derived)"),
}
HELD_PREFIXES = {
    "naija2@sib200_": "SIB-200 is an evaluation set (owner's note 1); the split of these files cannot be verified as train-only",
    "naija@sib200_": "SIB-200 is an evaluation set (owner's note 1); the split of these files cannot be verified as train-only",
}

EXCLUDED_REPOS = {
    # evaluation-only sets (owner's note 1, and the other worker's eval_exclusions.json)
    **{r: "evaluation set" for r in (
        (os.environ['HF_ORG'] + "/uhura-eval"), (os.environ['HF_ORG'] + "/uhura-eval-harness"), (os.environ['HF_ORG'] + "/uhura-truthfulqa"), (os.environ['HF_ORG'] + "/uhura-truthfulqa-clean"), (os.environ['HF_ORG'] + "/uhura-arc-easy"),
        (os.environ['HF_ORG'] + "/uhura-arc-easy-clean"), (os.environ['HF_ORG'] + "/uhura"), (os.environ['HF_ORG'] + "/mafand_eval"), (os.environ['HF_ORG'] + "/masakhaner_eval"), (os.environ['HF_ORG'] + "/afrixnli"),
        (os.environ['HF_ORG'] + "/afrimmlu"), (os.environ['HF_ORG'] + "/afrimgsm"), (os.environ['HF_ORG'] + "/MMMLU_yoruba"), (os.environ['HF_ORG'] + "/pretrain_test"), (os.environ['HF_ORG'] + "/wili_2018"),
        (os.environ['HF_ORG'] + "/xtreme-up-semantic-parsing"), (os.environ['HF_ORG'] + "/SemRel2024"), (os.environ['HF_ORG_LEGACY'] + "/afrisenti_eval"), (os.environ['HF_ORG_LEGACY'] + "/masakhanews_eval"),
        (os.environ['HF_ORG'] + "/sib200"), (os.environ['HF_ORG'] + "/sib200-swahili"), "mteb/AfriXNLI", "McGill-NLP/mgsm-pro", "sbmaruf/forai_ml_masakhane_mafand")},
    (os.environ['HF_ORG_LEGACY'] + "/french-pretraining-corpus"): "owner: do not handle",
    "ymoslem/UN-EN-AR-QE": "Arabic: the tokenizer has no <ara> tag (id 134 is French)",
    "ymoslem/news-commentary-eng-arz": "Arabic / Egyptian Arabic: the tokenizer has no tag for either",
    "mbaye930/wolof-arabic-parallel-corpus": "non-commercial licence; Arabic has no tag",
    "mteb/sib200.v2": "14 integer classes with no label names on the card (70 train rows per language)",
    "mteb/TswanaNewsClassification": "integer labels without names; the same articles are in dsfsi/daily-news-dikgang (converted)",
    "mteb/InjongoIntent": "same data as masakhane/InjongoIntent (converted); duplicates",
    # owner 2026-10-15 audit: sources present in final_v2 that the 2026-10-10 licence scan never saw (added by later catalog batches); the catalog
    # records CC BY-NC for all of them and the standing rule is "skip + log non-commercial licences with size"
    "lelapa/Inkuba-instruct": "CC BY-NC 4.0 (README) and machine-translated; 3,038,744 rows were in final_v2",
    **{f"saillab/{n}": "CC BY-NC (catalog, README); machine-translated Alpaca" for n in (
        "alpaca_xhosa_taco", "alpaca_ewe_taco", "alpaca_zulu_taco", "alpaca_amharic_taco", "alpaca_sepedi_taco", "alpaca_oromo_taco", "alpaca_kinyarwanda_taco",
        "alpaca_bambara_taco", "alpaca_afrikaans_taco", "alpaca_somali_taco", "alpaca_swahili_taco", "alpaca_sesotho_taco", "alpaca_lingala_taco",
        "alpaca_chichewa_taco", "alpaca_malagasy_taco", "alpaca_tsonga_taco", "alpaca_shona_taco", "alpaca_luganda_taco", "alpaca_tigrinya_taco",
        "alpaca_twi_taco", "alpaca_hausa_taco", "alpaca_yoruba_taco", "alpaca_igbo_taco", "alpaca-bambara-cleaned", "alpaca-amharic-cleaned",
        "alpaca-swahili-cleaned", "alpaca-oromo-cleaned", "alpaca-zulu-cleaned", "alpaca-lingala-cleaned", "alpaca-kinyarwanda-cleaned",
        "alpaca-sepedi-cleaned", "alpaca-somali-cleaned", "alpaca-luganda-cleaned", "alpaca-chichewa-cleaned", "alpaca-shona-cleaned",
        "alpaca-sesotho-cleaned", "alpaca-malagasy-cleaned", "alpaca-tsonga-cleaned", "alpaca-tigrinya-cleaned", "alpaca-afrikaans-cleaned",
        "alpaca-ewe-cleaned", "alpaca-xhosa-cleaned", "alpaca-twi-cleaned")},
}


def is_held(step: str) -> str | None:
    base = step.split("@", 1)[0]
    if base in HELD:
        return HELD[base]
    for p, why in HELD_PREFIXES.items():
        if step.startswith(p):
            return why
    return None
