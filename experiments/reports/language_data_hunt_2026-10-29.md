# Data hunt: English<->{bbj, dag, fat, gaa, knc, nde, tzm, ful, efi, tiv, bin, ish} (agent L1, 2026-10-29)

Method: Hub API (about 60 search terms), OPUS API, eBible `translations.csv`, FLORES+ and google/smol file trees, Common Voice 27 JSON, GhanaNLP and Efik papers, web search, and our own `data-catalog/african_hf` and `reports/data/sft_clean_2026-10-28/source_access_2026-10-28.json`. A few KB of samples only; no token used; nothing staged. "Had" = already in our catalog or source_access; "NEW" = in neither. Licence labels are the uploaders'.

## Cross-cutting findings
1. JW-derived "MT560" pairs are the bulk for efi, tiv, bin, ish and one Ga set (`michsethowusu/english-{efik,tiv,edo,esan}_sentence-pairs_mt560`, `Davlan/ibom-mt-en-efi`, `ghananlpcommunity/english-ga-sentence-pairs-400k`); samples are Watchtower text. OPUS no longer serves JW300. Treat as unknown rights, disclose.
2. NLLB-mined sets are note-only: OPUS NLLB en-tzm 212,768 and en-ff 1,511,026; HF copies `michsethowusu/english-central_atlas_tamazight_sentence-pairs` and `michsethowusu/english-fulah_sentence-pairs`; CCAligned en-ff (73,023, web-mined, no licence). Do not use.
3. `IsahMBukar/kanurimt` is contaminated (merges TWB Gamayun, TICO-19 (protected) and Tatoeba); already excluded.
4. GATITOS (lexicon part of `google/smol`, CC-BY-4.0, about 4k entries each for efi, ff, gaa, kr, tiv, ber) is in the same repo as protected files: ask the owner before using it.

## Protected, do not train
FLORES+ dev/devtest: `fuv_Latn`, `knc_Latn`/`knc_Arab`, `zgh_Tfng` (not bbj, dag, fat, gaa, nde, tzm, efi, tiv, bin, ish). TICO-19: en-fuv and en-kr (3,071 each; also inside KanuriMT). MAFAND-MT fr-bbj dev 1,133 / test 1,430 (train 2,232 usable but CC-BY-NC). IbomMT `eng-efi` dev 997 / test 1,012 (and `Davlan/ibom-mt-en-efi` dev/test, probably the same). SmolSent/SmolDoc: efi, ff, gaa, kr, tiv, ber (+ fuc). GlotStoryBook: dag, fat, gaa, ful, kau and others.

## Per language (best candidates)
* **bbj**: `stephanedonna/english_ghomala` (7,916 en-bbj NT pairs, no licence; use with disclosure); `flagship-ai/ghomala-spoken-bible` (about 20 h audio with aligned chapter text, card says CC-BY-4.0, uploader claim); `stfotso/french-ghomala-bandjoun` (15,190 fr-bbj dictionary entries, Apache label, copyright unverified, not English); MAFAND fr-bbj train (NC); Common Voice 27 `bbj` (CC0 transcripts, monolingual). Realistic: about 8k en-bbj.
* **dag**: `narteybrown/sobolo-corpus-dagbani-bible` (30,291 Bible pairs, YouVersion scrape, no licence; disclosure); OPUS translatewiki en-dag 7,455 (UI strings); OPUS wikimedia en-dag 902 (CC-BY-SA); `fiifinketia/navigation-corpus-dagbani` 4,642 (Had; GhanaNLP copies CC-BY-NC). Realistic: about 30k Bible + about 8k small.
* **fat**: `Ghana-NLP/FANTE_ENGLISH_PARALLEL_TEXT` (Had; 3,500 public rows of 5,001, non-commercial unless permitted); OPUS translatewiki 1,534; wikimedia 591 (CC-BY-SA). Realistic: about 2k clear + 3.5-5k NC.
* **gaa**: `Ghana-NLP/GA_ENGLISH_PARALLEL_TEXT` (NEW; 8,155 public of 11,652, NC unless permitted); `ghananlpcommunity/english-ga-sentence-pairs-400k` (Had, 409,659 JW-style, CC-BY-NC); `michsethowusu/ghana-chat-corpus-gaa` (Had, 226,300 machine-translated, NC); OPUS translatewiki 1,162. Realistic: about 1.2k clear.
* **knc**: `CLEAR-Global/Gamayun-kits` config `kau-eng` (5,000 Tatoeba sentences translated by the TWB community; CC-BY-4.0 under the TWB Gamayun terms, derivatives must be open and free; use with disclosure); `IsahMBukar/kanurimt` (skip); `xri/kanuri_synth_*` (LLM synthetic, skip); `ciatech-africa/kanuri-english-lexicon` (Had, 8,370 dictionary entries); OPUS translatewiki 597, GNOME 321. Monolingual: `CLEAR-Global/kanuri-books-corpus` 10,281 sentences (Had). Realistic: about 5.0-5.9k.
* **nde (North Ndebele, Zimbabwe)**: eBible `nde` (full Bible, CC BY-SA 4.0, Biblica; align by verse to a public-domain English Bible: about 31k pairs) is the best commercial-legal source; `SalomonMetre13/nde_en` (30,839 verses, KJV, no licence) and `k-mktr/ndebele_bible_nd` (nde only) as cross-checks. FineWeb-2 nde 22,889 docs (ODC-By). Realistic: about 31k.
* **tzm**: `Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight` (Had; 6,193 human-translated Wikipedia sentences, CC-BY-SA-4.0; NLLB-Seed provenance, check against the NLLB rule); translatewiki 1,038 and Pontoon 266 (Had); neighbouring `zgh`: Tatoeba en-zgh 12,548 (CC-BY 2.0 FR), Weblate en-zgh 5,479, translatewiki en-zgh 3,173. Realistic: about 7.5k tzm (+ about 12.5k zgh if treated as one).
* **ful**: OPUS wikimedia en-ff 23,533 (CC-BY-SA, largest clean set); OPUS XLEnt en-ff 7,132; `stephanedonna/english_fulfulde` (30,802 Adamawa Fulfulde `fub` Bible, no licence); `ARPRIM/pulaar_fulfulde` (Had, 1,708 fr-Pulaar, CC-BY-4.0, French not English); `pulaarai-mauritania/MauriCorpus-arabic-pulaar` (2,000 ar-Pulaar, CC-BY-NC-SA); TransFula (40,000, Apache label, provenance unstated, silver) and the silver sets (skip); eBible ffm/fuf/fue/fuhbkf (ND clauses, skip). Keep dialect codes apart. Realistic: about 23.5k clean en-ff + about 30k Adamawa Bible.
* **efi**: `michsethowusu/english-efik_sentence-pairs_mt560` (Had; 332,589 JW-derived); `howard-nlp/ibom-mt` train 1,000 (CC-BY-NC); `Davlan/ibom-mt-en-efi` train (same JW content, dev/test protected); Edet et al. 2026 (arXiv 2603.14873) 13,865 human pairs: no public release found, contact the authors; private `<HF_ORG>/English_efik_bible_translation` (Had); OPUS QED 2,446, translatewiki 539.
* **tiv**: `michsethowusu/english-tiv_sentence-pairs_mt560` (Had; 184,113 JW-derived); nothing else verified (`victorachede/tiv-translator-data` returns 401, inconsistent card).
* **bin (Edo)**: `michsethowusu/english-edo_sentence-pairs_mt560` (Had; 11,635 JW-derived); Orife (2020) also used JW300.
* **ish (Esan)**: `michsethowusu/english-esan_sentence-pairs_mt560` (Had; 5,221 JW-derived).

## Summary: pairs obtainable
| lang | clear (commercial OK) | unclear / disclose | NC only | protected |
|---|---|---|---|---|
| bbj | 0 verified | about 7.9k en NT Bible, about 15k fr dictionary | about 2.2k fr (MAFAND train) | MAFAND fr-bbj dev/test 2.6k |
| dag | about 0.9k + 7.5k UI strings | about 30k Bible | 4.6k navigation | GlotStoryBook |
| fat | about 0.6k + 1.5k UI strings | none | 3.5k public (5.0k total) | GlotStoryBook |
| gaa | about 1.2k UI strings | about 410k JW | 8.2k public (11.7k total), 226k synthetic chat | SmolSent/Doc, GlotStoryBook |
| knc | about 0.9k OPUS | 5.0k Gamayun (TWB terms) | none | FLORES+, TICO-19, KanuriMT overlap |
| nde | about 31k (CC-BY-SA eBible + public-domain English) | about 31k other Bible copies | none | none |
| tzm | about 6.2k (share-alike NLLB-Seed) + about 1.3k UI/Pontoon | none | none | none for tzm (FLORES+ has zgh) |
| ful | about 23.5k (en-ff wikimedia) + 7k XLEnt | about 30k Adamawa Bible, 45k silver, 1.7k fr-Pulaar | 2k ar-Pulaar | FLORES+ fuv, TICO-19 fuv, SmolSent/Doc ff/fuc |
| efi | about 3k small | about 330k JW; 13.9k human corpus unreleased | 1k (IbomMT train) | IbomMT dev/test, SmolSent/Doc |
| tiv | none | about 184k JW | none | SmolSent/Doc |
| bin | none | about 11.6k JW | none | none |
| ish | none | about 5.2k JW | none | none |

## Recommendations in priority order
1. nde: eBible `nde` (CC-BY-SA 4.0) paired with a public-domain English Bible (e.g. eBible `engwebp`): the only language here with a full, commercial-clear, human-translated set.
2. ful: OPUS wikimedia en-ff (share-alike); for Adamawa, `stephanedonna/english_fulfulde` with disclosure.
3. knc: Gamayun `kau-eng` 5k with the TWB terms; not KanuriMT.
4. tzm: NLLB-Seed `eng-tzm_Tfng` (6,193) if the owner accepts NLLB-Seed provenance.
5. dag: `sobolo-corpus-dagbani-bible` with disclosure.
6. bbj: `stephanedonna/english_ghomala` with disclosure, after asking the uploader.
7. efi, tiv, bin, ish: JW-derived mt560 is all that is public; ask the Edet et al. authors for the 13.9k human Efik corpus.
8. fat, gaa: only NC data exists (Ghana NLP); owner decision.

Links: arXiv 2603.13793 (GhanaNLP parallel corpora), arXiv 2603.14873 (English-Efik), eBible nde (https://ebible.org/find/details.php?id=nde), Autshumato en-isiNdebele (https://repo.sadilar.org/handle/20.500.12185/572; this is `nbl`, a different language).
