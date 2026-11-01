# Closed MT systems: target-language coverage for the 77 African languages (retrieved 2026-10-20)

Legend: a code = explicit support as a target language on the official page; `proxy: ...` = close but not identical language (caveat in note); `no` = not on the page's list; `unverified` = none (every language could be checked against the fetched lists).
Google column reflects the NMT list ("Translation LLM" list adds no African language that NMT lacks; its African entries are af, sw, zu, ln [Official] and am, ha, ig, so [Experimental], all of which are also in NMT). Microsoft column reflects the "Cloud - Text translation" column (no African languages are marked preview on the page; the "LLM translation" column has no African language ticked).

| iso | English name | Google Translate | Microsoft Translator | note |
|---|---|---|---|---|
| yor | Yoruba | yo | yo | |
| hau | Hausa | ha | ha | |
| ibo | Igbo | ig | ig | |
| pcm | Pidgin (Nigerian) | proxy: kri (Krio) | no | Krio is a related West-African English-lexified creole but a distinct language; weak proxy |
| efi | Efik | no | no | |
| twi | Twi | ak ("Twi (Akan)") | no | Google label is Twi (Akan); ak is the Akan macrolanguage code, treated as match for Twi |
| ewe | Ewe | ee | no | |
| fon | Fon | no | no | not on Google's list as of this fetch |
| wol | Wolof | no | no | not on Google's list as of this fetch |
| swh | Swahili | sw | sw | MS table label "Swahili (Latin)" |
| kin | Kinyarwanda | rw | rw | |
| som | Somali | so | so | MS table labels it "Somali (Arabic)" (code so); treat as so, label oddity noted |
| orm | Oromo | om | no | |
| amh | Amharic | am | am | |
| tir | Tigrinya | ti | ti | |
| lug | Luganda | lg | lug | |
| lin | Lingala | ln | ln | Google LLM table: Official |
| zul | Zulu | zu | zu | Google LLM table: Official |
| xho | Xhosa | xh | xh | |
| afr | Afrikaans | af | af | |
| sna | Shona | sn | sn | MS label chiShona |
| nya | Chichewa | ny | nya | MS label Nyanja |
| sot | Sesotho | st | st | |
| tsn | Setswana | tn | tn | |
| mlg | Malagasy | mg | mg | |
| ibb | Ibibio | no | no | |
| urh | Urhobo | no | no | |
| bem | Bemba | bem | no | |
| tum | Tumbuka | no | no | |
| run | Kirundi | rn | run | label Rundi |
| ven | Venda | no | no | Google's "Dombe (dov)" is a different language, not Venda |
| tso | Tsonga | ts | no | |
| nso | Sepedi | nso | nso | label Northern Sotho (Sepedi) / Sesotho sa Leboa |
| ssw | Swati | ss | no | |
| nbl | South Ndebele | nr | no | Google label "Ndebele (South)" |
| sag | Sango | sg | no | |
| tiv | Tiv | no | no | |
| umb | Umbundu | no | no | |
| kmb | Kimbundu | no | no | |
| mos | Mossi | no | no | |
| kik | Kikuyu | no | no | |
| luo | Luo | luo | no | Google "Luo" (Dholuo) |
| kam | Kamba | no | no | |
| bin | Edo | no | no | |
| dag | Dagbani | no | no | |
| oke | Okpe | no | no | |
| shi | Tachelhit | no | no | no Tamazight/Tachelhit/Tifinagh entry on either page |
| dyu | Dyula | no | no | not on Google's list as of this fetch |
| bam | Bambara | bm | no | |
| kbp | Kabiye | no | no | |
| bbj | Ghomala | no | no | |
| kon | Kongo | proxy: ktu (Kituba) | no | Kituba is a Kongo-based creole/lingua franca, not Kikongo proper; proxy only |
| kea | Kabuverdianu | no | no | |
| kab | Kabyle | no | no | |
| nus | Nuer | nus | no | |
| swc | Congo Swahili | proxy: sw (Swahili) | proxy: sw (Swahili) | only generic Swahili offered; the Congolese variety is not distinguished |
| lua | Luba-Kasai | no | no | |
| kqn | Kaonde | no | no | |
| lun | Lunda | no | no | |
| iso | Isoko | no | no | |
| loz | Lozi | no | no | |
| wal | Wolaytta | no | no | |
| ach | Acholi | ach | no | |
| tsc | Tswa | no | no | Google's Tsonga (ts) is a different language |
| vmw | Makhuwa | no | no | |
| kua | Kuanyama | no | no | |
| seh | Sena | no | no | |
| nyn | Nyankore | proxy: cgg (Kiga/Chiga) | no | Runyankore and Rukiga are closely related (often treated as one Runyankore-Rukiga); Google lists only Kiga |
| her | Herero | no | no | |
| yao | Yao | no | no | |
| sid | Sidamo | no | no | |
| ndc | Ndau | no | no | |
| kri | Krio | kri | no | |
| ish | Esan | no | no | |
| dua | Duala | no | no | |
| bas | Basaa | no | no | |
| wes | Cameroon Pidgin | no | no | Krio (Google) is only a loosely related creole; not claimed as proxy |

## Counts (out of 77)

| System | explicit | proxy | not supported |
|---|---|---|---|
| Google Translate (NMT list; Translation LLM list adds nothing new for these) | 33 | 4 (pcm, kon, swc, nyn) | 40 |
| Microsoft Translator (text translation, Cloud) | 20 | 1 (swc) | 56 |

Google explicit (33): yor hau ibo twi ewe swh kin som orm amh tir lug lin zul xho afr sna nya sot tsn mlg bem run tso nso ssw nbl sag luo bam nus ach kri.
Microsoft explicit (20): yor hau ibo swh kin som amh tir lug lin zul xho afr sna nya sot tsn mlg run nso.
Google-only (explicit, not MS): twi ewe orm bem tso ssw nbl sag luo bam nus ach kri (13). Microsoft-only: none (all 20 MS languages are also on Google's list).

Supported by NEITHER (no explicit and no proxy in either system) - 40 languages:
efi fon wol ibb urh tum ven tiv umb kmb mos kik kam bin dag oke shi dyu kbp bbj kea kab lua kqn lun iso loz wal tsc vmw kua seh her yao sid ndc ish dua bas wes

(These 40 are exactly Google's "no" set; Microsoft supports none of them. Languages with proxy-only coverage by at least one system and no explicit support anywhere: pcm, kon, swc, nyn.)

## Sources and caveats

- Google: https://cloud.google.com/translate/docs/languages (301-redirects to https://docs.cloud.google.com/translate/docs/languages), retrieved 2026-10-20. Page has Translation LLM, NMT, romanization, custom-model and variant tables; it has NO separate "latest additions" table, only a note that the NMT list is updated as languages are added. The page was read via the WebFetch tool, which returns a model-generated extraction of the page, not raw HTML; the list was not re-checked against the raw page. The page does not show a "last updated" date in the extraction, so currency of the list relative to 2026-10-20 is not independently confirmed. Languages absent from the list (e.g. Fon, Wolof, Dyula, Tamazight/Tachelhit, Venda, Kikuyu) were absent in this extraction; I am relying on that extraction.
- Microsoft: https://learn.microsoft.com/en-us/azure/ai-services/translator/language-support, retrieved 2026-10-20. Page metadata: ms.date 2026-06-06, updated_at 2026-08-23. Text translation table ("Cloud - Text translation and Document translation" column) used. No African language is flagged preview on this page. Document-translation scanned-PDF table is a different feature and was not used.
- Language-code look-alikes: Google "Twi (Akan)" = ak (counted as Twi match); Google "Ganda (Luganda)" = lg, MS uses lug; Google "Oromo" = om (MS has none); Google "Dombe" (dov) is not Venda; Google "Tsonga" (ts) is not Tswa; Google "Luo" and "Alur" (alz) and "Acholi" (ach) are separate entries; Google "Kiga" (cgg) is proxy only for Nyankore; Google "Krio" (kri) exact for Krio.
- Quality caveat for the benchmark: presence on a list does not imply comparable quality; Google's Translation LLM table marks am, ha, ig, so as Experimental.
- Note: "unverified" was not needed; every language was checked against the extracted lists.
