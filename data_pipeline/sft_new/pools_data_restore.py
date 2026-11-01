"""Instruction-template banks for the two text-restoration tasks (owner 2026-10-15): `diacritize` (restore tone marks / special letters) and
`denoise` (<clean>: repair typos and stray characters). The combined task joins one template of each.  18 instruction languages like the other banks.

Construction (same safety rule as pools_data_d): the request first, then a LABEL for the language (`Taal: {target}.`), so a language name never has to
take a case ending or a class prefix.  English, French, Afrikaans and Swahili are written freely; yor / hau / ibo / pcm extend the older ext_pools
banks.  The wording of the low-tier languages should be reviewed by a native speaker before it is trusted (the tier is recorded in every record).
"""
TIER = {"zul": "medium", "xho": "low", "sna": "low", "kin": "low", "som": "low", "amh": "medium", "orm": "low", "tsn": "low", "sot": "low", "nya": "low",
        "swh": "medium", "afr": "medium", "fra": "high"}

# (diacritize pre, diacritize post, denoise pre, denoise post, label word)
_L = {
    "zul": (["Engeza ama-diacritics alahlekile kulo mbhalo.", "Buyisela ama-diacritics kulo mbhalo olandelayo.", "Bhala kabusha lo mbhalo unawo wonke ama-diacritics."],
            ["Engeza ama-diacritics alahlekile embhalweni ongenhla.", "Buyisela ama-diacritics embhalweni ongenhla."],
            ["Hlanza lo mbhalo onomsindo ubuyisele owangempela.", "Lungisa amaphutha nezinhlamvu ezingafuneki kulo mbhalo olandelayo.", "Susa umsindo kulo mbhalo."],
            ["Hlanza umbhalo ongenhla ubuyisele owangempela.", "Lungisa amaphutha embhalweni ongenhla."], "Ulimi"),
    "xho": (["Yongeza ii-diacritics ezilahlekileyo kulo mbhalo.", "Buyisela ii-diacritics kulo mbhalo ulandelayo.", "Bhala kwakhona lo mbhalo unazo zonke ii-diacritics."],
            ["Yongeza ii-diacritics ezilahlekileyo kumbhalo ongentla.", "Buyisela ii-diacritics kumbhalo ongentla."],
            ["Coca lo mbhalo onengxolo ubuyisele owokuqala.", "Lungisa iimpazamo neempawu ezingafunekiyo kulo mbhalo ulandelayo.", "Susa ingxolo kulo mbhalo."],
            ["Coca umbhalo ongentla ubuyisele owokuqala.", "Lungisa iimpazamo kumbhalo ongentla."], "Ulwimi"),
    "sna": (["Wedzera ma-diacritics akashaikwa mukunyora uku.", "Dzorera ma-diacritics mukunyora uku kunotevera.", "Nyora zvakare kunyora uku kune ma-diacritics ese."],
            ["Wedzera ma-diacritics akashaikwa mukunyora kuri pamusoro.", "Dzorera ma-diacritics mukunyora kuri pamusoro."],
            ["Chenesa kunyora uku kune ruzha udzorere chekutanga.", "Gadzirisa zvikanganiso nezvimwe zvisina basa mukunyora uku kunotevera.", "Bvisa ruzha mukunyora uku."],
            ["Chenesa kunyora kuri pamusoro udzorere chekutanga.", "Gadzirisa zvikanganiso mukunyora kuri pamusoro."], "Mutauro"),
    "kin": (["Ongeraho ibimenyetso (diacritics) byabuze muri iyi nyandiko.", "Subiza ibimenyetso (diacritics) muri iyi nyandiko ikurikira.", "Ongera wandike iyi nyandiko ufite ibimenyetso byose."],
            ["Ongeraho ibimenyetso byabuze mu nyandiko iri hejuru.", "Subiza ibimenyetso (diacritics) mu nyandiko iri hejuru."],
            ["Sukura iyi nyandiko irimo urusaku usubize iy'umwimerere.", "Kosora amakosa n'inyuguti zidakenewe muri iyi nyandiko ikurikira.", "Kuraho urusaku muri iyi nyandiko."],
            ["Sukura inyandiko iri hejuru usubize iy'umwimerere.", "Kosora amakosa mu nyandiko iri hejuru."], "Ururimi"),
    "som": (["Ku dar calaamadaha (diacritics) maqan qoraalkan.", "Dib u celi calaamadaha (diacritics) qoraalka soo socda.", "Dib u qor qoraalkan adigoo ku daraya dhammaan calaamadaha."],
            ["Ku dar calaamadaha maqan qoraalka kor ku xusan.", "Dib u celi calaamadaha (diacritics) qoraalka kor ku yaal."],
            ["Nadiifi qoraalkan buuqsan oo soo celi kii asalka ahaa.", "Sax khaladaadka iyo xarfaha aan loo baahnayn ee qoraalka soo socda.", "Ka saar buuqa qoraalkan."],
            ["Nadiifi qoraalka kor ku xusan oo soo celi kii asalka ahaa.", "Sax khaladaadka qoraalka kor ku yaal."], "Luqadda"),
    "amh": (["በዚህ ጽሑፍ ላይ የጠፉ ምልክቶችን (diacritics) ጨምር።", "በሚከተለው ጽሑፍ ውስጥ ምልክቶችን (diacritics) መልስ።", "ይህን ጽሑፍ ሁሉንም ምልክቶች በማስገባት እንደገና ጻፍ።"],
            ["ከላይ ባለው ጽሑፍ ላይ የጠፉ ምልክቶችን ጨምር።", "ከላይ ባለው ጽሑፍ ምልክቶችን (diacritics) መልስ።"],
            ["ይህን ጫጫታ ያለበት ጽሑፍ አጽዳና የመጀመሪያውን መልስ።", "በሚከተለው ጽሑፍ ውስጥ ያሉ ስህተቶችንና አላስፈላጊ ፊደላትን አስተካክል።", "ከዚህ ጽሑፍ ጫጫታውን አስወግድ።"],
            ["ከላይ ያለውን ጽሑፍ አጽዳና የመጀመሪያውን መልስ።", "ከላይ ባለው ጽሑፍ ያሉ ስህተቶችን አስተካክል።"], "ቋንቋ"),
    "orm": (["Mallattoolee (diacritics) hir'atan barreeffama kana irratti dabali.", "Mallattoolee (diacritics) barreeffama kana keessatti deebisi.", "Barreeffama kana mallattoolee hundumaan deebi'ii barreessi."],
            ["Mallattoolee hir'atan barreeffama armaan olii irratti dabali.", "Mallattoolee (diacritics) barreeffama armaan olii keessatti deebisi."],
            ["Barreeffama sagalee qabu kana qulqulleessi, kan jalqabaa deebisi.", "Dogoggora fi qubeewwan barbaachisan hin taane barreeffama kana keessatti sirreessi.", "Sagalee barreeffama kana keessaa balleessi."],
            ["Barreeffama armaan olii qulqulleessi kan jalqabaa deebisi.", "Dogoggora barreeffama armaan olii sirreessi."], "Afaan"),
    "tsn": (["Okeletsa matshwao (diacritics) a a haelang mo mokwalong ono.", "Busetsa matshwao (diacritics) mo mokwalong o o latelang.", "Kwala mokwalo ono gape ka matshwao otlhe."],
            ["Okeletsa matshwao a a haelang mo mokwalong o o fa godimo.", "Busetsa matshwao (diacritics) mo mokwalong o o fa godimo."],
            ["Phepafatsa mokwalo ono o o nang le modumo, o busetse wa ntlha.", "Baakanya diphoso le ditlhaka tse di sa tlhokegeng mo mokwalong o o latelang.", "Tlosa modumo mo mokwalong ono."],
            ["Phepafatsa mokwalo o o fa godimo, o busetse wa ntlha.", "Baakanya diphoso mo mokwalong o o fa godimo."], "Puo"),
    "sot": (["Eketsa matšoao (diacritics) a haellang temaneng ena.", "Khutlisetsa matšoao (diacritics) temaneng e latelang.", "Ngola temana ena hape ka matšoao ohle."],
            ["Eketsa matšoao a haellang temaneng e kaholimo.", "Khutlisetsa matšoao (diacritics) temaneng e kaholimo."],
            ["Hloekisa temana ena e nang le lerata, u khutlisetse ea pele.", "Lokisa diphoso le litlhaku tse sa hlokahaleng temaneng e latelang.", "Tlosa lerata temaneng ena."],
            ["Hloekisa temana e kaholimo, u khutlisetse ea pele.", "Lokisa diphoso temaneng e kaholimo."], "Puo"),
    "nya": (["Onjezani zizindikiro (diacritics) zosowa m'malemba awa.", "Bwezerani zizindikiro (diacritics) m'malemba otsatirawa.", "Lembaninso malemba awa ndi zizindikiro zonse."],
            ["Onjezani zizindikiro zosowa m'malemba omwe ali pamwambapa.", "Bwezerani zizindikiro (diacritics) m'malemba omwe ali pamwambapa."],
            ["Yeretsani malemba awa omwe ali ndi phokoso ndi kubwezeretsa oyambirira.", "Konzani zolakwika ndi zilembo zosafunikira m'malemba otsatirawa.", "Chotsani phokoso m'malemba awa."],
            ["Yeretsani malemba omwe ali pamwambapa ndi kubwezeretsa oyambirira.", "Konzani zolakwika m'malemba omwe ali pamwambapa."], "Chilankhulo"),
    "swh": (["Ongeza alama za herufi (diacritics) zilizokosekana kwenye maandishi haya.", "Rudisha alama (diacritics) katika maandishi yafuatayo.", "Andika upya maandishi haya yakiwa na alama zote."],
            ["Ongeza alama zilizokosekana kwenye maandishi yaliyo hapo juu.", "Rudisha alama (diacritics) kwenye maandishi yaliyo hapo juu."],
            ["Safisha maandishi haya yenye kelele na urudishe ya asili.", "Sahihisha makosa na herufi zisizohitajika katika maandishi yafuatayo.", "Ondoa kelele kwenye maandishi haya."],
            ["Safisha maandishi yaliyo hapo juu na urudishe ya asili.", "Sahihisha makosa katika maandishi yaliyo hapo juu."], "Lugha"),
    "afr": (["Voeg die ontbrekende diakritiese tekens by hierdie teks.", "Herstel die diakritiese tekens in die volgende teks.", "Skryf hierdie teks oor met al sy diakritiese tekens en spesiale letters."],
            ["Voeg die ontbrekende diakritiese tekens by die teks hierbo.", "Herstel die diakritiese tekens van die teks hierbo."],
            ["Maak hierdie raserige teks skoon en gee die oorspronklike terug.", "Korrigeer die tikfoute en vreemde karakters in die volgende teks.", "Verwyder die geraas uit hierdie teks."],
            ["Maak die teks hierbo skoon en gee die oorspronklike terug.", "Korrigeer die tikfoute in die teks hierbo."], "Taal"),
    "fra": (["Ajoute les diacritiques manquants à ce texte.", "Rétablis les accents et signes diacritiques dans le texte suivant.", "Réécris ce texte avec tous ses diacritiques et lettres spéciales."],
            ["Ajoute les diacritiques manquants au texte ci-dessus.", "Rétablis les signes diacritiques du texte ci-dessus."],
            ["Nettoie ce texte bruité et redonne l'original.", "Corrige les fautes de frappe et les caractères parasites du texte suivant.", "Supprime le bruit de ce texte."],
            ["Nettoie le texte ci-dessus et redonne l'original.", "Corrige les fautes du texte ci-dessus."], "Langue"),
}


def _labelled(items, word):
    return [f"{t} {word} : {{target}}." if word == "Langue" else f"{t} {word}: {{target}}." for t in items]


BANKS = {lang: {"diacritize": {"pre": _labelled(d_pre, w), "post": _labelled(d_post, w)},
                "denoise": {"pre": _labelled(n_pre, w), "post": _labelled(n_post, w)}}
         for lang, (d_pre, d_post, n_pre, n_post, w) in _L.items()}

# English is written freely (the language name is natural there) and extends ext_pools; so do the four languages of the older banks
BANKS["eng"] = {
    "diacritize": {"pre": ["Add the missing diacritics and tone marks to this {target} text:", "Restore the diacritics in the following {target} sentence:",
                           "Rewrite this {target} text with all its diacritics and special letters in place:", "This {target} text was typed without diacritics. Write it correctly:",
                           "Put the tone marks and special letters back in this {target} text:"],
                   "post": ["Add the correct diacritics to the {target} text above.", "Restore the tone marks and special letters of the text above.",
                            "Rewrite the {target} text above with its proper diacritics."]},
    "denoise": {"pre": ["Clean up this noisy {target} text and give back the original sentence:", "The following {target} text has typos and stray characters. Correct it:",
                        "Remove the noise from this {target} text and write it correctly:", "Fix the corrupted characters in this {target} text:",
                        "Restore the original {target} text from this damaged version:"],
                "post": ["Clean the {target} text above and restore the original.", "Correct the typos and noise in the text above.", "Repair the damaged {target} text above."]}}
BANKS["yor"] = {
    "diacritize": {"pre": ["Ṣàfikún àwọn àmì ohùn tó sọnù sí ọ̀rọ̀ {target} yìí:", "Kọ ọ̀rọ̀ {target} yìí padà pẹ̀lú gbogbo àmì ohùn àti àmì abẹ́ rẹ̀:"],
                   "post": ["Ṣàfikún àwọn àmì ohùn tó yẹ sí ọ̀rọ̀ {target} tó wà lókè."]},
    "denoise": {"pre": ["Fọ ọ̀rọ̀ {target} onígbọ̀ngbọ̀n yìí mọ́, kí o sì dá gbólóhùn àkọ́kọ́ padà:", "Ọ̀rọ̀ {target} yìí ní àṣìṣe ìtẹ̀wé àti àwọn lẹ́tà tí kò yẹ. Ṣàtúnṣe rẹ̀:"],
                "post": ["Fọ ọ̀rọ̀ {target} tó wà lókè mọ́ kí o sì dá èyí àkọ́kọ́ padà."]}}
BANKS["hau"] = {
    "diacritize": {"pre": ["Ƙara alamomin da suka ɓace a wannan matanin {target}:", "Sake rubuta wannan matanin {target} da dukkan alamominsa:"],
                   "post": ["Ƙara alamomin da suka dace a matanin {target} da ke sama."]},
    "denoise": {"pre": ["Tsabtace wannan matanin {target} mai hayaniya ka dawo da jimlar asali:", "Wannan matanin {target} yana da kurakuran rubutu da haruffan da ba su dace ba. Gyara shi:"],
                "post": ["Tsabtace matanin {target} da ke sama ka dawo da na asali."]}}
BANKS["ibo"] = {
    "diacritize": {"pre": ["Tinye akara ndị na-efu n'ederede {target} a:", "Dee ederede {target} a ọzọ ya nwere akara niile ya kwesịrị inwe:"],
                   "post": ["Tinye akara ndị kwesịrị ekwesị n'ederede {target} dị n'elu."]},
    "denoise": {"pre": ["Hichaa ederede {target} a nwere mkpọtụ ma weghachite ahịrịokwu mbụ:", "Ederede {target} a nwere njehie ederede na mkpụrụedemede ndị na-ekwesịghị. Mezie ya:"],
                "post": ["Hichaa ederede {target} dị n'elu ma weghachite nke mbụ."]}}
BANKS["pcm"] = {
    "diacritize": {"pre": ["Add di tone mark and special letters wey miss for dis {target} text:", "Write dis {target} text again wit all im diacritics:"],
                   "post": ["Add di correct diacritics to di {target} text wey dey up there."]},
    "denoise": {"pre": ["Clean dis {target} text wey get typos and wahala letters, bring back di original sentence:", "Dis {target} text get spelling mistake and strange character. Correct am:"],
                "post": ["Clean di {target} text wey dey up there make e be like di original."]}}
