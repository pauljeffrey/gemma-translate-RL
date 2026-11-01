"""Instruction-template banks: isiZulu (zul), isiXhosa (xho), Amharic (amh).  See pools_multi.py for the contract.

zul / xho: language names are `isi...`; the prefix forms {target.nge} (ngesiZulu), {target.yi} (yisiZulu), {target.lwe}, {target.ku} (kusiZulu, zul) and
{target.kwi} (kwisiZulu, xho) are produced by africomet_translation.instructions._NguniName, so no template needs to know which language is named.
"""
TIER = {"zul": "medium", "xho": "low", "amh": "medium"}

NAMES = {
    "zul": {
        "eng": ["isiNgisi"], "fra": ["isiFulentshi"], "swh": ["isiSwahili"], "hau": ["isiHausa"], "yor": ["isiYoruba"], "ibo": ["isiIgbo"], "zul": ["isiZulu"],
        "xho": ["isiXhosa"], "afr": ["isiBhunu", "isiAfrikaans"], "amh": ["isiAmharic"], "som": ["isiSomali"], "kin": ["isiKinyarwanda"], "sna": ["isiShona"],
        "orm": ["isiOromo"], "tsn": ["isiTswana"], "sot": ["isiSuthu"], "nya": ["isiChichewa"], "wol": ["isiWolof"], "lin": ["isiLingala"], "lug": ["isiLuganda"],
        "ti": ["isiTigrinya"], "mlg": ["isiMalagasy"], "pcm": ["isiPidgin"], "twi": ["isiTwi"], "ewe": ["isiEwe"], "fon": ["isiFon"], "efi": ["isiEfik"],
        "nso": ["isiPedi"], "tso": ["isiTsonga"], "ven": ["isiVenda"], "ssw": ["isiSwati"], "nbl": ["isiNdebele"],
    },
    "xho": {
        "eng": ["isiNgesi"], "fra": ["isiFrentshi"], "swh": ["isiSwahili"], "hau": ["isiHausa"], "yor": ["isiYoruba"], "ibo": ["isiIgbo"], "zul": ["isiZulu"],
        "xho": ["isiXhosa"], "afr": ["isiBhulu", "isiAfrikaans"], "amh": ["isiAmharic"], "som": ["isiSomali"], "kin": ["isiKinyarwanda"], "sna": ["isiShona"],
        "orm": ["isiOromo"], "tsn": ["isiTswana"], "sot": ["isiSuthu"], "nya": ["isiChichewa"], "wol": ["isiWolof"], "lin": ["isiLingala"], "lug": ["isiLuganda"],
        "ti": ["isiTigrinya"], "mlg": ["isiMalagasy"], "pcm": ["isiPidgin"], "twi": ["isiTwi"], "ewe": ["isiEwe"], "fon": ["isiFon"], "efi": ["isiEfik"],
        "nso": ["isiPedi"], "tso": ["isiTsonga"], "ven": ["isiVenda"], "ssw": ["isiSwati"], "nbl": ["isiNdebele"],
    },
    "amh": {
        "eng": ["እንግሊዝኛ"], "fra": ["ፈረንሳይኛ"], "swh": ["ስዋሂሊ"], "hau": ["ሃውሳ"], "yor": ["ዮሩባ"], "ibo": ["ኢግቦ"], "zul": ["ዙሉ"], "xho": ["ኮሳ"],
        "afr": ["አፍሪካንስ"], "amh": ["አማርኛ"], "som": ["ሶማሊኛ"], "kin": ["ኪንያርዋንዳ"], "sna": ["ሾና"], "orm": ["ኦሮምኛ"], "tsn": ["ጽዋና"],
        "sot": ["ሶቶ"], "nya": ["ቺቼዋ"], "wol": ["ዎሎፍ"], "lin": ["ሊንጋላ"], "lug": ["ሉጋንዳ"], "ti": ["ትግርኛ"], "mlg": ["ማላጋሲ"],
        "pcm": ["ናይጄሪያ ፒጂን"], "twi": ["ትዊ"], "ewe": ["ኤዌ"], "bem": ["ቤምባ"], "kik": ["ኪኩዩ"], "luo": ["ሉኦ"],
    },
}

BANKS = {
    # ------------------------------------------------------------------------------------------------ isiZulu
    "zul": {
        "translate": {
            "pre": [
                "Humusha umbhalo olandelayo {target.nge}.",
                "Sicela uhumushe isiqephu esingezansi {target.nge}.",
                "Humusha lo mbhalo ube {target.yi}, ngendlela yemvelo hhayi igama negama.",
                "Umbhalo ongezansi ubhalwe {source.nge}. Ngicela ungiphe inguqulo yawo {target.nge} kuphela, ngaphandle kwezincazelo.",
                "Kusuka {source.ku} kuya {target.ku}:",
                "Wena ungumhumushi onekhono. Humusha isiqephu esingezansi {target.nge}, ugcine amagama, izinombolo nezimpawu zokubhala. Nikeza inguqulo kuphela.",
                "Ngicela ungibhalele inguqulo ethembekile {target.nge} yalokhu okulandelayo.",
                "Humusha {target.nge}, ushiye amagama abantu nawezindawo njengoba enjalo:",
            ],
            "post": [
                "Humusha umbhalo ongenhla {target.nge}.",
                "Sicela uhumushe isiqephu esingenhla {target.nge}.",
                "Ungakwazi yini ukuhumusha konke lokhu {target.nge}?",
                "Umbhalo ongenhla ubhalwe {source.nge}. Wuhumushe {target.nge}.",
                "Ngipha inguqulo {target.nge} yalokho okungenhla, ngaphandle kokuphawula.",
            ],
            "any": [
                "Humusha lokhu {target.nge}.",
                "Ngidinga inguqulo yalo mbhalo {target.nge}.",
                "Yisho lokhu {target.nge}.",
                "Humusha {target.nge}.",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "Fingqa umbhalo olandelayo.",
                "Fingqa lokhu {target.nge}:",
                "Ngipha isifinyezo esifushane sombhalo ongezansi.",
                "Sicela ubhale isifinyezo esifushane sendatshana elandelayo {target.nge}.",
                "Funda isiqephu esingezansi bese usifingqa ngemisho emibili noma emithathu.",
                "Uyini umcabango oyinhloko wombhalo ongezansi? Wufingqe {target.nge}.",
            ],
            "post": [
                "Fingqa umbhalo ongenhla.",
                "Ngipha isifinyezo esifushane salokho okungenhla {target.nge}.",
                "Sicela ufingqe lokhu {target.nge}.",
                "Ungakwazi yini ukufingqa lokhu ngemisho embalwa?",
                "Yiziphi izinto eziyinhloko? Phendula ngesifinyezo {target.nge}.",
            ],
            "any": ["Fingqa {target.nge}.", "Isifinyezo {target.nge}, ngiyacela."],
        },
        "headline": {
            "pre": [
                "Bhala isihloko sendatshana elandelayo.",
                "Nikeza lezi zindaba isihloko {target.nge}:",
                "Iyiphi inhloko-ndaba efanele yombhalo ongezansi? Phendula {target.nge}.",
            ],
            "post": ["Bhala isihloko sombhalo ongenhla.", "Nika indatshana engenhla isihloko {target.nge}."],
            "any": ["Isihloko {target.nge}, ngiyacela."],
        },
        "sentiment": {
            "pre": [
                "Uwuphi umuzwa (sentiment) wombhalo olandelayo? Phendula nge-Positive, Negative noma Neutral.",
                "Hlaziya imizwa yalo musho:",
                "Ingabe lo musho olandelayo unomuzwa omuhle, omubi noma ophakathi nendawo? Phendula nge-Positive, Negative noma Neutral.",
                "Yenza uhlaziyo lwemizwa yalo mbhalo:",
            ],
            "post": ["Imizwa yombhalo ongenhla iyini?", "Ingabe lokhu kuphakathi nendawo, kuhle noma kubi? Phendula nge-Positive, Negative noma Neutral."],
        },
        "topic": {
            "pre": [
                "Yisiphi isihloko (topic) salesi sihloko sezindaba?",
                "Beka lesi sihloko sezindaba esigabeni sesihloko esifanele:",
                "Umbhalo olandelayo ungowesigaba sini?",
                "Khetha ilebuli yesihloko efanele kakhulu kulo mbhalo:",
            ],
            "post": ["Umbhalo ongenhla ungowesiphi isihloko?", "Lesi yisigaba sini?"],
        },
        "emotion": {
            "pre": [
                "Lo mbhalo ubonisa muphi umuzwa? Khetha phakathi kwalokhu: {labels}.",
                "Hlukanisa umuzwa osembhalweni olandelayo ({labels}).",
                "Umbhali uzizwa kanjani? Phendula ngomunye wale mizwa: {labels}.",
            ],
            "post": ["Muphi umuzwa obonisiwe ngenhla? Khetha phakathi kwalokhu: {labels}."],
        },
        "language_id": {
            "pre": [
                "Lo mbhalo olandelayo ubhalwe ngaluphi ulimi?",
                "Khomba ulimi lwalo mbhalo:",
                "Luluphi lolu ulimi? Phendula ngegama lolimi.",
                "Nikeza igama lolimi olusetshenziswe embhalweni ongezansi.",
            ],
            "post": ["Umbhalo ongenhla ubhalwe ngaluphi ulimi?", "Luluphi lolo ulimi?"],
        },
        "qa_context": {
            "pre": [
                "Phendula umbuzo usebenzisa kuphela isiqephu esingezansi.",
                "Funda isiqephu bese uphendula umbuzo olandelayo.",
                "Sebenzisa umongo ukuze uphendule:",
            ],
            "post": ["Phendula umbuzo usebenzisa kuphela umongo ongenhla.", "Ngokuya ngesiqephu esingenhla, yini impendulo?"],
        },
        "qa_open": {
            "pre": [
                "Phendula umbuzo olandelayo {target.nge}.",
                "Sicela uphendule lo mbuzo {target.nge}:",
                "Nikeza impendulo emfushane nefanele {target.nge}:",
            ],
            "post": ["Phendula umbuzo ongenhla {target.nge}.", "Sicela uphendule lokho {target.nge}."],
        },
        "answer_in": {
            "post": [
                "Nikeza impendulo yakho {target.nge}.",
                "Phendula {target.nge} kuphela.",
                "Sicela uphendule {target.nge}.",
                "Bhala impendulo yakho {target.nge}.",
                "Impendulo yakho kumele ibe {target.nge}.",
            ],
        },
        "math": {
            "pre": [
                "Xazulula inkinga elandelayo isinyathelo ngesinyathelo, bese unikeza impendulo yokugcina.",
                "Sebenza le nkinga yezibalo bese unikeza impendulo yokugcina.",
                "Xazulula:",
                "Khombisa ukucabanga kwakho bese unikeza impendulo {target.nge}.",
            ],
            "post": ["Xazulula inkinga engenhla isinyathelo ngesinyathelo.", "Khombisa izinyathelo zakho unikeze impendulo yokugcina."],
        },
        "mcq": {
            "pre": [
                "Phendula umbuzo onezinketho eziningi ongezansi ngohlamvu lwenketho elungile.",
                "Khetha impendulo eyiyo (A, B, C noma D):",
                "Khetha inketho efanele yombuzo olandelayo.",
            ],
            "post": ["Iyiphi inketho eyiyo?", "Phendula nge-A, B, C noma D."],
        },
        "nli": {
            "pre": [
                "Nquma ukuthi umusho wesibili uhlobene kanjani nowokuqala. Phendula nge-entailment, neutral noma contradiction.",
                "Ukuphetha ngolimi lwemvelo: nikeza lo mbhangqwana wemisho ilebuli ethi entailment, neutral noma contradiction.",
            ],
            "post": ["Buyini ubudlelwano phakathi kwale misho emibili? Phendula nge-entailment, neutral noma contradiction."],
        },
        "ner": {
            "pre": [
                "Phawula amagama athile (abantu, izindawo, izinhlangano, izinsuku) kulo musho. Bhala igama ngalinye kulandele i-<tag> nelebuli yalo (PER, ORG, LOC, DATE noma O).",
            ],
        },
        "pos": {
            "pre": ["Nikeza ilebuli yohlobo lwegama ngalinye kulo musho njengamapheya egama <tag> ILEBULI."],
            "post": ["Nikeza uhlobo lwegama ngalinye emshweni ongenhla (igama <tag> ILEBULI)."],
        },
        "define_term": {
            "pre": [
                "Igama elithi \"{title}\" lisho ukuthini? Chaza {target.nge}.",
                "Nikeza incazelo yegama elithi \"{title}\" {target.nge}.",
                "Chaza \"{title}\" ({target.nge}).",
            ],
        },
        "wiki_ask": {
            "pre": [
                "Ngiyafuna ukwazi ngalokhu: {title}. Phendula {target.nge} kuphela.",
            ],
        },
    },
    # ------------------------------------------------------------------------------------------------ isiXhosa
    "xho": {
        "translate": {
            "pre": [
                "Guqulela umbhalo olandelayo {target.kwi}.",
                "Nceda uguqulele isiqendu esingezantsi {target.kwi}.",
                "Ndicela uguqulele lo mbhalo {target.kwi}, ngendlela yendalo hayi igama negama.",
                "Umbhalo ongezantsi ubhaliwe {source.nge}. Ndipha inguqulelo yawo {target.kwi} kuphela, ngaphandle kwenkcazelo.",
                "Ukusuka {source.kwi} ukuya {target.kwi}:",
                "Ungumguquleli onobuchule. Guqulela isiqendu esingezantsi {target.kwi}, ugcine amagama, amanani neempawu zokubhala. Nika inguqulelo kuphela.",
                "Guqulela {target.kwi}, ushiye amagama abantu neendawo njengoko injalo:",
            ],
            "post": [
                "Guqulela umbhalo ongentla {target.kwi}.",
                "Nceda uguqulele isiqendu esingentla {target.kwi}.",
                "Ungakwazi na ukuguqulela konke oku {target.kwi}?",
                "Umbhalo ongentla ubhaliwe {source.nge}. Wuguqulele {target.kwi}.",
            ],
            "any": [
                "Guqulela oku {target.kwi}.",
                "Ndifuna inguqulelo yalo mbhalo {target.kwi}.",
                "Yitsho le nto {target.nge}.",
                "Guqulela {target.kwi}.",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "Shwankathela umbhalo olandelayo.",
                "Shwankathela oku {target.nge}:",
                "Ndipha isishwankathelo esifutshane sombhalo ongezantsi.",
                "Nceda ubhale isishwankathelo esifutshane senqaku elandelayo {target.nge}.",
                "Funda isiqendu esingezantsi uze usishwankathele ngezivakalisi ezimbini okanye ezintathu.",
            ],
            "post": [
                "Shwankathela umbhalo ongentla.",
                "Ndipha isishwankathelo esifutshane sombhalo ongentla {target.nge}.",
                "Nceda ushwankathele oku {target.nge}.",
                "Ungakwazi na ukushwankathela oku ngezivakalisi ezimbalwa?",
            ],
            "any": ["Shwankathela {target.nge}.", "Isishwankathelo {target.nge}, ndicela."],
        },
        "headline": {
            "pre": [
                "Bhala isihloko senqaku elandelayo.",
                "Nika le ndaba isihloko {target.nge}:",
            ],
            "post": ["Bhala isihloko sombhalo ongentla.", "Nika inqaku elingentla isihloko {target.nge}."],
            "any": ["Isihloko {target.nge}, ndicela."],
        },
        "sentiment": {
            "pre": [
                "Yeyiphi imvakalelo (sentiment) yombhalo olandelayo? Phendula nge-Positive, Negative okanye Neutral.",
                "Hlalutya imvakalelo yesi sivakalisi:",
                "Esi sivakalisi silandelayo sinemvakalelo entle, embi okanye ephakathi? Phendula nge-Positive, Negative okanye Neutral.",
            ],
            "post": ["Yeyiphi imvakalelo yombhalo ongentla?"],
        },
        "topic": {
            "pre": [
                "Umxholo wesi sihloko sendaba yintoni?",
                "Beka esi sihloko sendaba kwicandelo lomxholo:",
                "Lo mbhalo olandelayo ungowecandelo elinjani?",
            ],
            "post": ["Umbhalo ongentla ubhekisele kumxholo onjani?", "Eli yicandelo elinjani?"],
        },
        "emotion": {
            "pre": [
                "Lo mbhalo ubonisa imvakalelo eyiphi? Khetha phakathi kwezi: {labels}.",
                "Hlela imvakalelo ekumbhalo olandelayo ({labels}).",
                "Umbhali uvakalelwa yintoni? Phendula ngenye kwezi: {labels}.",
            ],
            "post": ["Yeyiphi imvakalelo ebonisiweyo ngentla? Khetha phakathi kwezi: {labels}."],
        },
        "language_id": {
            "pre": [
                "Lo mbhalo olandelayo ubhalwe ngolwimi luni?",
                "Chonga ulwimi lwalo mbhalo:",
                "Lulwimi luni olu? Phendula ngegama lolwimi.",
            ],
            "post": ["Umbhalo ongentla ubhalwe ngolwimi luni?", "Lulwimi luni lolo?"],
        },
        "qa_context": {
            "pre": [
                "Phendula umbuzo usebenzisa kuphela isiqendu esingezantsi.",
                "Funda isiqendu uze uphendule umbuzo olandelayo.",
                "Sebenzisa imeko ukuphendula:",
            ],
            "post": ["Phendula umbuzo usebenzisa kuphela imeko engentla.", "Ngokwesiqendu esingentla, yeyiphi impendulo?"],
        },
        "qa_open": {
            "pre": [
                "Phendula lo mbuzo ulandelayo {target.nge}.",
                "Nceda uphendule lo mbuzo {target.nge}:",
                "Nika impendulo emfutshane echanekileyo {target.nge}:",
            ],
            "post": ["Phendula umbuzo ongentla {target.nge}.", "Nceda uphendule oko {target.nge}."],
        },
        "answer_in": {
            "post": [
                "Nika impendulo yakho {target.nge}.",
                "Phendula {target.nge} kuphela.",
                "Nceda uphendule {target.nge}.",
                "Bhala impendulo yakho {target.nge}.",
                "Impendulo yakho mayibe {target.nge}.",
            ],
        },
        "math": {
            "pre": [
                "Combulula ingxaki elandelayo inyathelo ngenyathelo, uze unike impendulo yokugqibela.",
                "Sebenza le ngxaki yemathematika uze unike impendulo yokugqibela.",
                "Combulula:",
            ],
            "post": ["Combulula ingxaki engentla inyathelo ngenyathelo.", "Bonisa amanyathelo akho uze unike impendulo yokugqibela."],
        },
        "mcq": {
            "pre": [
                "Phendula umbuzo onokhetho oluninzi ongezantsi ngonobumba wokhetho olululo.",
                "Khetha impendulo echanekileyo (A, B, C okanye D):",
                "Khetha ukhetho olufanelekileyo kulo mbuzo ulandelayo.",
            ],
            "post": ["Luluphi ukhetho olululo?", "Phendula nge-A, B, C okanye D."],
        },
        "nli": {
            "pre": [
                "Misela ukuba isivakalisi sesibini sinxulumene njani nesokuqala. Phendula nge-entailment, neutral okanye contradiction.",
            ],
            "post": ["Yintoni unxulumano phakathi kwezi zivakalisi zimbini? Phendula nge-entailment, neutral okanye contradiction."],
        },
        "ner": {
            "pre": [
                "Phawula amagama athile (abantu, iindawo, imibutho, imihla) kwesi sivakalisi. Bhala igama ngalinye kulandele i-<tag> kunye nelebhile yalo (PER, ORG, LOC, DATE okanye O).",
            ],
        },
        "pos": {
            "pre": ["Nika ilebhile yohlobo lwegama ngalinye kwesi sivakalisi njengezibini zegama <tag> ILEBHILE."],
        },
        "define_term": {
            "pre": [
                "Igama elithi \"{title}\" lithetha ntoni? Cacisa {target.nge}.",
                "Nika intsingiselo yegama elithi \"{title}\" {target.nge}.",
            ],
        },
    },
    # ------------------------------------------------------------------------------------------------ Amharic
    "amh": {
        "translate": {
            "pre": [
                "የሚከተለውን ጽሑፍ ወደ {target} ተርጉም።",
                "እባክዎ ከታች ያለውን አንቀጽ ወደ {target} ይተርጉሙ።",
                "ይህን ጽሑፍ ወደ {target} እንድትተረጉምልኝ እፈልጋለሁ፤ ቃል በቃል ሳይሆን ተፈጥሯዊ እንዲሆን።",
                "ከታች ያለው ጽሑፍ በ{source} የተጻፈ ነው። ትርጉሙን ያለ ተጨማሪ ማብራሪያ በ{target} ብቻ ስጠኝ።",
                "ከ{source} ወደ {target}፦",
                "አንተ ባለሙያ ተርጓሚ ነህ። ከታች ያለውን የ{source} አንቀጽ ወደ {target} ተርጉም፤ ስሞችን፣ ቁጥሮችንና ሥርዓተ ነጥቦችን ጠብቅ። ትርጉሙን ብቻ ስጥ።",
                "የሚከተለውን ወደ {target} ቀይረው፦",
                "ለሚከተለው ጽሑፍ ታማኝ የሆነ የ{target} ትርጉም ጻፍ።",
            ],
            "post": [
                "ከላይ ያለውን ጽሑፍ ወደ {target} ተርጉም።",
                "እባክዎ ከላይ ያለውን አንቀጽ ወደ {target} ይተርጉሙ።",
                "ይህን ሁሉ ወደ {target} መተርጎም ትችላለህ?",
                "ከላይ ያለው ጽሑፍ በ{source} የተጻፈ ነው። ወደ {target} ተርጉመው።",
                "ከላይ ላለው ጽሑፍ ያለ ምንም አስተያየት የ{target} ትርጉም ስጠኝ።",
            ],
            "any": [
                "ይህን ወደ {target} ተርጉም።",
                "የዚህ ጽሑፍ የ{target} ትርጉም ያስፈልገኛል።",
                "ይህን በ{target} በል።",
                "ወደ {target} ተርጉም።",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "የሚከተለውን ጽሑፍ አጠቃልል።",
                "ይህን በ{target} አጠቃልል፦",
                "ከታች ላለው ጽሑፍ አጭር ማጠቃለያ ስጠኝ።",
                "እባክዎ ለሚከተለው ጽሑፍ አጭር ማጠቃለያ በ{target} ይጻፉ።",
                "ከታች ያለውን አንቀጽ አንብበህ በሁለት ወይም በሦስት ዓረፍተ ነገሮች አጠቃልል።",
                "በሚከተለው ጽሑፍ ውስጥ ያለው ዋና ሐሳብ ምንድን ነው? በ{target} በአጭሩ አጠቃልል።",
            ],
            "post": [
                "ከላይ ያለውን ጽሑፍ አጠቃልል።",
                "ከላይ ላለው ጽሑፍ አጭር ማጠቃለያ በ{target} ስጠኝ።",
                "እባክዎ ይህን በ{target} ያጠቃልሉ።",
                "ይህን በጥቂት ዓረፍተ ነገሮች ማጠቃለል ትችላለህ?",
                "ዋና ዋና ነጥቦቹ ምንድን ናቸው? በ{target} በማጠቃለያ መልስ።",
            ],
            "any": ["በ{target} አጠቃልል።", "ማጠቃለያ በ{target} እባክህ።"],
        },
        "headline": {
            "pre": [
                "ለሚከተለው ጽሑፍ ርዕስ ጻፍ።",
                "ለዚህ ዜና በ{target} ርዕስ ስጥ፦",
                "ለዚህ ጽሑፍ አጭር ርዕስ ጠቁም።",
                "ከታች ላለው ጽሑፍ ጥሩ ርዕስ ምን ሊሆን ይችላል? በ{target} መልስ።",
            ],
            "post": ["ከላይ ላለው ጽሑፍ ርዕስ ጻፍ።", "ከላይ ላለው ዜና በ{target} ርዕስ ስጥ።"],
            "any": ["ርዕስ በ{target} እባክህ።"],
        },
        "sentiment": {
            "pre": [
                "የሚከተለው ጽሑፍ ምን ዓይነት ስሜት (sentiment) ያሳያል? በPositive፣ Negative ወይም Neutral መልስ።",
                "የዚህን ዓረፍተ ነገር ስሜት ፈርጅ፦",
                "የሚከተለው ዓረፍተ ነገር አዎንታዊ፣ አሉታዊ ወይስ ገለልተኛ ነው? በPositive፣ Negative ወይም Neutral መልስ።",
                "በዚህ ጽሑፍ ላይ የስሜት ትንተና አድርግ፦",
            ],
            "post": ["ከላይ ያለው ጽሑፍ ምን ዓይነት ስሜት አለው?", "ይህ አዎንታዊ፣ አሉታዊ ወይስ ገለልተኛ ነው? በPositive፣ Negative ወይም Neutral መልስ።"],
        },
        "topic": {
            "pre": [
                "የዚህ የዜና ርዕስ ርዕሰ ጉዳይ ምንድን ነው?",
                "ይህን የዜና ርዕስ በርዕሰ ጉዳይ ፈርጅ፦",
                "የሚከተለው ጽሑፍ የየትኛው ምድብ ነው?",
                "ለዚህ ጽሑፍ የሚስማማውን የርዕሰ ጉዳይ መለያ ምረጥ፦",
            ],
            "post": ["ከላይ ያለው ጽሑፍ የየትኛው ርዕሰ ጉዳይ ነው?", "ይህ የየትኛው ምድብ ነው?"],
        },
        "emotion": {
            "pre": [
                "ይህ ጽሑፍ ምን ዓይነት ስሜት ይገልጻል? ከእነዚህ ምረጥ፦ {labels}።",
                "በሚከተለው ጽሑፍ ውስጥ ያለውን ስሜት ፈርጅ ({labels})።",
                "ጸሐፊው ምን ይሰማዋል? ከእነዚህ በአንዱ መልስ፦ {labels}።",
            ],
            "post": ["ከላይ የተገለጸው ስሜት የትኛው ነው? ከእነዚህ ምረጥ፦ {labels}።"],
        },
        "hate": {
            "pre": [
                "ይህ ጽሑፍ የጥላቻ ንግግር፣ ስድብ ወይስ ተራ ንግግር ነው? በHate፣ Abuse ወይም Normal መልስ።",
                "ይህን ጽሑፍ እንደ Normal፣ Abuse ወይም Hate ፈርጅ፦",
            ],
            "post": ["ከላይ ያለው ጽሑፍ ጥላቻ ወይም ስድብ ይዟል?"],
        },
        "intent": {
            "pre": [
                "ተናጋሪው ምን ይፈልጋል? ዓላማውን (intent) ጥቀስ።",
                "የዚህን ጥያቄ ዓላማ ለይ፦",
            ],
            "post": ["ከላይ ያለው ንግግር ዓላማው ምንድን ነው?"],
        },
        "language_id": {
            "pre": [
                "የሚከተለው ጽሑፍ በየትኛው ቋንቋ ተጽፏል?",
                "የዚህን ጽሑፍ ቋንቋ ለይ፦",
                "ይህ ምን ቋንቋ ነው? በቋንቋው ስም መልስ።",
                "ከታች ባለው ጽሑፍ ውስጥ ጥቅም ላይ የዋለውን ቋንቋ ጥቀስ።",
            ],
            "post": ["ከላይ ያለው ጽሑፍ በየትኛው ቋንቋ ተጽፏል?", "ያ የትኛው ቋንቋ ነው?"],
        },
        "qa_context": {
            "pre": [
                "ከታች ያለውን አንቀጽ ብቻ በመጠቀም ጥያቄውን መልስ።",
                "አንቀጹን አንብበህ የሚከተለውን ጥያቄ መልስ።",
                "አውዱን ተጠቅመህ መልስ፦",
            ],
            "post": ["ከላይ ያለውን አውድ ብቻ በመጠቀም ጥያቄውን መልስ።", "ከላይ ባለው አንቀጽ መሠረት መልሱ ምንድን ነው?"],
        },
        "qa_open": {
            "pre": [
                "የሚከተለውን ጥያቄ በ{target} መልስ።",
                "እባክዎ ይህን ጥያቄ በ{target} ይመልሱ፦",
                "አጭር እና ትክክለኛ መልስ በ{target} ስጥ፦",
            ],
            "post": ["ከላይ ያለውን ጥያቄ በ{target} መልስ።", "እባክህ በ{target} መልስ።"],
        },
        "answer_in": {
            "post": [
                "መልስህን በ{target} ስጥ።",
                "በ{target} ብቻ መልስ።",
                "እባክህ በ{target} መልስ።",
                "መልስህን በ{target} ጻፍ።",
                "መልስህ በ{target} መሆን አለበት።",
            ],
        },
        "math": {
            "pre": [
                "የሚከተለውን ችግር ደረጃ በደረጃ ፍታ፤ ከዚያም የመጨረሻውን መልስ ስጥ።",
                "ይህን የሒሳብ ችግር ሠርተህ የመጨረሻውን መልስ ስጥ።",
                "ፍታ፦",
                "አስተሳሰብህን አሳይና መልሱን በ{target} ስጥ።",
            ],
            "post": ["ከላይ ያለውን ችግር ደረጃ በደረጃ ፍታ።", "አሠራርህን አሳይና የመጨረሻውን መልስ ስጥ።"],
        },
        "mcq": {
            "pre": [
                "ከታች ያለውን ብዙ ምርጫ ጥያቄ በትክክለኛው ምርጫ ፊደል መልስ።",
                "ትክክለኛውን መልስ ምረጥ (A፣ B፣ C ወይም D)፦",
                "ለሚከተለው ጥያቄ ትክክለኛውን ምርጫ ምረጥ።",
            ],
            "post": ["የትኛው ምርጫ ትክክል ነው?", "በA፣ B፣ C ወይም D መልስ።"],
        },
        "nli": {
            "pre": [
                "ሁለተኛው ዓረፍተ ነገር ከመጀመሪያው ጋር ያለውን ግንኙነት ወስን። በentailment፣ neutral ወይም contradiction መልስ።",
                "የተፈጥሮ ቋንቋ ድምዳሜ፦ ጥንዱን በentailment፣ neutral ወይም contradiction ፈርጅ።",
            ],
            "post": ["በእነዚህ ሁለት ዓረፍተ ነገሮች መካከል ያለው ግንኙነት ምንድን ነው? በentailment፣ neutral ወይም contradiction መልስ።"],
        },
        "ner": {
            "pre": [
                "በዚህ ዓረፍተ ነገር ውስጥ ያሉ ልዩ ስሞችን (ሰው፣ ቦታ፣ ድርጅት፣ ቀን) ምልክት አድርግ። እያንዳንዱን ቃል ከ<tag> እና ከመለያው (PER፣ ORG፣ LOC፣ DATE ወይም O) ጋር ጻፍ።",
            ],
        },
        "pos": {
            "pre": ["በዚህ ዓረፍተ ነገር ውስጥ ላለ እያንዳንዱ ቃል የቃል ክፍሉን መለያ «ቃል <tag> መለያ» በሚል ቅርጽ ስጥ።"],
            "post": ["ከላይ ባለው ዓረፍተ ነገር ውስጥ ላለው ለእያንዳንዱ ቃል የቃል ክፍል ስጥ (ቃል <tag> መለያ)።"],
        },
        "define_term": {
            "pre": [
                "\"{title}\" ማለት ምን ማለት ነው? በ{target} አብራራ።",
                "\"{title}\" የሚለውን ቃል ትርጉም በ{target} ስጥ።",
            ],
        },
        "wiki_ask": {
            "pre": [
                "ስለ {title} ንገረኝ። መልስህን በ{target} ብቻ ስጥ።",
                "{title} ማን ወይም ምንድን ነው? በ{target} መልስ።",
                "ስለ {title} በ{target} የኢንሳይክሎፒዲያ ዓይነት ጽሑፍ ጻፍ።",
            ],
        },
        "story_prompt": {
            "pre": [
                "በ{target} አጭር ታሪክ ጻፍ።",
                "በ{target} ታሪክ ንገረኝ።",
                "ለልጆች የሚሆን ቀላል ታሪክ በ{target} ጻፍ።",
            ],
        },
    },
}
