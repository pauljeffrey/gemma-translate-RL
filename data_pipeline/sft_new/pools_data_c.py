"""Instruction-template banks: Kinyarwanda (kin), Shona (sna), Somali (som).  Tier `low`: short, standard constructions only; see pools_multi.py.

Language names follow how each language attaches them: kin `mu {target}` (names without the initial vowel: Cyongereza, Gifaransa ...), sna `mu{target}`
(muchiRungu), som `Af-{target}` (Af-Ingiriis).  A language without a known local name falls back to its English name, which is still readable.
"""
TIER = {"kin": "low", "sna": "low", "som": "low"}

NAMES = {
    "kin": {
        "eng": ["Cyongereza"], "fra": ["Gifaransa"], "kin": ["Kinyarwanda"], "swh": ["Giswahili"],
    },
    "sna": {
        "eng": ["chiRungu"], "fra": ["chiFurenzi"], "sna": ["chiShona"], "swh": ["chiSwahili"], "zul": ["chiZulu"], "xho": ["chiXhosa"], "afr": ["chiAfrikaans"],
        "hau": ["chiHausa"], "yor": ["chiYoruba"], "ibo": ["chiIgbo"], "amh": ["chiAmharic"], "som": ["chiSomali"], "kin": ["chiKinyarwanda"],
        "nya": ["chiChewa"], "tsn": ["chiTswana"], "sot": ["chiSotho"], "lin": ["chiLingala"], "wol": ["chiWolof"], "orm": ["chiOromo"],
    },
    "som": {
        "eng": ["Ingiriis"], "fra": ["Faransiis"], "som": ["Soomaali"], "swh": ["Sawaaxili"], "hau": ["Hawsa"], "yor": ["Yoruba"], "ibo": ["Igbo"],
        "zul": ["Zulu"], "xho": ["Xhosa"], "afr": ["Afrikaans"], "amh": ["Amxaari"], "orm": ["Oromo"], "ti": ["Tigrinya"], "kin": ["Kinyarwanda"],
        "sna": ["Shona"], "nya": ["Chichewa"], "lin": ["Lingala"], "wol": ["Wolof"], "mlg": ["Malagasi"],
    },
}

BANKS = {
    # ------------------------------------------------------------------------------------------------ Kinyarwanda
    "kin": {
        "translate": {
            "pre": [
                "Hindura iyi nyandiko ikurikira mu {target}.",
                "Nyamuneka hindura igika kiri hasi mu {target}.",
                "Ndashaka ko uhindura iyi nyandiko mu {target}, mu buryo busanzwe aho kuba ijambo ku rindi.",
                "Iyi nyandiko iri hasi yanditse mu {source}. Mpa ubuhinduzi bwayo mu {target} gusa, nta bindi bisobanuro.",
                "Kuva mu {source} ujya mu {target}:",
                "Uri umuhinduzi w'umwuga. Hindura igika cyo mu {source} kiri hasi mu {target}, urinde amazina, imibare n'ibimenyetso by'imyandikire. Tanga ubuhinduzi gusa.",
            ],
            "post": [
                "Hindura inyandiko iri hejuru mu {target}.",
                "Nyamuneka hindura igika kiri hejuru mu {target}.",
                "Wahindura ibi byose mu {target}?",
                "Inyandiko iri hejuru yanditse mu {source}. Iyihindure mu {target}.",
            ],
            "any": [
                "Hindura ibi mu {target}.",
                "Nkeneye ubuhinduzi bw'iyi nyandiko mu {target}.",
                "Vuga ibi mu {target}.",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "Tanga incamake y'inyandiko ikurikira.",
                "Incamake y'iyi nyandiko mu {target}:",
                "Mpa incamake ngufi y'inyandiko iri hasi.",
                "Nyamuneka andika incamake ngufi y'iyi ngingo mu {target}.",
                "Soma igika kiri hasi hanyuma ugikoreho incamake mu nteruro ebyiri cyangwa eshatu.",
            ],
            "post": [
                "Tanga incamake y'inyandiko iri hejuru.",
                "Mpa incamake ngufi y'ibyavuzwe hejuru mu {target}.",
                "Wabigabanya mu nteruro nke?",
            ],
            "any": ["Incamake mu {target}.", "Incamake mu {target}, nyamuneka."],
        },
        "headline": {
            "pre": ["Andika umutwe w'inkuru ikurikira.", "Tanga umutwe w'iyi nkuru mu {target}:"],
            "post": ["Andika umutwe w'inyandiko iri hejuru."],
            "any": ["Umutwe w'inkuru mu {target}, nyamuneka."],
        },
        "sentiment": {
            "pre": [
                "Amarangamutima y'iyi nyandiko ikurikira ni ayahe? Subiza Positive, Negative cyangwa Neutral.",
                "Suzuma amarangamutima y'iyi nteruro:",
            ],
            "post": ["Amarangamutima y'inyandiko iri hejuru ni ayahe?"],
        },
        "topic": {
            "pre": ["Iyi nkuru ivuga ku ngingo ki?", "Shyira iyi nkuru mu cyiciro gikwiye:", "Iyi nyandiko ikurikira ni iy'icyiciro ki?"],
            "post": ["Inyandiko iri hejuru ni iy'icyiciro ki?"],
        },
        "language_id": {
            "pre": [
                "Iyi nyandiko yanditse mu rurimi rurihe?",
                "Menya ururimi rw'iyi nyandiko:",
                "Uru ni urwuhe rurimi? Subiza ukoresheje izina ry'ururimi.",
            ],
            "post": ["Inyandiko iri hejuru yanditse mu rurimi rurihe?"],
        },
        "qa_context": {
            "pre": [
                "Subiza ikibazo ukoresheje gusa igika kiri hasi.",
                "Soma igika hanyuma usubize ikibazo gikurikira.",
            ],
            "post": ["Subiza ikibazo ukoresheje gusa inyandiko iri hejuru.", "Hakurikijwe igika kiri hejuru, igisubizo ni ikihe?"],
        },
        "qa_open": {
            "pre": [
                "Subiza ikibazo gikurikira mu {target}.",
                "Nyamuneka subiza iki kibazo mu {target}:",
                "Tanga igisubizo kigufi kandi nyacyo mu {target}:",
            ],
            "post": ["Subiza ikibazo kiri hejuru mu {target}.", "Nyamuneka subiza icyo kibazo mu {target}."],
        },
        "answer_in": {
            "post": [
                "Tanga igisubizo cyawe mu {target}.",
                "Subiza mu {target} gusa.",
                "Nyamuneka subiza mu {target}.",
                "Andika igisubizo cyawe mu {target}.",
                "Igisubizo cyawe kigomba kuba mu {target}.",
            ],
        },
        "math": {
            "pre": ["Kemura ikibazo gikurikira intambwe ku ntambwe, hanyuma utange igisubizo cya nyuma.", "Kemura:"],
            "post": ["Kemura ikibazo kiri hejuru intambwe ku ntambwe."],
        },
    },
    # ------------------------------------------------------------------------------------------------ Shona
    "sna": {
        "translate": {
            "pre": [
                "Dudzira zvinyorwa zvinotevera mu{target}.",
                "Ndapota dudzira ndima iri pazasi mu{target}.",
                "Ndinoda kuti ududzire zvinyorwa izvi mu{target}, nenzira yakanaka isiri izwi nezwi.",
                "Zvinyorwa zviri pazasi zvakanyorwa mu{source}. Ndipe kududzira kwazvo mu{target} chete, pasina tsananguro.",
                "Kubva mu{source} kuenda mu{target}:",
                "Uri mududziri ane unyanzvi. Dudzira ndima iri pazasi mu{target}, uchichengeta mazita, nhamba nezviratidzo zvekunyora. Ingopa kududzira.",
            ],
            "post": [
                "Dudzira zvinyorwa zviri pamusoro mu{target}.",
                "Ndapota dudzira ndima iri pamusoro mu{target}.",
                "Unogona kududzira zvese izvi mu{target} here?",
                "Zvinyorwa zviri pamusoro zvakanyorwa mu{source}. Zvidudzire mu{target}.",
            ],
            "any": [
                "Dudzira izvi mu{target}.",
                "Ndinoda kududzira kwezvinyorwa izvi mu{target}.",
                "Taura izvi mu{target}.",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "Pfupisa zvinyorwa zvinotevera.",
                "Pfupisa izvi mu{target}:",
                "Ndipe chidimbu chipfupi chezvinyorwa zviri pazasi.",
                "Ndapota nyora chidimbu chipfupi chenyaya inotevera mu{target}.",
                "Verenga ndima iri pazasi wozoipfupisa mumitsara miviri kana mitatu.",
            ],
            "post": [
                "Pfupisa zvinyorwa zviri pamusoro.",
                "Ndipe chidimbu chipfupi chezviri pamusoro mu{target}.",
                "Ndapota pfupisa izvi mu{target}.",
            ],
            "any": ["Pfupisa mu{target}."],
        },
        "headline": {
            "pre": ["Nyora musoro wenyaya inotevera.", "Ipa nyaya iyi musoro mu{target}:"],
            "post": ["Nyora musoro wezvinyorwa zviri pamusoro."],
            "any": ["Musoro wenyaya mu{target}, ndapota."],
        },
        "sentiment": {
            "pre": [
                "Manzwiro ezvinyorwa zvinotevera ndeapi? Pindura nePositive, Negative kana Neutral.",
                "Ongorora manzwiro echirevo ichi:",
            ],
            "post": ["Manzwiro ezvinyorwa zviri pamusoro ndeapi?"],
        },
        "topic": {
            "pre": ["Nyaya iyi inotaura nezvei?", "Isa nyaya iyi muchikamu chakakodzera:", "Zvinyorwa zvinotevera zvinoenda muchikamu chipi?"],
            "post": ["Zvinyorwa zviri pamusoro zvinoenda muchikamu chipi?"],
        },
        "language_id": {
            "pre": [
                "Zvinyorwa zvinotevera zvakanyorwa nemutauro upi?",
                "Ziva mutauro wezvinyorwa izvi:",
                "Uyu mutauro upi? Pindura nezita remutauro.",
            ],
            "post": ["Zvinyorwa zviri pamusoro zvakanyorwa nemutauro upi?"],
        },
        "qa_context": {
            "pre": ["Pindura mubvunzo uchishandisa ndima iri pazasi chete.", "Verenga ndima wozopindura mubvunzo unotevera."],
            "post": ["Pindura mubvunzo uchishandisa zvinyorwa zviri pamusoro chete."],
        },
        "qa_open": {
            "pre": ["Pindura mubvunzo unotevera mu{target}.", "Ndapota pindura mubvunzo uyu mu{target}:"],
            "post": ["Pindura mubvunzo uri pamusoro mu{target}."],
        },
        "answer_in": {
            "post": [
                "Ipa mhinduro yako mu{target}.",
                "Pindura mu{target} chete.",
                "Ndapota pindura mu{target}.",
                "Nyora mhinduro yako mu{target}.",
            ],
        },
        "math": {
            "pre": ["Gadzirisa dambudziko rinotevera nhanho nenhanho, wozopa mhinduro yekupedzisira.", "Gadzirisa:"],
            "post": ["Gadzirisa dambudziko riri pamusoro nhanho nenhanho."],
        },
    },
    # ------------------------------------------------------------------------------------------------ Somali
    "som": {
        "translate": {
            "pre": [
                "U tarjun qoraalka soo socda Af-{target}.",
                "Fadlan u tarjun qeybta hoose Af-{target}.",
                "Ii tarjun qoraalkan Af-{target}, si dabiici ah oo aan ahayn eray-ereyga.",
                "Qoraalka hoose waxa uu ku qoran yahay Af-{source}. Ii sii tarjumaaddiisa Af-{target} oo keliya, iyadoo aan faahfaahin dheeraad ah lahayn.",
                "Laga bilaabo Af-{source} ilaa Af-{target}:",
                "Adigu waxaad tahay turjubaan xirfad leh. U tarjun qeybta Af-{source} ee hoose Af-{target}, adigoo ilaalinaya magacyada, tirooyinka iyo calaamadaha qoraalka. Bixi tarjumaadda oo keliya.",
            ],
            "post": [
                "U tarjun qoraalka kor ku xusan Af-{target}.",
                "Fadlan u tarjun qeybta kor ku qoran Af-{target}.",
                "Ma u tarjumi kartaa wax kasta oo kor ku qoran Af-{target}?",
                "Qoraalka kor ku qoran waxa uu ku qoran yahay Af-{source}. U tarjun Af-{target}.",
            ],
            "any": [
                "Tan u tarjun Af-{target}.",
                "Waxaan u baahanahay tarjumaadda qoraalkan Af-{target}.",
                "U tarjun Af-{target}.",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "Soo koob qoraalka soo socda.",
                "Tan soo koob Af-{target}:",
                "Ii sii soo koobid gaaban oo ku saabsan qoraalka hoose.",
                "Fadlan qor soo koobid gaaban oo ku saabsan maqaalkan soo socda Af-{target}.",
                "Akhri qeybta hoose kadibna ku soo koob laba ama saddex jumladood.",
            ],
            "post": ["Soo koob qoraalka kor ku xusan.", "Ii sii soo koobid gaaban oo ku saabsan waxa kor ku qoran Af-{target}."],
            "any": ["Soo koob Af-{target}."],
        },
        "headline": {
            "pre": ["Qor cinwaanka maqaalka soo socda.", "U qor sheekadan cinwaan Af-{target}:"],
            "post": ["Qor cinwaanka qoraalka kor ku xusan."],
            "any": ["Cinwaan Af-{target}, fadlan."],
        },
        "sentiment": {
            "pre": [
                "Dareenka (sentiment) qoraalkan soo socda waa maxay? Ku jawaab Positive, Negative ama Neutral.",
                "Falanqee dareenka jumladdan:",
            ],
            "post": ["Dareenka qoraalka kor ku xusan waa maxay?"],
        },
        "topic": {
            "pre": ["Mawduuca ciwaankan wararka waa maxay?", "Qoraalka soo socda waa nooc noocee ah?"],
            "post": ["Qoraalka kor ku xusan wuxuu ku saabsan yahay mawduuc noocee ah?"],
        },
        "language_id": {
            "pre": [
                "Qoraalka soo socda luqad kee ku qoran yahay?",
                "Garo luqadda qoraalkan:",
                "Luqaddan waa luqad kee? Ku jawaab magaca luqadda.",
            ],
            "post": ["Qoraalka kor ku xusan luqad kee ku qoran yahay?"],
        },
        "qa_context": {
            "pre": [
                "Ka jawaab su'aasha adigoo isticmaalaya qeybta hoose oo keliya.",
                "Akhri qeybta kadibna ka jawaab su'aasha soo socota.",
            ],
            "post": ["Ka jawaab su'aasha adigoo isticmaalaya qoraalka kor ku xusan oo keliya."],
        },
        "qa_open": {
            "pre": ["Ka jawaab su'aasha soo socota Af-{target}.", "Fadlan ka jawaab su'aashan Af-{target}:"],
            "post": ["Ka jawaab su'aasha kor ku xusan Af-{target}."],
        },
        "answer_in": {
            "post": [
                "Ku bixi jawaabtaada Af-{target}.",
                "Ku jawaab Af-{target} oo keliya.",
                "Fadlan ku jawaab Af-{target}.",
                "Ku qor jawaabtaada Af-{target}.",
                "Jawaabtaadu waa inay noqotaa Af-{target}.",
            ],
        },
        "math": {
            "pre": ["Xalli su'aasha xisaabta ee soo socota tallaabo tallaabo, kadibna bixi jawaabta ugu dambeysa.", "Xalli:"],
            "post": ["Xalli su'aasha xisaabta ee kor ku xusan tallaabo tallaabo."],
        },
    },
}
