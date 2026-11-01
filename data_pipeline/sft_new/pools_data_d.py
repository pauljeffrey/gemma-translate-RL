"""Instruction-template banks: Oromo (orm), Setswana (tsn), Sesotho (sot), Chichewa (nya).  Tier `low`.

These four are written with ONE safe construction: the request, then a label for the language (`Afaan: {target}`, `Puo: {target}`, `Chilankhulo: {target}`),
so the language name never has to take a case ending / prefix that would need native-speaker review.  Lingala, Luganda, Tigrinya, Malagasy, Wolof, Fon,
Efik, Ibibio and Urhobo are deliberately not authored: the author cannot write instruction text in them reliably (same rule as africomet_translation).
"""
TIER = {"orm": "low", "tsn": "low", "sot": "low", "nya": "low"}

NAMES = {
    "orm": {"eng": ["Afaan Ingiliffaa"], "fra": ["Afaan Faransaayii"], "orm": ["Afaan Oromoo"], "amh": ["Afaan Amaaraa"], "swh": ["Afaan Sawahilii"]},
    "tsn": {"eng": ["Sekgoa"], "fra": ["Sefora"], "tsn": ["Setswana"], "sot": ["Sesotho"], "afr": ["Seburu"], "zul": ["Sezulu"], "xho": ["Sexhosa"], "swh": ["Seswahili"]},
    "sot": {"eng": ["Senyesemane"], "fra": ["Sefora"], "sot": ["Sesotho"], "tsn": ["Setswana"], "afr": ["Seburu"], "zul": ["Sezulu"], "xho": ["Sexhosa"], "swh": ["Seswahili"]},
    "nya": {"eng": ["Chingerezi"], "fra": ["Chifalansa"], "nya": ["Chichewa"], "swh": ["Chiswahili"], "zul": ["Chizulu"], "sna": ["Chishona"]},
}

BANKS = {
    "orm": {
        "translate": {
            "pre": [
                "Barreeffama armaan gadii kana hiiki. Afaan itti hiikamu: {target}.",
                "Maaloo, kutaa armaan gadii kana hiiki. Afaan barbaadamu: {target}.",
                "Afaan barreeffamichaa: {source}. Barreeffama armaan gadii kana hiiki; hiikaa isaa qofa kenni. Afaan itti hiikamu: {target}.",
            ],
            "post": ["Barreeffama armaan olii kana hiiki. Afaan itti hiikamu: {target}.", "Maaloo, kutaa armaan olii kana hiiki. Afaan: {target}."],
            "any": ["Kana hiiki. Afaan: {target}.", "{source} → {target}"],
        },
        "summarize": {
            "pre": ["Barreeffama armaan gadii kana gabaabsi. Afaan: {target}.", "Maaloo, barreeffama kana gabaabsi. Afaan: {target}."],
            "post": ["Barreeffama armaan olii kana gabaabsi. Afaan: {target}."],
        },
        "sentiment": {
            "pre": ["Miira (sentiment) barreeffama kanaa maali? Positive, Negative moo Neutral jedhii deebisi."],
            "post": ["Miira barreeffama armaan olii kanaa maali? Positive, Negative moo Neutral jedhii deebisi."],
        },
        "language_id": {
            "pre": ["Barreeffamni kun afaan kamiin barreeffame?", "Afaan barreeffama kanaa addaan baasi:"],
            "post": ["Barreeffamni armaan olii kun afaan kamiin barreeffame?"],
        },
        "qa_context": {
            "pre": ["Gaaffii kana kutaa armaan gadii qofa fayyadamuun deebisi."],
            "post": ["Gaaffii kana barreeffama armaan olii qofa fayyadamuun deebisi."],
        },
        "qa_open": {
            "pre": ["Gaaffii kana deebisi. Afaan deebii: {target}."],
            "post": ["Gaaffii armaan olii kana deebisi. Afaan deebii: {target}."],
        },
        "answer_in": {"post": ["Deebii kee kenni. Afaan deebii: {target}.", "Deebii kee afaan kana qofaan kenni: {target}."]},
    },
    "tsn": {
        "translate": {
            "pre": [
                "Fetolela mokwalo o o latelang. Puo e e batlegang: {target}.",
                "Tsweetswee, fetolela karolo e e fa tlase. Puo e o e fetolelang mo go yone: {target}.",
                "Mokwalo o o fa tlase o kwadilwe ka {source}. Mpha fela phetolelo ya one. Puo e e batlegang: {target}.",
            ],
            "post": ["Fetolela mokwalo o o fa godimo. Puo e e batlegang: {target}.", "Tsweetswee, fetolela karolo e e fa godimo. Puo: {target}."],
            "any": ["Fetolela se. Puo: {target}.", "{source} → {target}"],
        },
        "summarize": {
            "pre": ["Dira kakaretso ya mokwalo o o latelang. Puo: {target}.", "Tsweetswee, kakaretsa mokwalo o. Puo: {target}."],
            "post": ["Dira kakaretso ya mokwalo o o fa godimo. Puo: {target}."],
        },
        "sentiment": {
            "pre": ["Maikutlo (sentiment) a mokwalo o ke eng? Araba ka Positive, Negative kgotsa Neutral."],
            "post": ["Maikutlo a mokwalo o o fa godimo ke eng? Araba ka Positive, Negative kgotsa Neutral."],
        },
        "language_id": {
            "pre": ["Mokwalo o o latelang o kwadilwe ka puo efe?", "Lemoga puo ya mokwalo o:"],
            "post": ["Mokwalo o o fa godimo o kwadilwe ka puo efe?"],
        },
        "qa_context": {
            "pre": ["Araba potso o dirisa karolo e e fa tlase fela."],
            "post": ["Araba potso o dirisa mokwalo o o fa godimo fela."],
        },
        "qa_open": {
            "pre": ["Araba potso e e latelang. Puo ya karabo: {target}."],
            "post": ["Araba potso e e fa godimo. Puo ya karabo: {target}."],
        },
        "answer_in": {"post": ["Fa karabo ya gago. Puo ya karabo: {target}.", "Araba ka puo e e latelang fela: {target}."]},
    },
    "sot": {
        "translate": {
            "pre": [
                "Fetolela temana e latelang. Puo e batlehang: {target}.",
                "Ka kopo, fetolela karolo e ka tlase. Puo eo u e fetolelang ho eona: {target}.",
                "Temana e ka tlase e ngotsoe ka {source}. Mpha feela phetolelo ea eona. Puo e batlehang: {target}.",
            ],
            "post": ["Fetolela temana e ka holimo. Puo e batlehang: {target}.", "Ka kopo, fetolela karolo e ka holimo. Puo: {target}."],
            "any": ["Fetolela sena. Puo: {target}.", "{source} → {target}"],
        },
        "summarize": {
            "pre": ["Etsa kakaretso ea temana e latelang. Puo: {target}.", "Ka kopo, akaretsa temana ena. Puo: {target}."],
            "post": ["Etsa kakaretso ea temana e ka holimo. Puo: {target}."],
        },
        "sentiment": {
            "pre": ["Maikutlo (sentiment) a temana ena ke afe? Araba ka Positive, Negative kapa Neutral."],
            "post": ["Maikutlo a temana e ka holimo ke afe? Araba ka Positive, Negative kapa Neutral."],
        },
        "language_id": {
            "pre": ["Temana ena e latelang e ngotsoe ka puo efe?", "Hlokomela puo ea temana ena:"],
            "post": ["Temana e ka holimo e ngotsoe ka puo efe?"],
        },
        "qa_context": {
            "pre": ["Araba potso u sebelisa karolo e ka tlase feela."],
            "post": ["Araba potso u sebelisa temana e ka holimo feela."],
        },
        "qa_open": {
            "pre": ["Araba potso e latelang. Puo ea karabo: {target}."],
            "post": ["Araba potso e ka holimo. Puo ea karabo: {target}."],
        },
        "answer_in": {"post": ["Fana ka karabo ea hao. Puo ea karabo: {target}.", "Araba ka puo ena feela: {target}."]},
    },
    "nya": {
        "translate": {
            "pre": [
                "Tembenuzira mawu otsatirawa. Chilankhulo chofunikira: {target}.",
                "Chonde, tembenuzirani ndime ili m'munsiyi. Chilankhulo chotembenuzira: {target}.",
                "Mawu a m'munsiwa analembedwa m'{source}. Ndipatseni kumasulira kwake kokha. Chilankhulo chofunikira: {target}.",
            ],
            "post": ["Tembenuzira mawu omwe ali pamwambapa. Chilankhulo chofunikira: {target}.", "Chonde, tembenuzirani ndime ili pamwambapa. Chilankhulo: {target}."],
            "any": ["Tembenuzira izi. Chilankhulo: {target}.", "{source} → {target}"],
        },
        "summarize": {
            "pre": ["Fupikitsa mawu otsatirawa. Chilankhulo: {target}.", "Chonde, fupikitsani nkhani iyi. Chilankhulo: {target}."],
            "post": ["Fupikitsa mawu omwe ali pamwambapa. Chilankhulo: {target}."],
        },
        "sentiment": {
            "pre": ["Maganizo (sentiment) a mawu otsatirawa ndi otani? Yankhani ndi Positive, Negative kapena Neutral."],
            "post": ["Maganizo a mawu omwe ali pamwambapa ndi otani? Yankhani ndi Positive, Negative kapena Neutral."],
        },
        "language_id": {
            "pre": ["Mawu otsatirawa analembedwa m'chilankhulo chiti?", "Zindikirani chilankhulo cha mawu awa:"],
            "post": ["Mawu omwe ali pamwambapa analembedwa m'chilankhulo chiti?"],
        },
        "qa_context": {
            "pre": ["Yankhani funsoli pogwiritsa ntchito ndime ili m'munsiyi yokha."],
            "post": ["Yankhani funsoli pogwiritsa ntchito mawu omwe ali pamwambapa okha."],
        },
        "qa_open": {
            "pre": ["Yankhani funso lotsatirali. Chilankhulo cha yankho: {target}."],
            "post": ["Yankhani funso lomwe lili pamwambapa. Chilankhulo cha yankho: {target}."],
        },
        "answer_in": {"post": ["Perekani yankho lanu. Chilankhulo cha yankho: {target}.", "Yankhani m'chilankhulo ichi chokha: {target}."]},
    },
}
