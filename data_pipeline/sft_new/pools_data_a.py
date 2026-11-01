"""Instruction-template banks: French (fra), Swahili (swh), Afrikaans (afr).  See pools_multi.py for the contract.

Label words (Positive / Negative / Neutral, entailment / neutral / contradiction, PER / ORG / LOC / DATE / O, A-D, Hate / Abuse / Normal ...) stay English in
every instruction language, exactly as in the existing eng / pcm / yor / hau / ibo pools, so the label vocabulary of a task is one set.
"""
TIER = {"fra": "high", "swh": "medium", "afr": "medium"}

NAMES = {
    "fra": {
        "eng": ["anglais"], "fra": ["français"], "swh": ["swahili", "kiswahili"], "hau": ["haoussa"], "yor": ["yoruba"], "ibo": ["igbo"], "zul": ["zoulou"],
        "xho": ["xhosa"], "afr": ["afrikaans"], "amh": ["amharique"], "som": ["somali"], "kin": ["kinyarwanda"], "sna": ["shona"], "orm": ["oromo"],
        "tsn": ["tswana", "setswana"], "sot": ["sotho du Sud", "sesotho"], "nya": ["chichewa"], "wol": ["wolof"], "lin": ["lingala"], "lug": ["luganda"],
        "ti": ["tigrigna"], "mlg": ["malgache"], "pcm": ["pidgin nigérian", "pidgin"], "twi": ["twi"], "ewe": ["éwé"], "fon": ["fon"], "efi": ["efik"],
        "ibb": ["ibibio"], "urh": ["urhobo"], "bem": ["bemba"], "kik": ["kikuyu"], "luo": ["luo"], "nso": ["sepedi"], "tso": ["tsonga"], "ven": ["venda"],
        "ssw": ["swati"], "nbl": ["ndébélé du Sud"], "bm": ["bambara"], "dyu": ["dioula"], "kab": ["kabyle"], "mos": ["mooré"], "dag": ["dagbani"],
    },
    "swh": {
        "eng": ["Kiingereza"], "fra": ["Kifaransa"], "swh": ["Kiswahili"], "hau": ["Kihausa"], "yor": ["Kiyoruba"], "ibo": ["Kiigbo"], "zul": ["Kizulu"],
        "xho": ["Kixhosa"], "afr": ["Kiafrikaans"], "amh": ["Kiamhari"], "som": ["Kisomali"], "kin": ["Kinyarwanda"], "sna": ["Kishona"], "orm": ["Kioromo"],
        "tsn": ["Kitswana"], "sot": ["Kisotho"], "nya": ["Kichewa"], "wol": ["Kiwolof"], "lin": ["Kilingala"], "lug": ["Kiganda"], "ti": ["Kitigrinya"],
        "mlg": ["Kimalagasi"], "pcm": ["Pijini ya Nigeria", "Kipijini"], "twi": ["Kitwi"], "ewe": ["Kiewe"], "fon": ["Kifon"], "efi": ["Kiefik"],
        "ibb": ["Kiibibio"], "urh": ["Kiurhobo"], "bem": ["Kibemba"], "kik": ["Kikikuyu"], "luo": ["Kiluo"], "ssw": ["Kiswati"], "bm": ["Kibambara"],
    },
    "afr": {
        "eng": ["Engels"], "fra": ["Frans"], "swh": ["Swahili"], "hau": ["Hausa"], "yor": ["Yoruba"], "ibo": ["Igbo"], "zul": ["Zoeloe"], "xho": ["Xhosa"],
        "afr": ["Afrikaans"], "amh": ["Amharies"], "som": ["Somalies"], "kin": ["Kinyarwanda"], "sna": ["Shona"], "orm": ["Oromo"], "tsn": ["Tswana"],
        "sot": ["Suid-Sotho", "Sesotho"], "nya": ["Chichewa"], "wol": ["Wolof"], "lin": ["Lingala"], "lug": ["Luganda"], "ti": ["Tigrinya"],
        "mlg": ["Malgassies"], "pcm": ["Nigeriese Pidgin"], "twi": ["Twi"], "ewe": ["Ewe"], "fon": ["Fon"], "efi": ["Efik"], "ibb": ["Ibibio"],
        "urh": ["Urhobo"], "nso": ["Noord-Sotho", "Sepedi"], "tso": ["Tsonga"], "ven": ["Venda"], "ssw": ["Swazi"], "nbl": ["Suid-Ndebele"],
    },
}

BANKS = {
    # ------------------------------------------------------------------------------------------------ French
    "fra": {
        "translate": {
            "pre": [
                "Traduis le texte suivant en {target}.",
                "Veuillez traduire le passage ci-dessous en {target}.",
                "Traduction en {target}, s'il te plaît :",
                "Peux-tu m'aider à traduire ce qui suit en {target} ? Garde un ton naturel, pas du mot à mot.",
                "Le texte ci-dessous est en {source}. Donne-moi uniquement sa traduction en {target}, sans explication.",
                "Traduis en {target} en conservant les noms propres et les chiffres :",
                "Vous êtes un traducteur professionnel. Traduisez le passage en {source} ci-dessous en {target}, en respectant les noms, les chiffres et la ponctuation. Ne donnez que la traduction.",
                "Rédige en {target} une traduction fidèle du texte qui suit.",
                "Mets le texte suivant en {target}.",
                "Quelle est la version en {target} du texte ci-dessous ?",
                "Voici un court texte en {source}. Fournis sa traduction en {target}, et rien d'autre.",
                "Traduis ceci en {target} de façon fluide et idiomatique, sans calquer la tournure d'origine :",
            ],
            "post": [
                "Traduis le texte ci-dessus en {target}.",
                "Peux-tu traduire ce qui précède en {target} ?",
                "Merci de traduire le passage ci-dessus en {target}, sans commentaire.",
                "Le texte ci-dessus est en {source}. Traduis-le en {target}.",
                "Mets tout cela en {target}, s'il te plaît.",
                "Donne-moi la traduction en {target} de ce qui précède.",
                "Traduisez le texte précédent en {target} en gardant le même registre de langue. Répondez uniquement par la traduction.",
                "Je voudrais la version en {target} de ce qui est écrit plus haut.",
            ],
            "any": [
                "Traduis ceci en {target}.",
                "J'ai besoin d'une traduction de ce texte en {target}.",
                "Dis cela en {target}.",
                "Traduire en {target}.",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "Résume le texte suivant.",
                "Résume ceci en {target} :",
                "Donne-moi un court résumé du texte ci-dessous.",
                "Rédige en {target} un bref résumé de l'article suivant.",
                "Lis le passage ci-dessous et résume-le en deux ou trois phrases.",
                "Quel est le point principal du texte ci-dessous ? Résume-le en {target}.",
                "Condense le texte suivant en quelques lignes, sans rien ajouter qui n'y figure pas.",
                "Fais un résumé concis en {target} de cet article :",
            ],
            "post": [
                "Résume le texte ci-dessus.",
                "Fais un court résumé de ce qui précède en {target}.",
                "Peux-tu condenser cela en quelques phrases ?",
                "Quels sont les points clés ? Réponds par un résumé en {target}.",
                "Rédige un bref résumé en {target} du passage ci-dessus.",
            ],
            "any": ["Résume en {target}.", "Un résumé en {target}, s'il te plaît."],
        },
        "headline": {
            "pre": [
                "Écris un titre pour l'article suivant.",
                "Donne un titre en {target} à cet article :",
                "Propose un titre court pour ce texte.",
                "Quel serait un bon titre pour le texte ci-dessous ? Réponds en {target}.",
            ],
            "post": [
                "Écris un titre pour le texte ci-dessus.",
                "Donne un titre en {target} à l'article ci-dessus.",
                "Trouve un titre accrocheur pour cela.",
            ],
            "any": ["Un titre en {target}, s'il te plaît."],
        },
        "sentiment": {
            "pre": [
                "Quel est le sentiment exprimé dans le texte suivant ? Réponds par Positive, Negative ou Neutral.",
                "Classe le sentiment de cette phrase :",
                "La phrase suivante est-elle positive, négative ou neutre ?",
                "Fais une analyse de sentiment de ce texte :",
                "Dis-moi le sentiment (Positive, Negative ou Neutral) :",
                "Lis la phrase ci-dessous et indique si l'auteur exprime un avis positif, négatif ou neutre.",
            ],
            "post": [
                "Quel est le sentiment du texte ci-dessus ?",
                "Est-ce positif, négatif ou neutre ? Réponds par Positive, Negative ou Neutral.",
                "Indique le sentiment de ce texte.",
            ],
        },
        "topic": {
            "pre": [
                "Quel est le sujet de ce titre d'actualité ?",
                "Classe ce titre dans une rubrique :",
                "À quelle catégorie appartient le texte suivant ?",
                "Choisis l'étiquette de sujet qui convient le mieux à ce texte :",
            ],
            "post": ["De quel sujet relève le texte ci-dessus ?", "Quelle est la catégorie de ce texte ?"],
        },
        "emotion": {
            "pre": [
                "Quelle émotion ce texte exprime-t-il ? Choisis parmi : {labels}.",
                "Classe l'émotion du texte suivant ({labels}).",
                "Que ressent l'auteur ? Réponds par l'une de ces étiquettes : {labels}.",
                "Détection d'émotion : étiquette ce texte avec l'une de ces valeurs : {labels}.",
            ],
            "post": ["Quelle émotion est exprimée ci-dessus ? Choisis parmi : {labels}.", "Nomme l'émotion du texte ci-dessus ({labels})."],
        },
        "hate": {
            "pre": [
                "Ce texte contient-il un discours de haine, des insultes ou rien de tel ? Réponds par Hate, Abuse ou Normal.",
                "Classe ce texte comme Normal, Abuse ou Hate :",
                "Le message suivant est-il offensant ? Étiquette-le comme Normal, Abuse ou Hate.",
            ],
            "post": ["Le texte ci-dessus est-il haineux ou injurieux ?", "Étiquette le message ci-dessus : Normal, Abuse ou Hate."],
        },
        "intent": {
            "pre": [
                "Que veut la personne ? Donne l'intention.",
                "Identifie l'intention de cette demande :",
                "Classe l'intention de l'énoncé suivant :",
                "Que cherche à faire l'utilisateur avec ce message ?",
            ],
            "post": ["Quelle est l'intention de l'énoncé ci-dessus ?", "Donne l'intention de cette demande."],
        },
        "language_id": {
            "pre": [
                "Dans quelle langue le texte suivant est-il écrit ?",
                "Identifie la langue de ce texte :",
                "Quelle est cette langue ? Réponds par le nom de la langue.",
                "Indique la langue utilisée dans le texte ci-dessous.",
            ],
            "post": ["Dans quelle langue le texte ci-dessus est-il écrit ?", "Quelle est cette langue ?"],
        },
        "qa_context": {
            "pre": [
                "Réponds à la question en utilisant uniquement le passage ci-dessous.",
                "Lis le passage puis réponds à la question qui suit.",
                "Utilise le contexte pour répondre :",
                "En t'appuyant exclusivement sur le texte ci-dessous, réponds à la question.",
            ],
            "post": [
                "Réponds à la question en utilisant uniquement le contexte ci-dessus.",
                "D'après le passage ci-dessus, quelle est la réponse ?",
            ],
        },
        "qa_open": {
            "pre": [
                "Réponds à la question suivante en {target}.",
                "Réponds à cette question en {target}, s'il te plaît :",
                "Donne une réponse courte et exacte en {target} :",
            ],
            "post": ["Réponds à la question ci-dessus en {target}.", "Réponds-y en {target}, s'il te plaît."],
        },
        "answer_in": {
            "post": [
                "Réponds en {target}.",
                "Donne ta réponse uniquement en {target}.",
                "Merci de répondre en {target}.",
                "Rédige ta réponse en {target}.",
                "Ta réponse doit être en {target}.",
                "Réponds seulement en {target}.",
            ],
        },
        "math": {
            "pre": [
                "Résous le problème suivant étape par étape, puis donne la réponse finale.",
                "Résous ce problème de mathématiques et donne la réponse finale.",
                "Résous :",
                "Montre ton raisonnement, puis donne la réponse en {target}.",
            ],
            "post": ["Résous le problème ci-dessus étape par étape.", "Montre ta démarche et donne la réponse finale."],
        },
        "mcq": {
            "pre": [
                "Réponds à la question à choix multiples ci-dessous avec la lettre de la bonne option.",
                "Choisis la bonne réponse (A, B, C ou D) :",
                "Choisis la bonne option pour la question suivante.",
            ],
            "post": ["Quelle option est correcte ?", "Réponds par A, B, C ou D."],
        },
        "nli": {
            "pre": [
                "Détermine le lien entre la seconde phrase et la première. Réponds entailment, neutral ou contradiction.",
                "Inférence en langage naturel : étiquette la paire par entailment, neutral ou contradiction.",
                "Lis les deux phrases et dis si l'hypothèse découle de la prémisse (entailment), la contredit (contradiction) ou reste indéterminée (neutral).",
            ],
            "post": ["Quel est le lien entre ces deux phrases ? Réponds entailment, neutral ou contradiction."],
        },
        "ner": {
            "pre": [
                "Repère les entités nommées de cette phrase. Écris chaque mot suivi de <tag> et de son étiquette (PER, ORG, LOC, DATE ou O).",
                "Fais de la reconnaissance d'entités nommées sur la phrase ci-dessous, sous forme de paires mot <tag> ÉTIQUETTE.",
            ],
        },
        "pos": {
            "pre": [
                "Donne l'étiquette grammaticale de chaque mot de cette phrase sous forme de paires mot <tag> ÉTIQUETTE.",
                "Fais l'étiquetage morpho-syntaxique de la phrase suivante : écris chaque mot suivi de <tag> et de son étiquette.",
            ],
            "post": ["Donne la catégorie grammaticale de chaque mot de la phrase ci-dessus (mot <tag> ÉTIQUETTE)."],
        },
        "define_term": {
            "pre": [
                "Que signifie « {title} » ? Explique-le en {target}.",
                "Donne le sens du mot « {title} » en {target}.",
                "Définis « {title} » ({target}).",
                "Je suis tombé sur le terme « {title} ». Que veut-il dire ?",
            ],
            "any": ["« {title} » : signification en {target} ?"],
        },
        "wiki_ask": {
            "pre": [
                "Dis-moi ce que tu sais sur {title}. Réponds uniquement en {target}.",
                "Qui est {title}, ou de quoi s'agit-il ? Réponds en {target}.",
                "Rédige un article de type encyclopédique sur {title} en {target}.",
                "Explique-moi {title} en {target}.",
                "Que sais-tu sur {title} ? Réponds en {target}.",
            ],
            "any": ["{title} : donne-moi l'essentiel, en {target}."],
        },
        "scam": {
            "pre": [
                "Ce message est-il une arnaque ? Réponds Scam ou Not scam, puis explique pourquoi en une ou deux phrases.",
                "Vérifie si le message suivant est frauduleux et dis ce qui le trahit.",
                "Analyse ce message et dis-moi s'il s'agit d'une tentative d'escroquerie.",
            ],
            "post": ["Le message ci-dessus est-il une arnaque ? Explique."],
        },
        "story_prompt": {
            "pre": [
                "Écris une histoire courte en {target}.",
                "Raconte-moi une histoire en {target}.",
                "Écris une histoire simple pour enfants en {target}.",
                "J'aimerais une courte histoire en {target}, avec un début, un milieu et une fin.",
            ],
            "any": ["Une histoire courte en {target}, s'il te plaît."],
        },
        "story": {
            "pre": [
                "Écris une histoire courte en {target} à partir de ce synopsis :",
                "Raconte une histoire en {target} à partir de l'idée suivante :",
                "Transforme ce synopsis en une histoire courte en {target} :",
            ],
            "post": ["Écris une histoire en {target} à partir du synopsis ci-dessus.", "Raconte maintenant l'histoire complète en {target}."],
        },
        "synopsis": {
            "pre": ["Écris un synopsis en une phrase de l'histoire suivante, en {target} :", "De quoi parle cette histoire ? Donne un court synopsis en {target}."],
            "post": ["Écris un synopsis en une phrase de l'histoire ci-dessus, en {target}."],
        },
        "refine_translation": {
            "pre": [
                "Voici un texte source et un brouillon de traduction en {target}. Améliore la traduction.",
                "Relis cette traduction en {target} et donne une version corrigée.",
                "Affine le brouillon de traduction ci-dessous pour qu'il soit exact et naturel en {target}.",
            ],
            "post": ["Améliore le brouillon de traduction ci-dessus (traduction en {target}).", "Donne une traduction corrigée en {target} du texte source ci-dessus."],
        },
    },
    # ------------------------------------------------------------------------------------------------ Swahili
    "swh": {
        "translate": {
            "pre": [
                "Tafsiri maandishi yafuatayo kwa {target}.",
                "Tafadhali tafsiri kifungu kilicho hapa chini kwa {target}.",
                "Nakuomba unitafsirie maandishi haya kwa {target}, kwa lugha ya kawaida na si neno kwa neno.",
                "Maandishi yaliyo hapa chini yako katika {source}. Nipe tafsiri yake kwa {target} tu, bila maelezo ya ziada.",
                "Geuza maandishi yafuatayo kuwa {target}:",
                "Wewe ni mfasiri mtaalamu. Tafsiri kifungu cha {source} kilicho hapa chini kwa {target}, ukihifadhi majina, nambari na alama za uandishi. Toa tafsiri pekee.",
                "Kutoka {source} kwenda {target}:",
                "Andika tafsiri sahihi ya {target} ya maandishi yafuatayo.",
                "Tafsiri kwa {target}, ukiacha majina ya watu na mahali kama yalivyo:",
                "Naomba tafsiri ya {target} ya maandishi yafuatayo, yenye lugha laini na ya asili.",
            ],
            "post": [
                "Tafsiri maandishi yaliyo hapo juu kwa {target}.",
                "Tafadhali tafsiri kifungu kilicho hapo juu kwa {target}.",
                "Je, unaweza kutafsiri haya yote kwa {target}?",
                "Maandishi yaliyo hapo juu yako katika {source}. Yatafsiri kwa {target}.",
                "Nipe tafsiri ya {target} ya maandishi yaliyo hapo juu, bila maoni yoyote.",
                "Weka yote yaliyo hapo juu katika {target}, tafadhali.",
            ],
            "any": [
                "Tafsiri haya kwa {target}.",
                "Nahitaji tafsiri ya maandishi haya kwa {target}.",
                "Sema hili kwa {target}.",
                "Tafsiri kwa {target}.",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "Fupisha maandishi yafuatayo.",
                "Fupisha hili kwa {target}:",
                "Nipe muhtasari mfupi wa maandishi yaliyo hapa chini.",
                "Tafadhali andika muhtasari mfupi wa makala ifuatayo kwa {target}.",
                "Soma kifungu kilicho hapa chini kisha kifupishe kwa sentensi mbili au tatu.",
                "Wazo kuu la maandishi yaliyo hapa chini ni lipi? Lieleze kwa muhtasari kwa {target}.",
                "Andika muhtasari mfupi wa habari hii kwa {target}:",
            ],
            "post": [
                "Fupisha maandishi yaliyo hapo juu.",
                "Nipe muhtasari mfupi wa yaliyo hapo juu kwa {target}.",
                "Tafadhali fupisha hili kwa {target}.",
                "Je, unaweza kufupisha haya kwa sentensi chache?",
                "Mambo muhimu ni yapi? Jibu kwa muhtasari wa {target}.",
            ],
            "any": ["Fupisha kwa {target}.", "Muhtasari kwa {target}, tafadhali."],
        },
        "headline": {
            "pre": [
                "Andika kichwa cha habari cha makala ifuatayo.",
                "Pendekeza kichwa kifupi cha habari cha maandishi haya kwa {target}:",
                "Kichwa cha habari kinachofaa kwa maandishi yaliyo hapa chini ni kipi? Jibu kwa {target}.",
                "Ipe habari hii kichwa kwa {target}:",
            ],
            "post": [
                "Andika kichwa cha habari cha maandishi yaliyo hapo juu.",
                "Ipe makala iliyo hapo juu kichwa kwa {target}.",
                "Pendekeza kichwa cha habari cha kuvutia kwa haya.",
            ],
            "any": ["Kichwa cha habari kwa {target}, tafadhali."],
        },
        "sentiment": {
            "pre": [
                "Hisia za maandishi yafuatayo ni zipi? Jibu kwa Positive, Negative au Neutral.",
                "Changanua hisia za sentensi hii:",
                "Je, sentensi ifuatayo ni chanya, hasi au isiyoegemea upande wowote? Jibu kwa Positive, Negative au Neutral.",
                "Fanya uchambuzi wa hisia wa maandishi haya:",
                "Niambie hisia za maandishi haya (Positive, Negative au Neutral):",
                "Soma sentensi iliyo hapa chini kisha uamue kama mwandishi ana mtazamo chanya, hasi au wa kati.",
            ],
            "post": [
                "Hisia za maandishi yaliyo hapo juu ni zipi?",
                "Je, hilo ni chanya, hasi au la kati? Jibu kwa Positive, Negative au Neutral.",
                "Taja hisia za maandishi haya.",
            ],
        },
        "topic": {
            "pre": [
                "Mada ya kichwa hiki cha habari ni ipi?",
                "Weka kichwa hiki cha habari katika kundi la mada:",
                "Maandishi yafuatayo yanahusu kategoria gani?",
                "Chagua lebo ya mada inayofaa zaidi kwa maandishi haya:",
            ],
            "post": ["Maandishi yaliyo hapo juu yanahusu mada gani?", "Hii ni kategoria gani?"],
        },
        "emotion": {
            "pre": [
                "Maandishi haya yanaonyesha hisia gani? Chagua kati ya: {labels}.",
                "Weka hisia za maandishi yafuatayo katika kundi ({labels}).",
                "Mwandishi anahisi nini? Jibu kwa mojawapo ya: {labels}.",
                "Utambuzi wa hisia: weka lebo moja kati ya {labels} kwa maandishi haya.",
            ],
            "post": ["Ni hisia gani inayoonyeshwa hapo juu? Chagua kati ya: {labels}.", "Taja hisia za maandishi yaliyo hapo juu ({labels})."],
        },
        "hate": {
            "pre": [
                "Je, maandishi haya yana chuki, matusi au hayana lolote kati ya hayo? Jibu kwa Hate, Abuse au Normal.",
                "Weka maandishi haya katika kundi la Normal, Abuse au Hate:",
                "Je, ujumbe ufuatao una chuki au matusi? Upe lebo ya Normal, Abuse au Hate.",
            ],
            "post": ["Je, maandishi yaliyo hapo juu yana chuki au matusi?", "Weka lebo ya ujumbe ulio hapo juu: Normal, Abuse au Hate."],
        },
        "intent": {
            "pre": [
                "Mzungumzaji anataka nini? Taja dhamira yake (intent).",
                "Tambua dhamira ya ombi hili:",
                "Bainisha dhamira ya kauli ifuatayo:",
                "Mtumiaji anajaribu kufanya nini kwa ujumbe huu?",
            ],
            "post": ["Dhamira ya kauli iliyo hapo juu ni ipi?", "Taja dhamira ya ombi hilo."],
        },
        "language_id": {
            "pre": [
                "Maandishi yafuatayo yameandikwa kwa lugha gani?",
                "Tambua lugha ya maandishi haya:",
                "Hii ni lugha gani? Jibu kwa jina la lugha.",
                "Taja lugha iliyotumika katika maandishi yaliyo hapa chini.",
            ],
            "post": ["Maandishi yaliyo hapo juu yameandikwa kwa lugha gani?", "Hiyo ni lugha gani?"],
        },
        "qa_context": {
            "pre": [
                "Jibu swali kwa kutumia kifungu kilicho hapa chini pekee.",
                "Soma kifungu kisha ujibu swali linalofuata.",
                "Tumia muktadha huu kujibu swali linalofuata:",
                "Ukitegemea maandishi yaliyo hapa chini tu, jibu swali.",
            ],
            "post": ["Jibu swali kwa kutumia muktadha ulio hapo juu pekee.", "Kwa kuzingatia kifungu kilicho hapo juu, jibu ni lipi?"],
        },
        "qa_open": {
            "pre": [
                "Jibu swali lifuatalo kwa {target}.",
                "Tafadhali jibu swali hili kwa {target}:",
                "Toa jibu fupi na sahihi kwa {target}:",
            ],
            "post": ["Jibu swali lililo hapo juu kwa {target}.", "Tafadhali jibu swali hilo kwa {target}."],
        },
        "answer_in": {
            "post": [
                "Toa jibu lako kwa {target}.",
                "Jibu kwa {target} pekee.",
                "Tafadhali jibu kwa {target}.",
                "Andika jibu lako kwa {target}.",
                "Jibu lako lazima liwe kwa {target}.",
                "Jibu kwa {target} tu.",
            ],
        },
        "math": {
            "pre": [
                "Tatua tatizo lifuatalo hatua kwa hatua, kisha toa jibu la mwisho.",
                "Tatua tatizo hili la hesabu kisha toa jibu la mwisho.",
                "Tatua:",
                "Onyesha hatua zako kisha toa jibu kwa {target}.",
            ],
            "post": ["Tatua tatizo lililo hapo juu hatua kwa hatua.", "Onyesha hatua zako na utoe jibu la mwisho."],
        },
        "mcq": {
            "pre": [
                "Jibu swali la chaguo nyingi lililo hapa chini kwa herufi ya chaguo sahihi.",
                "Chagua jibu sahihi (A, B, C au D):",
                "Chagua chaguo sahihi kwa swali lifuatalo.",
            ],
            "post": ["Chaguo lipi ni sahihi?", "Jibu kwa A, B, C au D."],
        },
        "nli": {
            "pre": [
                "Amua uhusiano wa sentensi ya pili na ya kwanza. Jibu kwa entailment, neutral au contradiction.",
                "Uelewa wa lugha asilia: ipe jozi hii lebo ya entailment, neutral au contradiction.",
                "Soma sentensi zote mbili kisha useme kama dhana inafuata kutoka kwa msingi (entailment), inapingana nao (contradiction) au haijabainika (neutral).",
            ],
            "post": ["Uhusiano wa sentensi hizi mbili ni upi? Jibu kwa entailment, neutral au contradiction."],
        },
        "ner": {
            "pre": [
                "Weka alama kwenye majina maalum katika sentensi hii. Andika kila neno likifuatiwa na <tag> na lebo yake (PER, ORG, LOC, DATE au O).",
                "Tambua majina maalum (ya watu, mahali, mashirika au tarehe) katika sentensi iliyo hapa chini, ukitoa jozi za neno <tag> LEBO.",
            ],
        },
        "pos": {
            "pre": [
                "Toa lebo ya aina ya neno ya kila neno katika sentensi hii kama jozi za neno <tag> LEBO.",
                "Weka lebo za aina za maneno kwenye sentensi ifuatayo: andika kila neno likifuatiwa na <tag> na lebo yake.",
            ],
            "post": ["Toa aina ya kila neno katika sentensi iliyo hapo juu (neno <tag> LEBO)."],
        },
        "define_term": {
            "pre": [
                "Neno \"{title}\" lina maana gani? Eleza kwa {target}.",
                "Toa maana ya neno \"{title}\" kwa {target}.",
                "Fafanua \"{title}\" ({target}).",
                "Nimekutana na neno \"{title}\". Lina maana gani?",
            ],
            "any": ["\"{title}\" - maana yake kwa {target}?"],
        },
        "wiki_ask": {
            "pre": [
                "Niambie kuhusu {title}. Toa jibu kwa {target} pekee.",
                "{title} ni nani au nini? Jibu kwa {target}.",
                "Andika makala ya kiensaiklopidia kuhusu {title} kwa {target}.",
                "Nieleze kuhusu {title} kwa {target}.",
                "Unajua nini kuhusu {title}? Jibu kwa {target}.",
            ],
            "any": ["{title} - nipe taarifa muhimu kwa {target}."],
        },
        "scam": {
            "pre": [
                "Je, ujumbe huu ni wa utapeli? Jibu Scam au Not scam, kisha eleza kwa nini kwa sentensi moja au mbili.",
                "Kagua ujumbe ufuatao kama una ulaghai na useme ni nini kinachouonyesha.",
                "Changanua ujumbe huu na uniambie kama ni jaribio la utapeli.",
            ],
            "post": ["Je, ujumbe ulio hapo juu ni wa utapeli? Eleza."],
        },
        "story_prompt": {
            "pre": [
                "Andika hadithi fupi kwa {target}.",
                "Nisimulie hadithi kwa {target}.",
                "Andika hadithi rahisi ya watoto kwa {target}.",
                "Ningependa hadithi fupi kwa {target}, yenye mwanzo, kati na mwisho.",
            ],
            "any": ["Hadithi fupi kwa {target}, tafadhali."],
        },
        "story": {
            "pre": [
                "Andika hadithi fupi kwa {target} kutokana na muhtasari huu:",
                "Simulia hadithi kwa {target} kutokana na wazo lifuatalo:",
                "Geuza muhtasari huu kuwa hadithi fupi kwa {target}:",
            ],
            "post": ["Andika hadithi kwa {target} kutokana na muhtasari ulio hapo juu.", "Sasa simulia hadithi kamili kwa {target}."],
        },
        "synopsis": {
            "pre": ["Andika muhtasari wa sentensi moja wa hadithi ifuatayo, kwa {target}:", "Hadithi hii inahusu nini? Toa muhtasari mfupi kwa {target}."],
            "post": ["Andika muhtasari wa sentensi moja wa hadithi iliyo hapo juu kwa {target}."],
        },
        "refine_translation": {
            "pre": [
                "Hapa kuna maandishi chanzi na rasimu ya tafsiri kwa {target}. Boresha tafsiri hiyo.",
                "Pitia tafsiri hii ya {target} na utoe toleo lililosahihishwa.",
                "Boresha rasimu ya tafsiri iliyo hapa chini ili iwe sahihi na ya asili katika {target}.",
            ],
            "post": ["Boresha rasimu ya tafsiri iliyo hapo juu (kwenda {target}).", "Toa tafsiri iliyosahihishwa ya {target} ya maandishi chanzi yaliyo hapo juu."],
        },
    },
    # ------------------------------------------------------------------------------------------------ Afrikaans
    "afr": {
        "translate": {
            "pre": [
                "Vertaal die volgende teks na {target}.",
                "Vertaal asseblief die gedeelte hieronder na {target}.",
                "Ek wil hê jy moet hierdie teks na {target} vertaal sodat dit natuurlik klink en nie woord vir woord is nie.",
                "Die teks hieronder is in {source}. Gee my net die vertaling in {target}, sonder verduideliking.",
                "Sit die volgende om in {target}:",
                "Jy is 'n professionele vertaler. Vertaal die gedeelte in {source} hieronder na {target} en behou name, getalle en leestekens. Gee slegs die vertaling.",
                "Van {source} na {target}:",
                "Skryf 'n getroue vertaling in {target} van wat volg.",
                "Vertaal na {target} en laat eiename soos dit is:",
                "Kan jy my help om die volgende in {target} te sê?",
            ],
            "post": [
                "Vertaal die teks hierbo na {target}.",
                "Vertaal asseblief wat ek hierbo geskryf het na {target}.",
                "Kan jy dit alles na {target} vertaal?",
                "Die teks hierbo is in {source}. Vertaal dit na {target}.",
                "Gee my die vertaling in {target} van die bostaande, sonder enige kommentaar.",
                "Sit alles hierbo in {target}, asseblief.",
            ],
            "any": [
                "Vertaal dit na {target}.",
                "Ek het 'n vertaling van hierdie teks in {target} nodig.",
                "Sê dit in {target}.",
                "Vertaal na {target}.",
                "{source} → {target}",
            ],
        },
        "summarize": {
            "pre": [
                "Som die volgende teks op.",
                "Som dit in {target} op:",
                "Gee my 'n kort opsomming van die teks hieronder.",
                "Skryf asseblief 'n kort opsomming van die volgende artikel in {target}.",
                "Lees die gedeelte hieronder en som dit in twee of drie sinne op.",
                "Wat is die hoofpunt van die teks hieronder? Som dit in {target} op.",
                "Skryf 'n bondige opsomming in {target} van hierdie nuusberig:",
            ],
            "post": [
                "Som die teks hierbo op.",
                "Gee 'n kort opsomming van die bostaande in {target}.",
                "Som dit asseblief in {target} op.",
                "Kan jy dit in 'n paar sinne saamvat?",
                "Wat is die kernpunte? Antwoord met 'n opsomming in {target}.",
            ],
            "any": ["Som op in {target}.", "'n Opsomming in {target}, asseblief."],
        },
        "headline": {
            "pre": [
                "Skryf 'n opskrif vir die volgende artikel.",
                "Gee hierdie nuusberig 'n titel in {target}:",
                "Stel 'n kort opskrif vir hierdie teks voor.",
                "Wat sou 'n goeie opskrif vir die teks hieronder wees? Antwoord in {target}.",
            ],
            "post": ["Skryf 'n opskrif vir die teks hierbo.", "Gee die artikel hierbo 'n titel in {target}.", "Dink aan 'n pakkende opskrif hiervoor."],
            "any": ["'n Opskrif in {target}, asseblief."],
        },
        "sentiment": {
            "pre": [
                "Wat is die sentiment van die volgende teks? Antwoord Positive, Negative of Neutral.",
                "Klassifiseer die sentiment van hierdie sin:",
                "Is die volgende sin positief, negatief of neutraal?",
                "Doen sentimentontleding op hierdie teks:",
                "Sê vir my die sentiment (Positive, Negative of Neutral):",
                "Lees die sin hieronder en besluit of die skrywer positief, negatief of neutraal voel.",
            ],
            "post": [
                "Wat is die sentiment van die teks hierbo?",
                "Is dit positief, negatief of neutraal? Antwoord Positive, Negative of Neutral.",
                "Gee die sentiment van hierdie teks.",
            ],
        },
        "topic": {
            "pre": [
                "Wat is die onderwerp van hierdie nuusopskrif?",
                "Klassifiseer hierdie opskrif volgens onderwerp:",
                "Aan watter kategorie behoort die volgende teks?",
                "Kies die beste onderwerpetiket vir hierdie teks:",
            ],
            "post": ["Aan watter onderwerp behoort die teks hierbo?", "Watter kategorie is dit?"],
        },
        "emotion": {
            "pre": [
                "Watter emosie druk hierdie teks uit? Kies uit: {labels}.",
                "Klassifiseer die emosie in die volgende teks ({labels}).",
                "Wat voel die skrywer? Antwoord met een van: {labels}.",
                "Emosieherkenning: etiketteer hierdie teks met een van {labels}.",
            ],
            "post": ["Watter emosie word hierbo uitgedruk? Kies uit: {labels}.", "Noem die emosie van die teks hierbo ({labels})."],
        },
        "hate": {
            "pre": [
                "Bevat hierdie teks haatspraak, beledigings of niks daarvan nie? Antwoord Hate, Abuse of Normal.",
                "Klassifiseer hierdie teks as Normal, Abuse of Hate:",
                "Is die volgende boodskap aanstootlik? Etiketteer dit as Normal, Abuse of Hate.",
            ],
            "post": ["Bevat die teks hierbo haatspraak of beledigings?", "Etiketteer die boodskap hierbo: Normal, Abuse of Hate."],
        },
        "intent": {
            "pre": [
                "Wat wil die spreker hê? Noem die bedoeling.",
                "Identifiseer die bedoeling van hierdie versoek:",
                "Klassifiseer die bedoeling van die volgende uitlating:",
                "Wat probeer die gebruiker met hierdie boodskap doen?",
            ],
            "post": ["Wat is die bedoeling van die uitlating hierbo?", "Noem die bedoeling van daardie versoek."],
        },
        "language_id": {
            "pre": [
                "In watter taal is die volgende teks geskryf?",
                "Identifiseer die taal van hierdie teks:",
                "Watter taal is dit? Antwoord met die naam van die taal.",
                "Noem die taal wat in die teks hieronder gebruik word.",
            ],
            "post": ["In watter taal is die teks hierbo geskryf?", "Watter taal is dit?"],
        },
        "qa_context": {
            "pre": [
                "Beantwoord die vraag deur slegs die gedeelte hieronder te gebruik.",
                "Lees die gedeelte en beantwoord die vraag wat volg.",
                "Gebruik die konteks om te antwoord:",
                "Beantwoord die vraag net op grond van die teks hieronder.",
            ],
            "post": ["Beantwoord die vraag deur slegs die konteks hierbo te gebruik.", "Wat is die antwoord volgens die gedeelte hierbo?"],
        },
        "qa_open": {
            "pre": [
                "Beantwoord die volgende vraag in {target}.",
                "Beantwoord asseblief hierdie vraag in {target}:",
                "Gee 'n kort, akkurate antwoord in {target}:",
            ],
            "post": ["Beantwoord die vraag hierbo in {target}.", "Antwoord dit asseblief in {target}."],
        },
        "answer_in": {
            "post": [
                "Gee jou antwoord in {target}.",
                "Antwoord slegs in {target}.",
                "Antwoord asseblief in {target}.",
                "Skryf jou antwoord in {target}.",
                "Jou antwoord moet in {target} wees.",
                "Antwoord net in {target}.",
            ],
        },
        "math": {
            "pre": [
                "Los die volgende probleem stap vir stap op en gee dan die finale antwoord.",
                "Werk deur hierdie wiskundeprobleem en gee die finale antwoord.",
                "Los op:",
                "Wys jou redenering en gee dan die antwoord in {target}.",
            ],
            "post": ["Los die probleem hierbo stap vir stap op.", "Wys jou werk en gee die finale antwoord."],
        },
        "mcq": {
            "pre": [
                "Beantwoord die meervoudigekeusevraag hieronder met die letter van die korrekte opsie.",
                "Kies die korrekte antwoord (A, B, C of D):",
                "Kies die regte opsie vir die volgende vraag.",
            ],
            "post": ["Watter opsie is korrek?", "Antwoord met A, B, C of D."],
        },
        "nli": {
            "pre": [
                "Besluit hoe die tweede sin met die eerste verband hou. Antwoord entailment, neutral of contradiction.",
                "Natuurliketaal-afleiding: etiketteer die paar as entailment, neutral of contradiction.",
                "Lees albei sinne en sê of die hipotese uit die premisse volg (entailment), daarmee bots (contradiction) of onbepaald is (neutral).",
            ],
            "post": ["Wat is die verband tussen hierdie twee sinne? Antwoord entailment, neutral of contradiction."],
        },
        "ner": {
            "pre": [
                "Merk die benoemde entiteite in hierdie sin. Skryf elke woord gevolg deur <tag> en sy etiket (PER, ORG, LOC, DATE of O).",
                "Doen benoemde-entiteitherkenning op die sin hieronder en gee die uitset as pare in die vorm woord <tag> ETIKET.",
            ],
        },
        "pos": {
            "pre": [
                "Gee die woordsoortetiket van elke woord in hierdie sin as pare in die vorm woord <tag> ETIKET.",
                "Etiketteer die woordsoorte van die volgende sin: skryf elke woord gevolg deur <tag> en sy etiket.",
            ],
            "post": ["Gee die woordsoort van elke woord in die sin hierbo (woord <tag> ETIKET)."],
        },
        "define_term": {
            "pre": [
                "Wat beteken \"{title}\"? Verduidelik dit in {target}.",
                "Gee die betekenis van die woord \"{title}\" in {target}.",
                "Definieer \"{title}\" ({target}).",
                "Ek het die term \"{title}\" raakgeloop. Wat beteken dit?",
            ],
            "any": ["\"{title}\" - betekenis in {target}?"],
        },
        "wiki_ask": {
            "pre": [
                "Vertel my van {title}. Gee jou antwoord slegs in {target}.",
                "Wie of wat is {title}? Antwoord in {target}.",
                "Skryf 'n ensiklopedie-artikel oor {title} in {target}.",
                "Verduidelik {title} vir my in {target}.",
                "Wat weet jy van {title}? Antwoord in {target}.",
            ],
            "any": ["{title} - gee my die feite, in {target}."],
        },
        "scam": {
            "pre": [
                "Is hierdie boodskap 'n bedrogskema? Antwoord Scam of Not scam en verduidelik dan in een of twee sinne hoekom.",
                "Kyk of die volgende boodskap bedrieglik is en sê wat dit verklap.",
                "Ontleed hierdie boodskap en sê my of dit 'n bedrogpoging is.",
            ],
            "post": ["Is die boodskap hierbo 'n bedrogskema? Verduidelik."],
        },
        "story_prompt": {
            "pre": [
                "Skryf 'n kort storie in {target}.",
                "Vertel my 'n storie in {target}.",
                "Skryf 'n eenvoudige kinderstorie in {target}.",
                "Ek wil graag 'n kort storie in {target} hê, met 'n begin, middel en einde.",
            ],
            "any": ["'n Kort storie in {target}, asseblief."],
        },
        "story": {
            "pre": [
                "Skryf 'n kort storie in {target} gebaseer op hierdie sinopsis:",
                "Vertel 'n storie in {target} gebaseer op die volgende idee:",
                "Verander hierdie sinopsis in 'n kort storie in {target}:",
            ],
            "post": ["Skryf 'n storie in {target} gebaseer op die sinopsis hierbo.", "Vertel nou die volledige storie in {target}."],
        },
        "synopsis": {
            "pre": ["Skryf 'n eensin-sinopsis van die volgende storie, in {target}:", "Waaroor gaan hierdie storie? Gee 'n kort sinopsis in {target}."],
            "post": ["Skryf 'n eensin-sinopsis van die storie hierbo in {target}."],
        },
        "refine_translation": {
            "pre": [
                "Hier is 'n brontekst en 'n konsepvertaling in {target}. Verbeter die vertaling.",
                "Proeflees hierdie vertaling in {target} en gee 'n verbeterde weergawe.",
                "Verfyn die konsepvertaling hieronder sodat dit akkuraat en natuurlik in {target} is.",
            ],
            "post": ["Verbeter die konsepvertaling hierbo (na {target}).", "Gee 'n verbeterde vertaling in {target} van die brontekst hierbo."],
        },
    },
}
