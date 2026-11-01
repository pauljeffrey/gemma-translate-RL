"""Bigger pools for the languages that already had instruction templates (eng, pcm, yor, hau, ibo) plus REPLACE, which corrects existing templates.

REPLACE[lang][task] = {old template: new template}.  Reason (found when the examples were reviewed, 2026-10-11): the existing yor / hau / ibo sentiment and
hate pools named the answer labels in local words (Rere / Búburú / Aláìṣègbè, Tabbatacce / Korau / Tsaka-tsaki, Ọma / Njọ / Ọnọdụ etiti, Déédé / Ìbàjẹ́ /
Ìkórìíra, ...) while every answer in the data is the English label (Positive / Negative / Neutral, Normal / Abuse / Hate).  The instruction must name the
labels the answer will use.
"""
TIER: dict = {}
NAMES: dict = {}

REPLACE = {
    "yor": {
        "sentiment": {
            "Kí ni ìmọ̀lára tó wà nínú gbólóhùn yìí? Rere, Búburú tàbí Aláìṣègbè:": "Kí ni ìmọ̀lára tó wà nínú gbólóhùn yìí? Positive, Negative tàbí Neutral:",
        },
        "hate": {
            "Pín ọ̀rọ̀ yìí sí Déédé, Ìbàjẹ́ tàbí Ìkórìíra:": "Pín ọ̀rọ̀ yìí sí Normal, Abuse tàbí Hate:",
        },
    },
    "hau": {
        "sentiment": {
            "Menene yanayin ra'ayin wannan jimla? Tabbatacce, Korau ko Tsaka-tsaki:": "Menene yanayin ra'ayin wannan jimla? Positive, Negative ko Neutral:",
        },
        "hate": {
            "Rarraba wannan saƙon: Na al'ada, Cin zarafi ko Ƙiyayya:": "Rarraba wannan saƙon: Normal, Abuse ko Hate:",
        },
    },
    "ibo": {
        "language_id": {
            "Asụsụ ole ka e ji dee ederede a?": "Kedu asụsụ e ji dee ederede a?",
            "Asụsụ ole bụ ederede dị n'elu?": "Kedu asụsụ e ji dee ederede dị n'elu?",
        },
        "nli": {
            "Lee mmekọrịta dị n'etiti ahịrịokwu abụọ a. Zaa na entailment, neutral ma ọ bụ contradiction.":
                "Lee mmekọrịta dị n'etiti ahịrịokwu abụọ a. Jiri entailment, neutral ma ọ bụ contradiction zaa.",
            "Ahịrịokwu nke abụọ ọ sochiri nke mbụ? Zaa na entailment, neutral ma ọ bụ contradiction:":
                "Ahịrịokwu nke abụọ ọ sochiri nke mbụ? Jiri entailment, neutral ma ọ bụ contradiction zaa:",
        },
        "sentiment": {
            "Kedu mmetụta dị n'ahịrịokwu a? Ọma, Njọ ma ọ bụ Ọnọdụ etiti:": "Kedu mmetụta dị n'ahịrịokwu a? Positive, Negative ma ọ bụ Neutral:",
        },
        "hate": {
            "Kewaa ozi a: Nkịtị, Mkparị ma ọ bụ Ịkpọasị:": "Kewaa ozi a: Normal, Abuse ma ọ bụ Hate:",
        },
    },
}

def _merge(extra: dict) -> None:
    for lang, tasks in extra.items():
        for task, mapping in tasks.items():
            REPLACE.setdefault(lang, {}).setdefault(task, {}).update(mapping)


# second review pass (pre-existing templates of pcm / yor / hau / ibo / twi / ewe, 2026-10-11)
_merge({
    "hau": {
        "summarize": {
            "Takaita wannan rubutun:": "Taƙaita wannan rubutun:",
            "Ka wannan labarin ka takaita shi a harshen {target}:": "Ka karanta wannan labarin ka taƙaita shi a harshen {target}:",
            "Takaita rubutun da ke sama.": "Taƙaita rubutun da ke sama.",
        },
        "headline": {"Ba wannan labarin taken kanun labari a {target}:": "Ba wannan labarin take a {target}:"},
        "topic": {"Wane rukuni ne rubutun da ke sama ya ke ciki?": "Wane rukuni ne rubutun da ke sama yake ciki?"},
        "scam": {"Shin wannan saƙo zamba ne? Bayyana dalilin:": "Shin wannan saƙo zamba ne? Bayyana dalilin hakan:"},
        "translate": {
            "Ina so ka fassara rubutun da ke ƙasa zuwa {target}, ta yadda zai yi daidai kuma ya fito a zahiri, ba kalma-da-kalma kawai ba.":
                "Ina so ka fassara rubutun da ke ƙasa zuwa {target}, ta yadda zai yi daidai kuma ya yi daɗin karantawa kamar yadda masu harshen ke magana, ba kalma-da-kalma kawai ba.",
        },
    },
    "yor": {
        "summarize": {"Fún mi ní àkótán kúkúrú ti àyọkà yìí:": "Fún mi ní àkótán kúkúrú àyọkà yìí:"},
        "hate": {"Ṣé ọ̀rọ̀ ìkórìíra tàbí ìwà ìbàjẹ́ wà nínú ọ̀rọ̀ yìí?": "Ṣé ọ̀rọ̀ ìkórìíra tàbí ọ̀rọ̀ èébú wà nínú ọ̀rọ̀ yìí? Dáhùn pẹ̀lú Hate, Abuse tàbí Normal."},
        "refine_translation": {
            "Èyí ni ọ̀rọ̀ orísun àti ìtúmọ̀ ìgbékalẹ̀ rẹ̀ sí {target}. Mú ìtúmọ̀ náà dára síi.":
                "Èyí ni ọ̀rọ̀ orísun àti ìtumọ̀ àkọ́kọ́ rẹ̀ sí {target}. Mú ìtumọ̀ náà dára sí i.",
        },
        "translate": {
            "Jọ̀wọ́, túmọ̀ àyọkà òkè yìí sí èdè {target}.": "Jọ̀wọ́, túmọ̀ àyọkà tó wà lókè yìí sí èdè {target}.",
            "Ọ̀rọ̀ tí mo kọ sókè wà ní èdè {source}; túmọ̀ rẹ̀ sí èdè {target} lọ́nà tí ó yé ni dáadáa.":
                "Ọ̀rọ̀ tí mo kọ lókè yìí wà ní èdè {source}; túmọ̀ rẹ̀ sí èdè {target} lọ́nà tí ó yé ni dáadáa.",
        },
    },
    "ibo": {
        "topic": {"Kedu ngalaba ederede dị n'elu bụ?": "Ederede dị n'elu bụ nke ngalaba kedu?"},
        "answer_in": {"Nyefee azịza gị n'asụsụ {target} naanị.": "Nye azịza gị n'asụsụ {target} naanị."},
        "math": {
            "Dozie nsogbu mgbakọ a nzọụkwụ site na nzọụkwụ:": "Dozie nsogbu mgbakọ na mwepụ a nzọụkwụ site na nzọụkwụ:",
            "Mee mgbakọ a wee nye azịza ikpeazụ:": "Mee mgbakọ na mwepụ a wee nye azịza ikpeazụ:",
        },
        "diacritize": {"Tinye akara ụda na akara ndị ọzọ efu n'ederede {target} a:": "Tinye akara ụda na akara ndị ọzọ furu efu n'ederede {target} a:"},
        "denoise": {
            "Megharịa ederede {target} a nwere njehie:": "Dozie ederede {target} a nwere njehie:",
            "Wepụ mkpụrụedemede ndị ọjọọ n'ederede a:": "Wepụ mkpụrụedemede ndị na-ezighi ezi n'ederede a:",
            "Megharịa ederede dị n'elu.": "Dozie ederede dị n'elu.",
        },
        "scam": {
            "Ọ bụ aghụghọ ka ozi a? Kọwaa ihe kpatara ya:": "Ozi a ọ bụ aghụghọ? Kọwaa ihe kpatara ya:",
            "Lelee ozi a ma ọ bụ aghụghọ:": "Lelee ozi a nke ọma: ọ bụ aghụghọ ka ọ bụghị?",
        },
        "define_term": {"Gịnị bụ ihe \"{title}\" pụtara? Kọwaa ya n'asụsụ {target}.": "Gịnị ka \"{title}\" pụtara? Kọwaa ya n'asụsụ {target}."},
        "translate": {
            "Ị nwere ike inyere m aka sụgharịa ihe dị n'okpuru a gaa n'asụsụ {target}?": "Ị nwere ike inyere m aka isụgharị ihe dị n'okpuru a gaa n'asụsụ {target}?",
            "Ederede a bụ n'asụsụ {source}. Achọrọ m nsụgharị ya n'asụsụ {target}; dee naanị nsụgharị ahụ, agbakwunyela nkọwa ọ bụla.":
                "Ederede a dị n'asụsụ {source}. Achọrọ m nsụgharị ya n'asụsụ {target}; dee naanị nsụgharị ahụ, agbakwunyela nkọwa ọ bụla.",
            "Dee ya n'asụsụ {target}.": "Sụgharịa ya gaa n'asụsụ {target}.",
        },
    },
    "pcm": {
        "sentiment": {"Dis text good, bad or e just dey normal? Tell me:": "Dis text na Positive, Negative or Neutral? Tell me:"},
        "wiki_ask": {"Who be {title}? Explain am for {target}.": "Who or wetin be {title}? Explain am for {target}."},
    },
    "twi": {"translate": {"Mesrɛ wo, kyerɛ nea metwerɛe wɔ atifi no ase wɔ {target} mu.": "Mesrɛ wo, kyerɛ nea mekyerɛwee wɔ soro no ase wɔ {target} mu."}},
    "ewe": {
        "translate": {
            "Ɖe nya siwo le ete lae gɔme ɖe {target} me.": "Ɖe nya siwo le ete la gɔme ɖe {target} me.",
            "Ɖe nya siwo le dziŋ gɔme ɖe {target} me.": "Ɖe nya siwo le dzi la gɔme ɖe {target} me.",
        },
    },
})

BANKS = {
    # ------------------------------------------------------------------------------------------------ English
    "eng": {
        "translate": {
            "pre": [
                "Translate the passage below into {target}. The result should read as if it had been written in {target} originally.",
                "Please give me a {target} version of this text. Preserve the paragraph breaks.",
                "I need this in {target}. Translate it accurately and do not summarise.",
                "Translate the following {source} text into {target}. If you meet a proper noun, leave it as it is.",
                "Act as a translator and convert the text below into {target}.",
                "Translate this into {target} so that a ten-year-old could understand it.",
                "Write the {target} equivalent of the text below.",
                "{target} please. Here is the text:",
            ],
            "post": [
                "Translate the text above into {target} without changing its meaning.",
                "Take the passage above and give me its {target} translation only.",
                "Please translate everything above into {target}; keep numbers and names as they are.",
                "What is the {target} translation of the above?",
            ],
            "any": ["Translate to {target}, please.", "Give me the {target} translation.", "Put this into {target}."],
        },
        "summarize": {
            "pre": [
                "Write a three-sentence summary of the following text in {target}.",
                "Summarise the article below for a busy reader. Reply in {target}.",
                "What happened in the following story? Summarise it briefly in {target}.",
                "Boil the text below down to its essentials, in {target}.",
                "Provide an abstract of the following passage in {target}.",
                "Give a one-paragraph summary of this text.",
            ],
            "post": [
                "Summarise the above in {target}, in no more than three sentences.",
                "Give the gist of the text above in {target}.",
                "Please write a short abstract of the passage above.",
            ],
            "any": ["Short summary in {target}, please.", "Summarise this."],
        },
        "headline": {
            "pre": [
                "Come up with a headline in {target} for the following news item.",
                "Write a short, clear title for this article in {target}:",
                "Which headline would suit this story? Answer in {target}.",
            ],
            "post": ["Give the news item above a short headline in {target}.", "Title the article above in {target}."],
        },
        "sentiment": {
            "pre": [
                "Is the following text positive, negative or neutral? Reply with Positive, Negative or Neutral.",
                "Determine the sentiment of the sentence below (Positive, Negative or Neutral):",
                "How does the author feel about the subject? Answer Positive, Negative or Neutral.",
                "Sentiment classification. Label the text as Positive, Negative or Neutral:",
            ],
            "post": [
                "What is the overall sentiment of the text above? Positive, Negative or Neutral?",
                "Label the sentiment of the sentence above: Positive, Negative or Neutral.",
            ],
        },
        "topic": {
            "pre": [
                "Which topic does the following text cover?",
                "Assign this text to a topic category:",
                "What is this news item about? Give the topic label.",
                "Topic classification: name the category of the text below.",
            ],
            "post": ["Give the topic label for the text above.", "What subject area does the text above belong to?"],
        },
        "emotion": {
            "pre": [
                "Identify the main emotion in the text below. Options: {labels}.",
                "Which of these emotions is expressed: {labels}?",
                "Read the text and name the emotion it conveys ({labels}).",
            ],
            "post": ["Which emotion does the text above convey? Options: {labels}."],
        },
        "hate": {
            "pre": [
                "Is this message hate speech, abusive, or normal? Answer Hate, Abuse or Normal.",
                "Moderate the following post and label it Normal, Abuse or Hate:",
                "Does the text below attack a person or group? Label it Normal, Abuse or Hate.",
            ],
            "post": ["Should the message above be labelled Normal, Abuse or Hate?"],
        },
        "intent": {
            "pre": [
                "What is the user asking for in the following message? Name the intent.",
                "Determine the intent behind this utterance:",
                "Intent detection: which goal does the speaker have?",
            ],
            "post": ["Name the intent behind the utterance above."],
        },
        "language_id": {
            "pre": [
                "Identify the language the text below is written in.",
                "In which language is this passage? Just name the language.",
                "Tell me which language this text is in:",
            ],
            "post": ["Name the language of the passage above.", "Which language is the text above in?"],
        },
        "qa_context": {
            "pre": [
                "Using only the information in the passage below, answer the question that follows.",
                "Here is a passage followed by a question. Answer from the passage alone.",
                "Read the context carefully and then answer the question.",
            ],
            "post": ["Answer the question from the passage above only.", "Using the context above, what is the answer?"],
        },
        "qa_open": {
            "pre": [
                "Answer the question below accurately and concisely, in {target}.",
                "Give a clear answer in {target} to this question:",
                "Please respond to the following question in {target}.",
            ],
            "post": ["Respond to the question above in {target}.", "Answer the above in {target}, briefly."],
        },
        "answer_in": {
            "post": [
                "Write your reply in {target}.",
                "Use {target} for your answer.",
                "Your answer should be in {target}.",
                "Respond using {target} only.",
            ],
        },
        "math": {
            "pre": [
                "Solve this problem. Show each step and finish with the final answer.",
                "Work out the answer to the following word problem step by step:",
                "Calculate the answer and explain your reasoning briefly:",
            ],
            "post": ["Work this out step by step and state the final answer.", "Solve the problem above, showing your calculation."],
        },
        "mcq": {
            "pre": [
                "Read the question and choose the correct option. Answer with its letter.",
                "Multiple choice: select the best answer (A, B, C or D).",
                "Choose the right answer from the options given:",
            ],
            "post": ["Which option is the correct one? Give the letter.", "Answer with the letter of the correct option."],
        },
        "nli": {
            "pre": [
                "Given the premise and the hypothesis, is the relationship entailment, neutral or contradiction?",
                "Does the first sentence imply the second? Answer entailment, neutral or contradiction.",
            ],
            "post": ["Does the second sentence follow from the first? Answer entailment, neutral or contradiction."],
        },
        "ner": {
            "pre": [
                "Find the people, organisations, locations and dates in this sentence. Write each word followed by <tag> and its label (PER, ORG, LOC, DATE or O).",
                "Named-entity tagging: output every word of the sentence below with <tag> and its label (PER, ORG, LOC, DATE or O).",
            ],
        },
        "define_term": {
            "pre": [
                "What is the meaning of \"{title}\"? Please answer in {target}.",
                "Explain the term \"{title}\" in {target}.",
                "Give a short definition of \"{title}\" in {target}.",
            ],
        },
        "wiki_ask": {
            "pre": [
                "Write an informative overview of {title} in {target}.",
                "What should I know about {title}? Please answer in {target}.",
                "Describe {title} in {target}, as an encyclopedia would.",
            ],
        },
        "scam": {
            "pre": [
                "Could this message be a scam? Answer Scam or Not scam and list the red flags.",
                "Read the message below and decide whether it is fraudulent. Explain briefly.",
            ],
            "post": ["Is the message above fraudulent? Answer Scam or Not scam and explain."],
        },
        "story_prompt": {
            "pre": [
                "Write a short story in {target} that a child could enjoy.",
                "Tell me a story in {target} with a clear beginning, middle and end.",
                "Please write a short folktale-style story in {target}.",
            ],
        },
        "story": {
            "pre": [
                "Here is a story idea. Write the full story in {target}:",
                "Develop this synopsis into a complete short story in {target}:",
            ],
        },
        "synopsis": {
            "pre": ["Summarise the plot of the following story in one sentence, in {target}:", "In {target}, what is the main idea of this story?"],
        },
        "refine_translation": {
            "pre": [
                "Review the draft translation into {target} below and correct any mistakes.",
                "Polish this {target} translation so that it matches the source exactly and reads naturally.",
            ],
            "post": ["Correct the draft {target} translation above."],
        },
        "assess_translation": {
            "pre": [
                "Read the source text and its {target} translation. Is the translation faithful and fluent? List any errors.",
                "Check this {target} translation against the source and report mistranslations, omissions and additions.",
            ],
            "post": ["Review the {target} translation above and list the problems you find."],
        },
        "offensive": {
            "pre": [
                "Does the text below contain offensive language? Answer Offensive or Not offensive.",
                "Label this text as Offensive or Not offensive:",
            ],
            "post": ["Label the text above as Offensive or Not offensive."],
        },
        "safety": {
            "pre": [
                "Rate how safe this message is: Safe, Review or Unsafe.",
                "Classify the safety of the following text as Safe, Review or Unsafe:",
            ],
            "post": ["Is the message above Safe, Review or Unsafe?"],
        },
        "rerank": {
            "pre": [
                "Order the passages below from most to least relevant to the query. Reply with their identifiers only.",
                "Which passage best answers the query? Rank all of them, best first.",
            ],
            "post": ["Rank the passages above by relevance to the query, most relevant first."],
        },
    },
    # ------------------------------------------------------------------------------------------------ Nigerian Pidgin
    "pcm": {
        "translate": {
            "pre": [
                "Abeg translate dis text go {target}, but make e no be word-for-word.",
                "I wan read dis one for {target}. Abeg translate am for me:",
                "Change dis text go {target}. Make you keep di names as dem be.",
                "Dis text na {source}. Put am for {target}, no add extra talk.",
            ],
            "post": [
                "Translate everything wey dey up there go {target}, abeg.",
                "Make you put wetin dey up there for {target}.",
                "I need di {target} version of di text wey dey up there.",
            ],
            "any": ["Translate am go {target} for me.", "Abeg, give me di {target} translation of dis one."],
        },
        "summarize": {
            "pre": [
                "Abeg tell me di main gist of dis text, make e short, for {target}:",
                "Cut dis story come short for {target}:",
                "Summarize dis text for me, make e no too long:",
            ],
            "post": ["Abeg summarize di text wey dey up there for {target}, make e short.", "Give me di gist of wetin dey up there."],
        },
        "headline": {
            "pre": ["Abeg give dis story one short headline for {target}:", "Wetin be di best headline for dis news? Talk am for {target}."],
            "post": ["Give di news wey dey up there headline for {target}."],
        },
        "sentiment": {
            "pre": [
                "Abeg tell me di sentiment of dis text (Positive, Negative or Neutral):",
                "Di person wey write dis one dey happy, dey vex, or e no dey show any feeling? Answer Positive, Negative or Neutral:",
            ],
            "post": ["Wetin be di sentiment of di text wey dey up there? Positive, Negative or Neutral?"],
        },
        "topic": {
            "pre": ["Which category dis text belong to?", "Pick di topic wey fit dis text:"],
            "post": ["Wetin be di topic of di text wey dey up there?"],
        },
        "emotion": {
            "pre": ["Abeg tell me di emotion wey dey inside dis text: {labels}.", "Which one for dis emotions dey inside dis text? {labels}."],
            "post": ["Which emotion dey inside di text wey dey up there? Choose from: {labels}."],
        },
        "hate": {
            "pre": ["Dis message na Normal, Abuse or Hate? Abeg talk:", "Check dis text well and tell me whether na Normal, Abuse or Hate:"],
            "post": ["Di message wey dey up there na Normal, Abuse or Hate?"],
        },
        "intent": {
            "pre": ["Wetin di person dey try do wit dis message?", "Abeg tell me di intent for dis talk:"],
            "post": ["Wetin be di intent for di message wey dey up there?"],
        },
        "language_id": {
            "pre": ["Abeg tell me which language dem write dis text for:", "Dis text na which language?"],
            "post": ["Di text wey dey up there na which language?"],
        },
        "qa_context": {
            "pre": ["Read dis passage well, den answer di question wey follow:", "Use only wetin dey inside dis passage answer di question:"],
            "post": ["Answer di question, use only wetin dey up there."],
        },
        "qa_open": {
            "pre": ["Abeg answer di question wey dey below for {target}:", "Give me short, correct answer for {target}:"],
            "post": ["Abeg answer for {target}.", "Answer di question wey dey up there, use {target}."],
        },
        "answer_in": {"post": ["Use {target} answer me.", "Make your answer dey for {target}.", "Reply me for {target} only."]},
        "math": {
            "pre": ["Solve dis maths wahala step by step den give di final answer:", "Show how you take solve am, den tell me di answer:"],
            "post": ["Show your working den give di final answer."],
        },
        "mcq": {
            "pre": ["Choose di correct option (A, B, C or D):", "Pick di right answer for dis question:"],
            "post": ["Answer wit A, B, C or D."],
        },
        "nli": {
            "pre": ["Di first sentence dey support di second one, e dey fight am, or e no get connection? Answer: entailment, neutral or contradiction."],
        },
    },
    # ------------------------------------------------------------------------------------------------ Yoruba
    "yor": {
        "translate": {
            "pre": [
                "Jọ̀wọ́ túmọ̀ gbólóhùn yìí sí èdè {target}, kí o má sì yí orúkọ àwọn ènìyàn padà.",
                "Èdè {target} ni mo fẹ́ kí o fi kọ ìtumọ̀ ọ̀rọ̀ yìí:",
                "Túmọ̀ àyọkà tó wà nísàlẹ̀ yìí sí èdè {target}, kí ìtumọ̀ náà péye.",
            ],
            "post": [
                "Jọ̀wọ́ túmọ̀ gbogbo ohun tó wà lókè yìí sí èdè {target}.",
                "Fún mi ní ìtumọ̀ èdè {target} ti ọ̀rọ̀ tó wà lókè yìí.",
            ],
        },
        "summarize": {
            "pre": [
                "Ṣe àkótán ọ̀rọ̀ tó wà nísàlẹ̀ yìí ní èdè {target}:",
                "Kí ni èrò pàtàkì ọ̀rọ̀ yìí? Ṣe àkótán rẹ̀ ní èdè {target}.",
            ],
            "post": ["Jọ̀wọ́ ṣe àkótán ohun tó wà lókè yìí ní èdè {target}."],
        },
        "headline": {
            "pre": [
                "Kọ àkọ́lé kúkúrú fún àyọkà yìí ní èdè {target}:",
                "Àkọ́lé wo ni ó yẹ fún ọ̀rọ̀ tó wà nísàlẹ̀ yìí? Dáhùn ní èdè {target}.",
            ],
            "post": ["Fún ìròyìn tó wà lókè yìí ní àkọ́lé ní èdè {target}."],
        },
        "sentiment": {
            "pre": [
                "Ṣe ìtúpalẹ̀ ìmọ̀lára gbólóhùn yìí. Dáhùn pẹ̀lú Positive, Negative tàbí Neutral:",
                "Ṣé gbólóhùn yìí jẹ́ rere, búburú tàbí aláìṣègbè? Dáhùn pẹ̀lú Positive, Negative tàbí Neutral:",
            ],
            "post": ["Kí ni ìmọ̀lára ọ̀rọ̀ tó wà lókè yìí? Positive, Negative tàbí Neutral?"],
        },
        "topic": {
            "pre": ["Ẹ̀ka wo ni ọ̀rọ̀ yìí wà nínú rẹ̀?", "Yan ẹ̀ka tó bá ọ̀rọ̀ yìí mu:"],
            "post": ["Kí ni kókó ọ̀rọ̀ tí ohun tó wà lókè yìí dá lé?"],
        },
        "language_id": {
            "pre": ["Èdè wo ni a kọ ọ̀rọ̀ tó wà nísàlẹ̀ yìí sí?", "Èdè wo ni èyí? Dáhùn pẹ̀lú orúkọ èdè náà."],
            "post": ["Èdè wo ni a kọ ọ̀rọ̀ tó wà lókè yìí sí?"],
        },
        "qa_context": {
            "pre": ["Ka àyọkà yìí, lẹ́yìn náà dáhùn ìbéèrè tó tẹ̀ lé e.", "Lo àyọkà tó wà nísàlẹ̀ nìkan láti dáhùn ìbéèrè náà."],
            "post": ["Dáhùn ìbéèrè náà nípa lílo àyọkà tó wà lókè nìkan."],
        },
        "qa_open": {
            "pre": ["Jọ̀wọ́ dáhùn ìbéèrè tó tẹ̀ lé e ní èdè {target}.", "Fún mi ní ìdáhùn kúkúrú àti pípéye ní èdè {target}:"],
            "post": ["Jọ̀wọ́ dáhùn ní èdè {target}."],
        },
        "answer_in": {"post": ["Fi èdè {target} nìkan dáhùn.", "Ìdáhùn rẹ gbọ́dọ̀ wà ní èdè {target}.", "Kọ ìdáhùn rẹ ní èdè {target}."]},
        "math": {
            "pre": [
                "Yanjú ìṣòro tó tẹ̀ lé e ní ìgbésẹ̀ ní ìgbésẹ̀, lẹ́yìn náà sọ èsì ìkẹyìn.",
                "Ṣe àfihàn bí o ṣe ṣe é, kí o sì fún mi ní èsì ìkẹyìn:",
            ],
            "post": ["Ṣe àfihàn iṣẹ́ rẹ kí o sì fún mi ní èsì ìkẹyìn."],
        },
        "mcq": {
            "pre": ["Dáhùn ìbéèrè onírúurú àṣàyàn tó wà nísàlẹ̀ pẹ̀lú lẹ́tà àṣàyàn tó tọ́.", "Yan àṣàyàn tó tọ́ fún ìbéèrè yìí."],
            "post": ["Dáhùn pẹ̀lú A, B, C tàbí D."],
        },
    },
    # ------------------------------------------------------------------------------------------------ Hausa
    "hau": {
        "translate": {
            "pre": [
                "Don Allah, fassara wannan jimla zuwa {target}, kuma kada ka canza sunayen mutane.",
                "Fassara rubutun da ke ƙasa zuwa {target} daidai gwargwado.",
                "Ina so a fassara min wannan rubutu zuwa {target} cikin harshe mai sauƙi.",
            ],
            "post": [
                "Don Allah, fassara duk abin da ke sama zuwa {target}.",
                "Ba ni fassarar {target} ta rubutun da ke sama.",
            ],
        },
        "summarize": {
            "pre": [
                "Ka karanta wannan rubutu ka taƙaita shi a jimloli biyu ko uku.",
                "Menene babban abin da wannan rubutu ya ƙunsa? Taƙaita shi a {target}.",
            ],
            "post": ["Don Allah, taƙaita abin da ke sama a {target}."],
        },
        "headline": {
            "pre": [
                "Rubuta gajeren kanun labari na wannan rubutu a {target}:",
                "Wane kanun labari ya dace da rubutun da ke ƙasa? Amsa a {target}.",
            ],
            "post": ["Ba da kanun labari ga rubutun da ke sama a {target}."],
        },
        "sentiment": {
            "pre": [
                "Yi nazarin ra'ayin wannan jimla. Amsa da Positive, Negative ko Neutral:",
                "Shin wannan jimla tana nuna ra'ayi mai kyau, mara kyau ko tsaka-tsaki? Amsa da Positive, Negative ko Neutral:",
            ],
            "post": ["Menene ra'ayin rubutun da ke sama? Positive, Negative ko Neutral?"],
        },
        "topic": {
            "pre": ["A wane rukuni wannan rubutu yake?", "Zaɓi rukunin da ya dace da wannan rubutu:"],
            "post": ["Rubutun da ke sama na wane rukuni ne?"],
        },
        "language_id": {
            "pre": ["A wane harshe aka rubuta rubutun da ke ƙasa?", "Wane harshe ne wannan? Amsa da sunan harshen."],
            "post": ["A wane harshe aka rubuta rubutun da ke sama?"],
        },
        "qa_context": {
            "pre": ["Karanta wannan rubutu sannan ka amsa tambayar da ke biye.", "Yi amfani da rubutun da ke ƙasa kawai ka amsa tambayar."],
            "post": ["Amsa tambayar ta amfani da rubutun da ke sama kawai."],
        },
        "qa_open": {
            "pre": ["Don Allah ka amsa tambayar da ke biye a {target}.", "Ba da gajeriyar amsa mai inganci a {target}:"],
            "post": ["Don Allah ka amsa a {target}."],
        },
        "answer_in": {"post": ["Ka yi amfani da {target} kawai wajen amsawa.", "Dole amsarka ta kasance a {target}.", "Rubuta amsarka a {target}."]},
        "math": {
            "pre": [
                "Warware matsalar da ke biye mataki-mataki, sannan ka bayar da amsar ƙarshe.",
                "Nuna yadda ka yi lissafin, sannan ka bayar da amsar ƙarshe:",
            ],
            "post": ["Nuna aikinka ka bayar da amsar ƙarshe."],
        },
        "mcq": {
            "pre": ["Amsa tambayar zaɓin amsa da ke ƙasa da harafin zaɓin da ya dace.", "Zaɓi zaɓin da ya dace da wannan tambaya."],
            "post": ["Amsa da A, B, C ko D."],
        },
    },
    # ------------------------------------------------------------------------------------------------ Igbo
    "ibo": {
        "translate": {
            "pre": [
                "Biko, sụgharịa ahịrịokwu a gaa n'asụsụ {target}, ma hapụ aha ndị mmadụ ka ha dị.",
                "Sụgharịa ederede dị n'okpuru gaa n'asụsụ {target} n'ụzọ ziri ezi.",
                "Achọrọ m ka ị sụgharịa ederede a gaa n'asụsụ {target} site n'iji asụsụ dị mfe.",
            ],
            "post": [
                "Biko, sụgharịa ihe niile dị n'elu gaa n'asụsụ {target}.",
                "Nye m nsụgharị {target} nke ederede dị n'elu.",
            ],
        },
        "summarize": {
            "pre": [
                "Gụọ ederede a ma chịkọta ya n'ahịrịokwu abụọ ma ọ bụ atọ.",
                "Kedu isi okwu nke ederede a? Chịkọta ya n'asụsụ {target}.",
            ],
            "post": ["Biko, chịkọta ihe dị n'elu n'asụsụ {target}."],
        },
        "headline": {
            "pre": [
                "Dee obere isiokwu maka ederede a n'asụsụ {target}:",
                "Kedu isiokwu kwesịrị ederede dị n'okpuru? Zaa n'asụsụ {target}.",
            ],
            "post": ["Dee isiokwu maka akụkọ dị n'elu n'asụsụ {target}."],
        },
        "sentiment": {
            "pre": [
                "Nyochaa mmetụta nke ahịrịokwu a. Jiri Positive, Negative ma ọ bụ Neutral zaa:",
                "Ahịrịokwu a ọ dị mma, ọ dị njọ, ka ọ bụ nke etiti? Jiri Positive, Negative ma ọ bụ Neutral zaa:",
            ],
            "post": ["Kedu mmetụta nke ederede dị n'elu? Positive, Negative ma ọ bụ Neutral?"],
        },
        "topic": {
            "pre": ["Ederede a bụ nke ngalaba kedu?", "Họrọ ngalaba kwesịrị ederede a:"],
            "post": ["Ederede dị n'elu bụ nke ngalaba kedu?"],
        },
        "language_id": {
            "pre": ["N'asụsụ kedu ka e dere ederede dị n'okpuru?", "Kedu asụsụ bụ nke a? Zaa site n'ịkpọ aha asụsụ ahụ."],
            "post": ["N'asụsụ kedu ka e dere ederede dị n'elu?"],
        },
        "qa_context": {
            "pre": ["Gụọ ederede a wee zaa ajụjụ na-esote.", "Jiri ederede dị n'okpuru naanị zaa ajụjụ ahụ."],
            "post": ["Jiri ederede dị n'elu naanị zaa ajụjụ ahụ."],
        },
        "qa_open": {
            "pre": ["Biko, zaa ajụjụ na-esote n'asụsụ {target}.", "Nye azịza dị mkpụmkpụ ma ziri ezi n'asụsụ {target}:"],
            "post": ["Biko, zaa n'asụsụ {target}."],
        },
        "answer_in": {"post": ["Jiri asụsụ {target} naanị zaa.", "Azịza gị ga-adị n'asụsụ {target}.", "Dee azịza gị n'asụsụ {target}."]},
        "math": {
            "pre": [
                "Dozie nsogbu na-esote nzọụkwụ site na nzọụkwụ, wee nye azịza ikpeazụ.",
                "Gosi etu i si mee ya, wee nye azịza ikpeazụ:",
            ],
            "post": ["Gosi ọrụ gị wee nye azịza ikpeazụ."],
        },
        "mcq": {
            "pre": ["Zaa ajụjụ nhọrọ dị n'okpuru site na mkpụrụedemede nhọrọ ziri ezi.", "Họrọ nhọrọ ziri ezi maka ajụjụ a."],
            "post": ["Jiri A, B, C ma ọ bụ D zaa."],
        },
    },
}
