"""Instruction pools for the tasks added with the second round of datasets (same shape and rules as templates.py).

Labels stay English words in every instruction language (Positive/Negative, entailment/neutral/contradiction, ...), exactly as the existing
sentiment / hate / intent pools do, so the label vocabulary of a task is one set whatever language the request is written in.
Slots: {target} language the answer must be in, {labels} the label set shown to the model, {n} a number.
"""

from __future__ import annotations

from sft_new.templates import _reg

# ---------------------------------------------------------------- natural language inference (<nli>)
_reg("nli", "eng",
     pre=["Decide how the second sentence relates to the first. Answer entailment, neutral or contradiction.",
          "Premise and hypothesis are given below. Does the premise entail the hypothesis, contradict it, or neither? Reply with entailment, neutral or contradiction.",
          "Natural language inference: label the pair as entailment, neutral or contradiction.",
          "Read both sentences and tell me whether the hypothesis follows from the premise (entailment), conflicts with it (contradiction) or is undetermined (neutral)."],
     post=["What is the relationship between these two sentences? Answer entailment, neutral or contradiction.",
           "Label the pair above: entailment, neutral or contradiction."])
_reg("nli", "pcm", pre=["Look how di second sentence connect to di first one. Answer with entailment, neutral or contradiction.",
                        "Dis two sentences, dem agree, dem no get relationship or dem dey fight each other? Talk: entailment, neutral or contradiction."],
     post=["Wetin be di relationship between di two sentences wey dey up there? entailment, neutral or contradiction?"])
_reg("nli", "yor", pre=["Wo ìbáṣepọ̀ tó wà láàrín àwọn gbólóhùn méjèèjì yìí. Dáhùn pẹ̀lú entailment, neutral tàbí contradiction.",
                        "Ṣe gbólóhùn kejì tẹ̀lé ti àkọ́kọ́ bí? Fèsì pẹ̀lú entailment, neutral tàbí contradiction:"],
     post=["Kí ni ìbáṣepọ̀ àwọn gbólóhùn méjì tó wà lókè yìí? entailment, neutral tàbí contradiction?"])
_reg("nli", "hau", pre=["Duba alaƙar da ke tsakanin waɗannan jimloli biyu. Ka amsa da entailment, neutral ko contradiction.",
                        "Shin jimla ta biyu ta biyo ta farko? Amsa da entailment, neutral ko contradiction:"],
     post=["Menene alaƙar jimlolin biyu da ke sama? entailment, neutral ko contradiction?"])
_reg("nli", "ibo", pre=["Lee mmekọrịta dị n'etiti ahịrịokwu abụọ a. Zaa na entailment, neutral ma ọ bụ contradiction.",
                        "Ahịrịokwu nke abụọ ọ sochiri nke mbụ? Zaa na entailment, neutral ma ọ bụ contradiction:"],
     post=["Kedu mmekọrịta dị n'etiti ahịrịokwu abụọ dị n'elu? entailment, neutral ma ọ bụ contradiction?"])

# ---------------------------------------------------------------- emotion
_reg("emotion", "eng",
     pre=["Which emotion does this text express? Choose from: {labels}.", "Classify the emotion in the following text ({labels}).",
          "What is the writer feeling? Answer with one of: {labels}.", "Emotion detection: label this text with one of {labels}."],
     post=["Which emotion is expressed above? Choose from: {labels}.", "Name the emotion of the text above ({labels})."])
_reg("emotion", "pcm", pre=["Wetin di person dey feel for dis text? Choose one: {labels}.", "Which emotion dey dis text? ({labels})"],
     post=["Which emotion dey di text wey dey up there? ({labels})"])
_reg("emotion", "yor", pre=["Ìmọ̀lára wo ni ọ̀rọ̀ yìí fi hàn? Yan ọ̀kan nínú: {labels}.", "Pín ìmọ̀lára inú ọ̀rọ̀ yìí sí ọ̀kan lára: {labels}."],
     post=["Ìmọ̀lára wo ló wà nínú ọ̀rọ̀ tó wà lókè yìí? ({labels})"])
_reg("emotion", "hau", pre=["Wane irin motsin rai wannan rubutu ya bayyana? Zaɓi ɗaya: {labels}.", "Rarraba motsin ran da ke cikin wannan rubutu: {labels}."],
     post=["Wane motsin rai ne rubutun da ke sama ya bayyana? ({labels})"])
_reg("emotion", "ibo", pre=["Kedu mmetụta ederede a na-egosi? Họrọ otu: {labels}.", "Kewaa mmetụta dị n'ederede a: {labels}."],
     post=["Kedu mmetụta ederede dị n'elu na-egosi? ({labels})"])

# ---------------------------------------------------------------- named entities, in more languages than the single English line in templates.py
_reg("ner", "pcm", pre=["Mark di names of people, place and organisation for dis sentence. Write each word, <tag>, then di label (PER, ORG, LOC, DATE or O)."])
_reg("ner", "yor", pre=["Ṣàmì àwọn orúkọ èèyàn, ibi àti ẹgbẹ́ nínú gbólóhùn yìí. Kọ ọ̀rọ̀ kọ̀ọ̀kan, <tag>, àti àmì rẹ̀ (PER, ORG, LOC, DATE tàbí O)."])
_reg("ner", "hau", pre=["Yi alamar sunayen mutane, wurare da ƙungiyoyi a wannan jimla. Rubuta kowace kalma, <tag>, sannan alamarta (PER, ORG, LOC, DATE ko O)."])
_reg("ner", "ibo", pre=["Maa aha ndị mmadụ, ebe na otu n'ahịrịokwu a. Dee okwu ọ bụla, <tag>, na akara ya (PER, ORG, LOC, DATE ma ọ bụ O)."])

# ---------------------------------------------------------------- language identification
_reg("language_id", "eng", pre=["Which language is the following text written in?", "Identify the language of this text:", "What language is this? Answer with the language name.",
                                "Name the language used in the text below."],
     post=["Which language is the text above written in?", "What language is that?"])
_reg("language_id", "pcm", pre=["Which language dem write dis text for?", "Tell me di language wey dis text dey use:"], post=["Which language be di one wey dey up there?"])
_reg("language_id", "yor", pre=["Èdè wo ni a fi kọ ọ̀rọ̀ yìí?", "Dárúkọ èdè tí ọ̀rọ̀ yìí wà ní:"], post=["Èdè wo ni ọ̀rọ̀ tó wà lókè yìí?"])
_reg("language_id", "hau", pre=["A wane harshe aka rubuta wannan rubutu?", "Gano harshen da aka yi amfani da shi a wannan rubutu:"], post=["A wane harshe ne rubutun da ke sama?"])
_reg("language_id", "ibo", pre=["Asụsụ ole ka e ji dee ederede a?", "Chọpụta asụsụ ederede a:"], post=["Asụsụ ole bụ ederede dị n'elu?"])

# ---------------------------------------------------------------- open question answering without a context
_reg("qa_open", "eng", pre=["Answer the following question in {target}.", "Please answer this question in {target}:", "Give a short, accurate answer in {target}:"],
     post=["Answer the question above in {target}.", "Please respond in {target}."])
_reg("qa_open", "pcm", pre=["Answer dis question for {target}:", "Abeg answer dis one for {target}:"], post=["Answer di question wey dey up there for {target}."])
_reg("qa_open", "yor", pre=["Dáhùn ìbéèrè yìí ní èdè {target}:", "Jọ̀wọ́ dáhùn ìbéèrè yìí ní {target}:"], post=["Dáhùn ìbéèrè tó wà lókè yìí ní èdè {target}."])
_reg("qa_open", "hau", pre=["Amsa wannan tambaya a harshen {target}:", "Don Allah ka amsa wannan tambaya a {target}:"], post=["Amsa tambayar da ke sama a {target}."])
_reg("qa_open", "ibo", pre=["Zaa ajụjụ a n'asụsụ {target}:", "Biko, zaa ajụjụ a n'asụsụ {target}:"], post=["Zaa ajụjụ dị n'elu n'asụsụ {target}."])

# ---------------------------------------------------------------- safety / offensive-language classification
_reg("safety", "eng", pre=["Is this text safe or does it need review or is it unsafe? Answer Safe, Review or Unsafe.", "Rate the safety of the following message: Safe, Review or Unsafe."],
     post=["Label the safety of the message above: Safe, Review or Unsafe."])
_reg("offensive", "eng", pre=["Is this text offensive? Answer Offensive or Not offensive.", "Does the following text contain offensive language? Answer Offensive or Not offensive."],
     post=["Is the text above offensive? Answer Offensive or Not offensive."])

# ---------------------------------------------------------------- passage ranking (listwise rerank)
_reg("rerank", "eng", pre=["Rank the passages below by how relevant they are to the query. Answer with the passage identifiers in order, most relevant first."],
     post=["Rank the passages above by relevance to the query."])

# ---------------------------------------------------------------- dialogue / safety-free "instruct" with an optional answer-language constraint is handled by answer_in (templates.py)

# ---------------------------------------------------------------- diacritic restoration (<diacritize>) and noise removal (<clean>)
_reg("diacritize", "eng", pre=["Add the missing diacritics and tone marks to this {target} text.", "Restore the diacritics in the following {target} sentence:",
                               "Rewrite this {target} text with correct tone marks and diacritics.", "Diacritize the text below (language: {target})."],
     post=["Add the correct diacritics to the {target} text above.", "Restore the tone marks of the sentence above."])
_reg("diacritize", "pcm", pre=["Add di missing tone mark and diacritics for dis {target} text:", "Put correct diacritics for dis {target} sentence:"],
     post=["Put di correct diacritics for di {target} text wey dey up there."])
_reg("diacritize", "yor", pre=["Fi àwọn àmì ohùn àti àmì abẹ́ sí ọ̀rọ̀ {target} yìí:", "Tún àwọn àmì ohùn ṣe nínú gbólóhùn {target} yìí:"],
     post=["Fi àwọn àmì ohùn tó tọ́ sí ọ̀rọ̀ {target} tó wà lókè yìí."])
_reg("diacritize", "hau", pre=["Ƙara alamomin sauti da suka ɓace a wannan rubutun {target}:", "Mayar da alamomin sauti a wannan jimla ta {target}:"],
     post=["Ƙara alamomin sauti a rubutun {target} da ke sama."])
_reg("diacritize", "ibo", pre=["Tinye akara ụda na akara ndị ọzọ efu n'ederede {target} a:", "Weghachi akara ụda n'ahịrịokwu {target} a:"],
     post=["Tinye akara ụda ziri ezi n'ederede {target} dị n'elu."])
_reg("denoise", "eng", pre=["Clean up this noisy {target} text and restore the original sentence.", "The following {target} text has typos and stray characters. Correct it:",
                            "Remove the noise from this text and write it correctly:", "Fix the corrupted characters in this {target} text."],
     post=["Clean the text above and restore the original {target} sentence.", "Correct the noise in the sentence above."])
_reg("denoise", "pcm", pre=["Clean dis noisy {target} text make e correct:", "Correct di wrong letters for dis {target} text:"], post=["Clean di text wey dey up there."])
_reg("denoise", "yor", pre=["Ṣàtúnṣe ọ̀rọ̀ {target} tí ó ní àṣìṣe yìí:", "Yọ àwọn àmì àìmọ́ kúrò nínú ọ̀rọ̀ {target} yìí:"], post=["Ṣàtúnṣe ọ̀rọ̀ tó wà lókè yìí."])
_reg("denoise", "hau", pre=["Gyara wannan rubutun {target} mai kurakurai:", "Cire ƙazantattun haruffa daga wannan rubutu:"], post=["Gyara rubutun da ke sama."])
_reg("denoise", "ibo", pre=["Megharịa ederede {target} a nwere njehie:", "Wepụ mkpụrụedemede ndị ọjọọ n'ederede a:"], post=["Megharịa ederede dị n'elu."])

# ---------------------------------------------------------------- scam / fraud detection with an explanation
_reg("scam", "eng", pre=["Is this message a scam? Answer Scam or Not scam, then explain why in one or two sentences.", "Check the following message for fraud. Say whether it is a scam and what gives it away.",
                         "Would you trust this message? Classify it as Scam or Legitimate and list the warning signs.", "Analyse this message and tell me if it is a scam attempt."],
     post=["Is the message above a scam? Explain.", "Classify the message above as Scam or Legitimate and give the reasons."])
_reg("scam", "pcm", pre=["Dis message na scam abi e correct? Tell me why:", "Check dis message well, na scam or no be scam? Explain."], post=["Na scam di message wey dey up there? Tell me why."])
_reg("scam", "yor", pre=["Ṣé ìwé ìránṣẹ́ yìí jẹ́ ẹ̀tàn? Ṣàlàyé ìdí rẹ̀:", "Ṣàyẹ̀wò ìwé ìránṣẹ́ yìí bóyá ẹ̀tàn ni:"], post=["Ṣé ẹ̀tàn ni ìwé ìránṣẹ́ tó wà lókè yìí? Ṣàlàyé."])
_reg("scam", "hau", pre=["Shin wannan saƙo zamba ne? Bayyana dalilin:", "Duba wannan saƙo ka ce ko yaudara ne:"], post=["Shin saƙon da ke sama zamba ne? Bayyana."])
_reg("scam", "ibo", pre=["Ọ bụ aghụghọ ka ozi a? Kọwaa ihe kpatara ya:", "Lelee ozi a ma ọ bụ aghụghọ:"], post=["Ozi dị n'elu ọ bụ aghụghọ? Kọwaa."])

# ---------------------------------------------------------------- dictionary: the meaning of a term
_reg("define_term", "eng", pre=["What does \"{title}\" mean? Explain it in {target}.", "Give the meaning of the word \"{title}\" in {target}.", "Define \"{title}\" ({target}).",
                                "I came across the term \"{title}\". What does it mean?"],
     any_=["\"{title}\" - meaning in {target}?"])
_reg("define_term", "pcm", pre=["Wetin \"{title}\" mean? Explain am for {target}.", "Tell me di meaning of \"{title}\" for {target}."], any_=["\"{title}\" - wetin e mean?"])
_reg("define_term", "yor", pre=["Kí ni \"{title}\" túmọ̀ sí? Ṣàlàyé rẹ̀ ní èdè {target}.", "Fún mi ní ìtumọ̀ \"{title}\" ní {target}."])
_reg("define_term", "hau", pre=["Menene ma'anar \"{title}\"? Bayyana a {target}.", "Ba ni ma'anar \"{title}\" a {target}."])
_reg("define_term", "ibo", pre=["Gịnị bụ ihe \"{title}\" pụtara? Kọwaa ya n'asụsụ {target}.", "Nye m nkọwa nke \"{title}\" n'asụsụ {target}."])

# ---------------------------------------------------------------- rewrite / refine a draft translation
_reg("refine_translation", "eng", pre=["Here is a source text and a draft translation into {target}. Improve the translation.", "Proofread this {target} translation and give a corrected version.",
                                       "Refine the draft translation below so that it is accurate and natural in {target}."],
     post=["Improve the draft translation above (into {target}).", "Give a corrected {target} translation of the source above."])
_reg("refine_translation", "pcm", pre=["See di source and di draft translation for {target}. Make di translation better."], post=["Fix di draft translation wey dey up there for {target}."])
_reg("refine_translation", "yor", pre=["Èyí ni ọ̀rọ̀ orísun àti ìtúmọ̀ ìgbékalẹ̀ rẹ̀ sí {target}. Mú ìtúmọ̀ náà dára síi."], post=["Ṣàtúnṣe ìtúmọ̀ tó wà lókè yìí sí {target}."])
_reg("refine_translation", "hau", pre=["Ga rubutun asali da fassarar farko zuwa {target}. Inganta fassarar."], post=["Inganta fassarar da ke sama zuwa {target}."])
_reg("refine_translation", "ibo", pre=["Lee ederede mbụ na nsụgharị mbụ gaa n'asụsụ {target}. Meziwanye nsụgharị ahụ."], post=["Meziwanye nsụgharị dị n'elu gaa n'asụsụ {target}."])
_reg("assess_translation", "eng", pre=["Assess the quality of this {target} translation of the source text. List any errors you find.", "Here is a source and its {target} translation. Evaluate the translation and point out errors."],
     post=["Assess the translation above and list its errors."])


# ---------------------------------------------------------------- part-of-speech: more phrasings, same five instruction languages as the other pools
from sft_new.templates import _POOLS as _P  # noqa: E402

_P["pos"]["eng"]["pre"] += ["Tag every word of the sentence below with its part of speech. Write each word, then <tag>, then its tag.",
                            "Part-of-speech tagging: output word <tag> TAG for each word of this sentence.",
                            "What is the grammatical category of each word in the sentence? Answer as word <tag> TAG pairs.",
                            "Label each word of the following sentence with its POS tag (word <tag> TAG)."]
_P["pos"]["eng"]["post"] += ["Give the part-of-speech tag of each word in the sentence above (word <tag> TAG).",
                             "POS-tag the sentence above, one word <tag> TAG pair per word."]
_reg("pos", "pcm", pre=["Give di part of speech for every word for dis sentence. Write di word, <tag>, then di tag.",
                        "Tag each word for dis sentence wit im part of speech (word <tag> TAG)."],
     post=["Wetin be di part of speech for each word for di sentence wey dey up there? Write am as word <tag> TAG."])
_reg("pos", "yor", pre=["Fún ọ̀rọ̀ kọ̀ọ̀kan nínú gbólóhùn yìí ní àmì ẹ̀ka ọ̀rọ̀ rẹ̀. Kọ ọ̀rọ̀ náà, <tag>, lẹ́yìn náà àmì rẹ̀.",
                        "Ṣàmì ẹ̀ka ọ̀rọ̀ (POS) ọ̀rọ̀ kọ̀ọ̀kan nínú gbólóhùn yìí (ọ̀rọ̀ <tag> ÀMÌ)."],
     post=["Kí ni ẹ̀ka ọ̀rọ̀ ọ̀rọ̀ kọ̀ọ̀kan nínú gbólóhùn tó wà lókè yìí? Kọ ọ́ báyìí: ọ̀rọ̀ <tag> ÀMÌ."])
_reg("pos", "hau", pre=["Ba kowace kalma a wannan jimla alamar nau'in kalmarta. Rubuta kalmar, <tag>, sannan alamarta.",
                        "Yi alamar nau'in kalma (POS) ga kowace kalma a wannan jimla (kalma <tag> ALAMA)."],
     post=["Wane nau'in kalma ne kowace kalma a jimlar da ke sama? Rubuta: kalma <tag> ALAMA."])
_reg("pos", "ibo", pre=["Nye okwu ọ bụla dị n'ahịrịokwu a akara ụdị okwu ya. Dee okwu ahụ, <tag>, wee dee akara ya.",
                        "Maa ụdị okwu (POS) nke okwu ọ bụla n'ahịrịokwu a (okwu <tag> AKARA)."],
     post=["Kedu ụdị okwu nke okwu ọ bụla n'ahịrịokwu dị n'elu? Dee ya otu a: okwu <tag> AKARA."])
