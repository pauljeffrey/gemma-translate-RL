import json, re
LANG = {
 "ish": dict(name="Esan", where="the Esan people of Edo State, Nigeria (Ekpoma, Irrua, Uromi)",
   note="Esan (also called Ishan) is an Edoid language written in Latin script with the dot-below vowels ẹ and ọ (capitals Ẹ, Ọ). Digraph consonants such as mh, nh, bh, gb, gh are common. Tone is not marked in writing. Keep personal and place names as in the examples."),
 "bin": dict(name="Edo (Bini)", where="the Edo (Bini) people of Edo State, Nigeria (Benin City)",
   note="Edo (Bini) is an Edoid language written in Latin script with the dot-below vowels ẹ and ọ (capitals Ẹ, Ọ). Digraph consonants such as kh, rh, vb, gb, gh, mw are common. Tone is not marked in writing. Keep personal and place names as in the examples."),
 "bbj": dict(name="Ghomala'", where="the Ghomala' (Bamileke) people of the West Region of Cameroon (Bandjoun, Baham)",
   note="Ghomala' is a Grassfields Bantu (Bamileke) language of Cameroon written in Latin script with the extra letters ə ɑ ɛ ɔ ʉ ŋ, with tone marked by diacritics (acute, grave, caron, e.g. á à ǎ) and the glottal stop written as an apostrophe (') as in the examples."),
 "oke": dict(name="Okpe", where="the Okpe people of Delta State, Nigeria (Orerokpe, Sapele area)",
   note="Okpe is a south-western Edoid language written in Latin script with the dot-below vowels ẹ and ọ (capitals Ẹ, Ọ). Tone is not marked in writing. Keep personal and place names as in the examples."),
 "tsc": dict(name="Tswa", where="the Tswa people of southern Mozambique (Inhambane Province)",
   note="Tswa (Citswa) is a Bantu language (Tswa-Ronga group) of southern Mozambique, closely related to but different from Tsonga/Changana. It is written in plain Latin script (very few diacritics; some Portuguese loanwords keep ã, ç, á, é). Tone is not marked."),
 "dag": dict(name="Dagbani", where="the Dagomba people of northern Ghana (Tamale)",
   note="Dagbani (Dagbanli) is a Gur (Oti-Volta) language of northern Ghana written in Latin script with the extra letters ɛ ɔ ɣ ŋ (capitals Ɛ Ɣ Ŋ) and ʒ; the glottal stop is written with an apostrophe. Tone is not marked in writing."),
}
def instr(iso, k, note=False):
    L=LANG[iso]; n=L["name"]
    if k==0:
        return (f"Translate the following English text into {n}, the language of {L['where']}. "
                f"Reply with the {n} translation only: no explanation, no notes, no English, no transliteration.")
    s=(f"You are an expert translator from English into {n}, the language of {L['where']}. ")
    if note: s+=f"\n\nLanguage notes: {L['note']}\n\n"
    s+=(f"Below are {k} example English-{n} translation pairs from a parallel corpus (the most similar to the new text last). "
        f"Translate the new English text into {n} in the same spelling and style as the examples. Reply with the {n} translation only: no explanation, no notes.")
    return s
def build_prompt(iso, src, shots, k, note=False):
    L=LANG[iso]; n=L["name"]
    if k==0:
        return instr(iso,0)+f"\n\nEnglish: {src}"
    sh=list(reversed(shots[:k]))     # shots are best-first; show most similar last
    body=instr(iso,len(sh),note)+"\n\n"
    for i,s in enumerate(sh,1):
        body+=f"Example {i}\nEnglish: {s['english']}\n{n}: {s['target']}\n\n"
    body+=f"New text\nEnglish: {src}\n{n}:"
    return body
def clean_out(t, iso):
    t=t.strip()
    n=LANG[iso]["name"]
    for p in (f"{n}:", "Translation:", f"{n.split(' ')[0]}:"):
        if t.startswith(p): t=t[len(p):].strip()
    if len(t)>1 and t[0] in "\"“" and t[-1] in "\"”": t=t[1:-1].strip()
    return t

def _a(iso,name,where,note): LANG[iso]=dict(name=name,where=where,note=note)
_a("ibb","Ibibio","the Ibibio people of Akwa Ibom State, Nigeria","Ibibio is a Lower Cross language written in Latin script with the dot-below vowels ọ ẹ ị ụ (e.g. ikọ, nnyịn). Follow the spelling of the examples.")
_a("urh","Urhobo","the Urhobo people of Delta State, Nigeria","Urhobo is a south-western Edoid language written in Latin script with the dot-below vowels ẹ and ọ (capitals Ẹ, Ọ) and the digraph vw (e.g. vwo, avwanre). Tone is not marked.")
_a("nbl","South Ndebele (isiNdebele)","the South Ndebele people of South Africa (Mpumalanga, Gauteng, Limpopo)","South Ndebele is an Nguni Bantu language written in plain Latin script without diacritics. Words are agglutinative: prefixes and verb extensions are written together as in the examples.")
_a("shi","Tachelhit (Tashelhit, Shilha)","the Tachelhit-speaking Amazigh (Berber) people of southern Morocco (Souss)","Tachelhit is an Amazigh language. In this corpus it is written in a Latin transliteration that uses combining dots and marks and letters such as š and ɛ; copy the conventions of the examples exactly (not Tifinagh, not Arabic script).")
_a("kea","Kabuverdianu (Cape Verdean Creole)","Cape Verde","Kabuverdianu is a Portuguese-lexified creole, written in Latin script with accents (á é í ó ú ê ô). It is not Portuguese: use Kabuverdianu grammar and spelling as in the examples.")
_a("ach","Acholi","the Acholi people of northern Uganda","Acholi is a Luo (Nilotic) language written in plain Latin script with almost no diacritics.")
_a("vmw","Makhuwa","the Makhuwa people of northern Mozambique (Nampula)","Makhuwa is a Bantu language written in Latin script with few diacritics; some loanwords come from Portuguese.")
_a("seh","Sena (Cisena)","the Sena people of central Mozambique (Sofala, Tete)","Sena is a Bantu language written in Latin script; Portuguese loanwords are common and may keep their diacritics.")
_a("nyn","Nyankore (Runyankore)","the Nyankore people of south-western Uganda","Nyankore is a Bantu language written in plain Latin script (Runyankore-Rukiga orthography).")
_a("her","Herero (Otjiherero)","the Herero people of Namibia","Herero is a Bantu language written in Latin script, with the letter ṱ (t with a mark below) as in the examples.")
_a("yao","Yao (Chiyao)","the Yao people of Malawi and northern Mozambique","Yao is a Bantu language written in Latin script with the letter ŵ (capital Ŵ).")
_a("sid","Sidamo (Sidaamu Afoo)","the Sidama people of southern Ethiopia","Sidamo is a Highland East Cushitic language written in Latin script; the glottal stop sign in this corpus is the character ˈ, as in the examples.")
_a("ndc","Ndau (Cindau)","the Ndau people of central Mozambique and eastern Zimbabwe","Ndau is a Bantu language of the Shona group, written in Latin script with few diacritics.")
_a("kri","Krio","Sierra Leone","Krio is an English-based creole written in Latin script with the letters ɔ ɛ ŋ (e.g. dɛn, fɔ). It is not English: use Krio words and the spelling of the examples.")
_a("dua","Duala","the Duala people of the Littoral Region of Cameroon (Douala)","Duala is a Bantu language written in Latin script with tone and length marks (ń, á, ṅ, ō, ī, ē, and a combining line below some letters); copy the marks as in the examples.")
_a("bas","Basaa","the Basaa people of Cameroon (Centre and Littoral Regions)","Basaa is a Bantu language written in Latin script with the letters é ô ñ (and ŋ-like ñ); follow the spelling of the examples.")
_a("wes","Cameroon Pidgin English (Kamtok)","Cameroon","Cameroon Pidgin is an English-based creole with its own spelling (e.g. weh, di, for, dem). It is not standard English: write Pidgin as in the examples.")
