Output of the OTHER T1 session (NOTE_T1a.txt on the box), built BEFORE the coordinator's plan change.  NOT the main result.
Plan of that session: slots = leaf tokens of freed scripts (CJK first; free_order[0:8000] Ge'ez, [8000:16000] Fon), variants Ge'ez N=1000/3000/8000, Fon N=1000/3000/5000 and a combined one.
These builds REMOVE vocabulary entries (CJK leaf tokens) and (Fon, 8000) merge rules, which the coordinator's plan forbids; the main results are in ../tokenizers, ../tests (unused ids 256001-262143, nothing removed).
Kept for transparency only: their identity tests (0 differences on text without the special characters; the freed-script text is changed because tokens were removed),
fertility, tokenizers (gz) and a partial sanity (log excerpts are quoted in the report).  Its merge lists were trained on slightly different text (wiki + FineTranslations slices, min count 3).
