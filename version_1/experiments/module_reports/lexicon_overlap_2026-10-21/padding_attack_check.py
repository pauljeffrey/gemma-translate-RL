import json, sys, numpy as np, collections
sys.path.insert(0, "/workspace/jobs/b3/code")
sys.path.insert(0, "/workspace/jobs/b3/code/scripts")
from gemma_mt.languages import LANGUAGES
from gemma_mt.rl.lexicon import load_lexicon, x_tokens, english_content_stems
import validate_lexicons as V
for iso in "fon kab kam kbp kea kmb mos sag umb vmw".split():
    lex = load_lexicon(iso, "/workspace/jobs/b3/lexicon_final")
    src, ref = V.flores_pairs("flores", LANGUAGES[iso].flores)
    st = V.stuffed_output(lex, src, 30).split()
    cand = lex.candidates
    out = []
    for i, s in enumerate(src[:300]):
        filler = [t for t in x_tokens(ref[(i + 17) % len(ref)]) if V.x_folded_tokens(t) and V.x_folded_tokens(t)[0] not in cand]
        r_plain = lex.overlap(s, " ".join(st))
        pad = " ".join(st + filler[:30])
        r = lex.overlap(s, pad)
        h = lex.overlap(s, ref[i])
        out.append((h.overlap, r.overlap, r.density, lex.guarded(r), h.density))
    a = np.array([[x if x is not None else np.nan for x in o] for o in out])
    print(iso, "human %.3f | stuffed30+nonCand filler: overlap %.3f density %.2f guarded %.3f | human density %.2f dmax %.2f" % (np.nanmean(a[:,0]), np.nanmean(a[:,1]), np.nanmean(a[:,2]), np.nanmean(a[:,3]), np.nanmean(a[:,4]), lex.density_max))
