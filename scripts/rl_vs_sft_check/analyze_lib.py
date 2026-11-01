
import unicodedata
from sacrebleu.metrics import CHRF
CH = CHRF(word_order=2)
def nfc(s): return unicodedata.normalize("NFC", s or "")
def sent_chrf(h, r): return CH.sentence_score(nfc(h), [nfc(r)]).score
