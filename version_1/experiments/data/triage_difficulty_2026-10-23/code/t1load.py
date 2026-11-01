import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from t1lib import *
BENCHES = ["flores_plus", "ntrex", "mafand", "smol_sent", "glot_story", "salt"]
BNAME = {"flores_plus": "FLORES+ devtest", "ntrex": "NTREX-128", "mafand": "MAFAND-MT", "smol_sent": "SmolSent", "glot_story": "GlotStoryBook", "salt": "SALT"}
# language-code harmonisation between benchmarks (FLORES+ uses ISO 639-3 swh)
ISO_FIX = {"swa": "swh"}

def build():
    cls = load_cls()
    parts = []
    for b in BENCHES:
        if os.path.exists(f"{DATA}/chunks_{b}.csv.gz"):
            parts.append(load_bench(b, cls))
    A = pd.concat(parts, ignore_index=True)
    A["iso"] = A.iso.replace(ISO_FIX)
    A["log_words"] = np.log(A.n_words)
    # language groups: terciles of RL-final macro chrF++ on FLORES+ devtest (no_pair, k=1) over the 45 languages
    f = A[(A.b == "flores_plus") & (A.k == 1) & (A.n_pair_seen == 0)]
    lm = f.groupby("iso").chrf_rl.mean().sort_values()
    n = len(lm); tier = {}
    for i, (iso, v) in enumerate(lm.items()): tier[iso] = "weak" if i < n / 3 else ("mid" if i < 2 * n / 3 else "strong")
    gm = pd.DataFrame({"iso": lm.index, "rl_flores_chrf": lm.values, "group": [tier[i] for i in lm.index]})
    A["group"] = A.iso.map(tier).fillna("other")
    A["efi"] = A.efi.astype(bool)
    return A, gm

if __name__ == "__main__":
    A, gm = build()
    print(A.shape, A.group.value_counts().to_dict())
    print(A.groupby("b").iso.unique().apply(lambda x: sorted(x)).to_dict())
    A.to_pickle(f"{DATA}/A.pkl"); gm.to_csv(f"{DATA}/lang_groups_flores.csv", index=False)
