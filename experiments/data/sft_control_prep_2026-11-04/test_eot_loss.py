"""CPU test of train_sft_eot: tiny random Gemma3 (torch, transformers 4.57.6 = .venv-comet). Checks (1) weight 1.0 == original sft_nll_sum, (2) weight 5 == manual sum, (3) count n unchanged."""
import os, sys, math
os.environ["EOT_LOSS_WEIGHT"] = "5.0"; os.environ["EOT_TOKEN_IDS"] = "106"
sys.path.insert(0, "gemma_mt"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import Gemma3TextConfig, Gemma3ForCausalLM
import gemma_mt.losses as L
orig = L.sft_nll_sum
import train_sft_eot as T
cfg = Gemma3TextConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=64, sliding_window=8, final_logit_softcapping=None, attn_logit_softcapping=None)
torch.manual_seed(0); m = Gemma3ForCausalLM(cfg).eval()
ids = torch.randint(3, 290, (2, 12)); ids[0, 5] = 106 if False else ids[0, 5]
labels = ids.clone(); labels[:, :3] = -100; labels[0, 8] = 106; labels[1, 10] = 106
kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
a, na = orig(m, kw, labels, 4, "chunked")
b, nb = T.sft_nll_sum_eot(m, kw, labels, 4, "chunked")
logp, sel = L.sft_selected_logp(m, kw, labels, 4, "chunked"); tgt = labels[:, 1:][sel]
manual = -(logp * torch.where(tgt == 106, 5.0, 1.0)).sum()
print("orig", float(a), "weighted", float(b), "manual", float(manual), "n", na, nb, "n_eot", int((tgt == 106).sum()))
assert na == nb and abs(float(b) - float(manual)) < 1e-5 and float(b) > float(a)
T.W = 1.0
c, _ = T.sft_nll_sum_eot(m, kw, labels, 4, "chunked"); assert abs(float(c) - float(a)) < 1e-6
print("OK")
