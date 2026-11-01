# Growing the model in depth (385M parameters)

**Why depth.** Below one billion parameters deeper-and-thinner models beat wider ones (MobileLLM). With 125 languages in the table, capacity matters
(the curse of multilinguality: more languages raise loss, and scaling the model helps more than scaling the data).

**What is grown.** The SFT model `gemma-3-270m-it` + 6 inserted local layers (24 layers, 301,539,968 parameters) becomes 39 layers (385,144,448
parameters): 15 new local (sliding-window) layers, laid out as 3 x (12 local + 1 global), `sliding_window_pattern` 13.

**How the new layers start (identity).** Each new layer is a copy of the layer it follows, except that `self_attn.o_proj` and `mlp.down_proj` are zero. Gemma applies
`post_attention_layernorm` and `post_feedforward_layernorm` to the sub-layer output before the residual add, and the RMSNorm of an all-zero vector is exactly zero, so
both residual branches add 0.0: the grown model computes the same function as the original (logits and hidden states differ by exactly 0; check:
`scripts/upcycle_check.py`). A small post-norm gain (0.01) damps the first updates. Code: `scripts/upcycle_layers.py`.

**Dedicated language tags.** The tag in front of the answer becomes one token per language, placed in free `<unusedK>` vocabulary slots (`gemma_mt/tags.py`; rows
99 and up take slots from the far end of the pool so they cannot collide with a future vocabulary swap). The 125 embedding rows start as the mean of the embeddings of the
old text-tag pieces (`scripts/init_tag_rows.py`). Existing packs are rewritten without changing pack structure (`data_pipeline/sft_clean/chain/retag_packs.py`).
A tokenizer variant that names the slots `<yor>`, `<hau>`, ... (same ids) is produced by `scripts/rename_tag_tokens.py`; the packed data stores a fingerprint of the
vocabulary, so training keeps using the original names.

**Stages.**

* Stage A (`configs/growth1/g1a.yaml`): 1,500 steps; only the 15 new layers and the 125 tag rows train (`freeze_regex`, `train_embedding_tags` in `train_sft.py`). Data: half
  instruction packs never seen by any model, half a language-balanced replay of the older translation packs.
* Stage B (`configs/growth1/g1b.yaml`): everything trains, one pass over every remaining pack (each language: low-resource languages already had up to four epochs
  in the earlier run), cut into six shards of about 5,300 steps that fit the training box's disk (`data_pipeline/sft_clean/fresh/make_shards.py`); each shard starts from the
  previous shard's weights.
* Replay (`add_replay_to_shard.py`, `make_replay.py`): after shard 1 the macro validation loss rose from 1.785 to about 2.03, worst on tiny languages (forgetting). The 40
  smallest languages (30,627 packs, 125M tokens) are replayed at about 9% of every later shard, and the peak learning rate is lowered from 5e-5 to 3e-5.

**Results so far** (macro validation loss over 78 languages, lower is better; `experiments/growth1/` has the raw logs):

| Point | Macro | Micro |
|---|---|---|
| end of the 270M run (24 layers) | 1.772 | |
| end of stage A (new layers only) | 1.785 | 1.750 |
| end of shard 1 (all trainable, no replay) | 1.980 | 1.816 |
| shard 2, step 9,000 | 1.676 | 1.592 |
| shard 2, step 10,000 | 1.640 | 1.561 |
