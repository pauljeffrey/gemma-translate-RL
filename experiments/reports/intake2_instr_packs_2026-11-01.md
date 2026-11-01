# Instruction stream: packs, UniMax schedule, interleave proposal (2026-11-01)

Follow-up of `reports/intake2_instr_2026-10-31.md`. CPU only; nothing under `state/`, `logs/qe_*`, `work/scores_*`, `hub/extra2k`, `finish_all.sh` was touched. Code: `data_pipeline/sft_clean/chain/intake2_instr_pack.py`, `intake2_aya_instr.py` (same files in `/workspace/work/code/sft_clean/chain/`).

## Owner decisions applied
1. New tags arz ary aeb bci (Dinka, Teso, Adhola) stay in the instruction shards, but **cannot enter the packs** (see "No tag" below).
2. `tumbuka_*` kept (synthetic flag stays in the shard column).
3. `instr/muri_it` moved to `instr/_excluded/muri_it` (including its 38.7k previously held rows; nothing deleted). Dinka (`din`) was muri-only, so it is gone.
4. Aya: new source `instr/aya_collection_aligned` (`intake2_aya_instr.py`): African side only (inputs -> user, targets -> assistant) of the **fully aligned 100k-blocks** (same dataset_name+id as the English split at the same position; blocks with share >= 0.999) of 12 splits (afr amh hau ibo knc mlg nso sna som sot swh xho); 1.2-2.0M aligned rows per language, **hash-sampled to 100k per language** (my cap, to keep Aya at 14% of the tokens; raise `--cap` to use more), then the usual cleaning, strict protected filter, dedupe. 1,164,985 rows (169M tokens), `synthetic=true`, `licence_raw` says NLLB-3.3B translations. The aya parallel pairs stay in `hub/extra2/aya_aligned_QUARANTINE_nllb` (not scored).
5. `_held_synthetic` moved back into the main path (15 files, list in `instr/_state/held_moved_back_2026-10-31.json`); `_held_synthetic` no longer exists.

## Packs: `/workspace/work/work/packed_instr/{train,val}` (separate dir, not appended to packed_ext)
Same files and dtypes as `gemma_mt.data.prepare_sft` (`sample_offset sample_len prompt_len target_len iso_id direction pack_start` .npy, `tokens.u32.bin`), `meta.json` with `iso_table` = `gemma_mt.languages.ALL_ISOS` (85) and the tokenizer fingerprint of `google/gemma-3-270m-it` (`69f7f26e2dee...`, vocab/special ids/chat template). seq_len 4096, one language per pack, best-fit-decreasing (`pack_buffered`), `PackedDataset` loads it (checked).
* Rendering: the tokenizer's own chat template (`fmt.render_chat`, system turn folded into the first user turn), exactly like the translation prompts. **No `{iso} ` tag in front of the answer** (that is a translation-task convention).
* Loss: **final assistant turn only** (`prompt_len` = everything before it, `target_len` = final turn through `<end_of_turn>`, the trailing newline is the masked tail). Earlier assistant turns of multi-turn chats get NO loss, because the pack format has a single prompt span and a single target span. Multi-turn rows: som_conversations, bsc_m_personas, pidgin_clean, oromiffa and a few others (78k rows, 1.8%).
* `direction` = 0 for all samples (translation concept; 0 keeps the loaders and validation happy). Samples > 4096 tokens dropped (2,658). Val = 0.2% by hash of the row id, at least 60 rows per language with >= 1,500 rows: 10,281 samples, 744 packs.
* Result: train **4,190,976 samples, 300,970 packs, 1,232,162,077 tokens** (efficiency 0.9995); about 608.6M loss tokens. **86% of the rows (90% of the tokens) are machine-translated or LLM-generated**; only about 0.59M samples are not.
* Audit: `intake2_instr_pack.py audit` (bos first, `<end_of_turn>` closes the target, lengths add up, packs <= 4096 and single-language, 20,000 samples per split: 0 problems). The translation `audit_packed` expects the translation template and fails on free chat by design.

| iso | rows (train+val) | tokens (train) | loss tokens | packs | draws | epochs |
|---|---:|---:|---:|---:|---:|---:|
| amh | 883,977 | 339.9M | 187.9M | 83,007 | 83,007 | 1.0 |
| tum | 627,590 | 163.1M | 15.4M | 39,852 | 39,852 | 1.0 |
| swh | 261,707 | 68.3M | 36.4M | 16,672 | 16,672 | 1.0 |
| bam | 458,558 | 58.3M | 31.6M | 14,233 | 14,233 | 1.0 |
| yor | 107,365 | 54.8M | 35.6M | 13,376 | 13,376 | 1.0 |
| xho | 151,148 | 49.1M | 24.1M | 12,001 | 12,001 | 1.0 |
| sna | 151,228 | 46.9M | 23.0M | 11,442 | 11,442 | 1.0 |
| ibo | 145,576 | 46.5M | 23.6M | 11,364 | 11,364 | 1.0 |
| sot | 141,384 | 45.0M | 23.9M | 10,997 | 10,997 | 1.0 |
| som | 216,467 | 44.6M | 26.4M | 10,902 | 10,902 | 1.0 |
| mlg | 169,678 | 44.3M | 23.9M | 10,827 | 10,827 | 1.0 |
| zul | 103,860 | 43.3M | 22.3M | 10,562 | 10,562 | 1.0 |
| hau | 140,795 | 36.0M | 17.1M | 8,797 | 8,797 | 1.0 |
| kin | 49,935 | 33.5M | 16.7M | 8,189 | 8,189 | 1.0 |
| nya | 38,307 | 24.4M | 12.4M | 5,965 | 5,965 | 1.0 |
| run | 40,070 | 24.4M | 13.6M | 5,947 | 5,947 | 1.0 |
| tir | 34,720 | 18.6M | 8.7M | 4,549 | 4,549 | 1.0 |
| twi | 51,656 | 16.8M | 14.4M | 4,115 | 4,115 | 1.0 |
| orm | 52,193 | 16.5M | 14.7M | 4,022 | 4,022 | 1.0 |
| afr | 132,192 | 14.7M | 8.3M | 3,581 | 3,581 | 1.0 |
| nso | 98,000 | 14.6M | 8.3M | 3,570 | 3,570 | 1.0 |
| wol | 49,080 | 13.7M | 11.8M | 3,356 | 3,356 | 1.0 |
| knc | 71,568 | 10.9M | 6.1M | 2,666 | 2,666 | 1.0 |
| tsn | 1,429 | 1.7M | 1.3M | 427 | 427 | 1.0 |
| kon | 7,751 | 0.7M | 0.4M | 167 | 167 | 1.0 |
| lug | 5,473 | 0.6M | 0.4M | 156 | 156 | 1.0 |
| lin | 3,873 | 0.4M | 0.2M | 105 | 105 | 1.0 |
| nyn | 4,251 | 0.3M | 0.2M | 85 | 85 | 1.0 |
| pcm | 1,426 | 0.2M | 0.1M | 38 | 60 | 1.579 |
| **total** | 4,201,257 | 1232.2M | 608.6M | 300,970 | 300,992 | |

(Amharic is 28% of the packs, 83k packs, mostly machine-translated sets: `amharic_sft_combined` 135M tokens, `amharic_finetome`, `yoseali_amh` 69M human-ish. Tumbuka 40k packs. If the owner wants a smaller Amharic share, cap `amharic_sft_combined` before the build: it is the largest single synthetic source. A few Amharic rows in `amharic_sft_combined` have the bare prompt "Complete the following task" (raw text continuation); not filtered.)

## No tag: kept out of the pack (rows are in the shards, not in `packed_instr`)
These six languages are not in `gemma_mt.languages.ALL_ISOS`, so there is no `iso_id` (the trainer, the sampler and the validation index by it). Adding them to the registry (appended at the end, so existing ids do not move) and re-running `build` takes about 6 minutes.

| iso | rows kept out of the pack | tokens (Gemma, chat template) | name |
|---|---:|---:|---|
| arz | 38,583 | 17.34M | Egyptian Arabic |
| bci | 16,304 | 2.89M | Baoule |
| aeb | 4,986 | 1.02M | Tunisian Arabic |
| teo | 4,256 | 0.32M | Ateso |
| adh | 4,256 | 0.30M | Dhopadhola |
| ary | 5,312 | 0.21M | Moroccan Arabic |

(Tag in the Gemma setup is plain text, so no tokenizer token swap is needed for the pack format itself; the swap question only arises if the owner wants a dedicated token.)

## UniMax schedule: `/workspace/work/work/sched_instr/` (schedule.npy 9,406 x 32, schedule.json sidecar, draws_per_language.csv)
a_i = clip(L, n_i, 4 n_i) (`gemma_mt.data.packed_dataset.unimax_allocation`, min 1 / max 4 epochs, no priority weights). Translation stream estimate (real packs do not exist yet; gate `keep_cal` x (2 x mean target tokens + 30)/(4096 x 0.9994) for 81 languages, L = 20,000): **1,202,176 packs = 4.92B tokens**; one quarter = 300,544 packs = 1.23B tokens.
* The instruction floor (every pack once) is already **300,970 packs = 1.232B tokens = 25.0% of the translation budget**. So the L that gives 1/4 is **not binding: every language sits at its one-epoch floor (L <= 60 packs, the smallest language)**, every pack is seen exactly once; the schedule has 300,992 draws (22 extra draws to round to 32-pack steps, given to `pcm`, 1.58 epochs). No language is repeated up to 4x.
* To get repeats, the share must grow: share 0.30 -> L 10,136; 0.35 -> 14,059; 0.40 -> 17,924; 0.50 -> 26,936 packs (table in `sched_instr/meta.json`). To stay at 1/4 and still repeat small languages, the big sets must shrink instead (amh, tum, bam, swh).

## Interleave proposal (not launched)
Extend the translation pack set with `extend_packed.py --base work/packed_ext --add work/packed_instr` **only after** making sure val is handled (it keeps the base val; the instruction val split would be unused; evaluate it separately), then merge draws: `intake2_instr_pack.py interleave` gives each stream draws the key (k+u)/draws and sorts, so the mixture is constant over the whole run (tested on a mock: 20.0% instruction draws in every 5,000-step window; 46,974 steps x 32 for 1.20M + 0.30M draws). Instruction pack indices are offset by the number of translation train packs. Loss of the instruction packs is on the final turn only. Language-balanced validation of the translation run is unchanged.

## Commands
```
cd /workspace/work/code/sft_clean/chain; source /workspace/work/env.sh; export PYTHONPATH=/workspace/work/code
python intake2_aya_instr.py --cap 100000 --workers 12                       # Aya aligned blocks -> instr/aya_collection_aligned
python intake2_instr_pack.py build --workers 24                             # -> work/packed_instr (about 7 min)
python intake2_instr_pack.py audit --packed /workspace/work/work/packed_instr
python intake2_instr_pack.py schedule [--trans-packs-csv iso,packs] [--share 0.25] [--lam-trans 20000]    # -> work/sched_instr (give the real translation packs once they exist)
python intake2_instr_pack.py interleave --trans SCHED_T --instr work/sched_instr --n-trans-packs N_T --out SCHED_TI   # proposal
```
