---
license: other
license_name: mixed-per-row-see-licence-column
pretty_name: SabiYarn clean English-to-African SFT pairs (decontaminated), 2026-10-28
---
# Clean English -> African SFT pairs (2026-10-28), private

English -> African-language sentence pairs rebuilt from the reachable sources of the old SFT build, with every sentence that occurs in a protected evaluation set removed (either side, any language, strict rule: exact sentence, contained sentence, or ONE shared word 8-gram).

* `data/pairs-*.parquet`: columns `english, target, iso, source_dataset, licence, flipped, decon_flag` (`decon_flag` is always empty: no weaker review was implemented, every hit is dropped). `licence` is the class of the source (`cc-by`, `odc-by`, `apache`, `mpl`, `share_alike`, `other`, `unspecified` = no usable licence on the card, used and flagged by the old build, `non_commercial` = IbomMT train, the only deliberate non-commercial source; filter it out for a commercial variant).
* `quarantine_nc/`: rows that were read but are NOT in the main output (provenance mentions NLLB, or licence doubt) with the reason column. Not for training without an owner decision.
* `rebuild/kept`, `rebuild/meta`, `rebuild/ledger.jsonl`: per-input-file shards before the global dedupe and the per-row accounting (no text in `meta`).
* `protected_index/`: the versioned protected-set HASH index (`protected_index.npz`, sealed `protected_manifest.json` with sha256, `protected_sentence_hashes.npz`). Hashes only, no benchmark text. Loader: `scripts/protected_index_loader.py`.
* `stats/`: per-language, per-source, per-protected-set, licence tables, exposure check. `scripts/`: the code.

Not done (needs a GPU or other inputs): AfriCOMET-QE scoring and the lowest-15% / min-QE-0.20 gate, GlotLID checks of the raw sources, token-balanced windowing. See the report `reports/sft_clean_rebuild_2026-10-28.md` in the project repository.
