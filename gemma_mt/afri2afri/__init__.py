# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data preparation for the inter-African-language translation model (African language X -> African language Y, D-46).

Modules: `text` (normalisation + hashing), `provenance` (source -> kind / tier), `pivot` (join of the en->X pairs on the
English text), `policy` (UniMax-style direction budgets), `filters` (pair quality), `contam` (benchmark leak filter),
`prompt` (the `from X to Y` prompt), `windows` (multi-sentence windows). The scripts are `scripts/build_afri2afri_sft.py`
and `scripts/build_afri2afri_rl.py`; the design is in `reports/afri2afri_plan_2026-10-20.md`.
"""
