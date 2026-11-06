#!/bin/bash
# E-04 pool: the stage-A pool (clean_v3_385m: general, news/, ref/) restricted to the 20 languages of the decoding study (symlinks, no copies)
P=/workspace/clean_rerun/prompts; D=$P/seed_subset20; rm -rf $D; mkdir -p $D/news $D/ref
for l in yor hau ibo swh amh zul xho kin som wol lug sna twi ewe lin tir orm nso tsn afr; do
  [ -e $P/clean_v3_385m/$l.parquet ] && ln -s $(readlink -f $P/clean_v3_385m/$l.parquet) $D/$l.parquet
  [ -e $P/clean_v3_385m/news/$l.parquet ] && ln -s $P/clean_v3_385m/news/$l.parquet $D/news/$l.parquet
  [ -e $P/clean_v3_385m/ref/$l.parquet ] && ln -s $P/clean_v3_385m/ref/$l.parquet $D/ref/$l.parquet
done
ls $D | wc -l; ls $D/news | wc -l; ls $D/ref | wc -l
