"""Language groups of the RL-vs-SFT check (2026-10-21)."""
CONSENSUS = ["ibb","urh","nbl","bin","dag","oke","shi","bbj","kea","ach","tsc","vmw","seh","nyn","her","yao","sid","ndc","kri","ish","dua","bas","wes"]
BLEND = ["efi","bam","fon","kab","kam","kbp","kmb","mos","nus","umb","dyu","sag","swc","loz","ven","iso","tiv","kqn","wal","lun","kua","kon"]
CONTROL = ["yor","hau","swh","kin","wol","zul","ibo","xho","som","sna"]
GROUP = {**{i:"consensus" for i in CONSENSUS}, **{i:"blend" for i in BLEND}, **{i:"control" for i in CONTROL}}
ALL_LANGS = CONSENSUS + BLEND + CONTROL
SENT_BENCHES = ["flores_plus","smol_sent","ntrex","mafand","ibom_mt","flores_moz","glot_story","salt"]
BENCH_PATH = {"flores_plus":"flores_plus","ntrex":"ntrex","mafand":"mafand","smol_sent":"smol_sent","ibom_mt":"ibom_mt","flores_moz":"flores_moz","glot_story":"glot_story","salt":"salt"}
