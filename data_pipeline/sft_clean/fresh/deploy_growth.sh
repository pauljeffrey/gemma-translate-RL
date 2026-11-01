#!/bin/bash
# Runs on the LAPTOP: copy the growth-1 code to the A100 (code/gemma_mt = the package, code/gemma_mt_root = configs + scripts, code/sft_clean/fresh = shell/launch scripts).
# Only the files of the growth build are copied; the first run keeps a backup of every file it replaces (<file>.pre_growth) so the old trainer can be restored.
#   bash deploy_growth.sh
K=$SSH_KEY; O="-i $K -o IdentitiesOnly=yes -o LogLevel=ERROR -o ConnectTimeout=30"; GPU="-p $BOX_PORT root@$BOX_HOST"
R="$(cd "$(dirname "$0")/../../.." && pwd)"; G="$R/gemma_mt"; D="/workspace/work/code"
put() { ssh $O $GPU "[ -f $2 ] && [ ! -f $2.pre_growth ] && cp -p $2 $2.pre_growth; true"; scp -q $O -P $BOX_PORT "$1" root@$BOX_HOST:$2 || { echo "FAILED $1"; exit 1; }; }
for f in train_sft.py growth.py tags.py languages.py; do put "$G/gemma_mt/$f" "$D/gemma_mt/$f"; done
for f in upcycle_layers.py init_tag_rows.py export_trainer_state.py; do put "$G/scripts/$f" "$D/gemma_mt_root/scripts/$f"; done
ssh $O $GPU "mkdir -p $D/gemma_mt_root/configs/growth1"
for f in g1a.yaml g1b.yaml; do put "$G/configs/growth1/$f" "$D/gemma_mt_root/configs/growth1/$f"; done
for f in launch_growth.sh; do put "$R/data_pipeline/sft_clean/fresh/$f" "$D/sft_clean/fresh/$f"; done
put "$R/data_pipeline/sft_clean/chain/extend_packed.py" "$D/sft_clean/chain/extend_packed.py"
ssh $O $GPU "cd $D/gemma_mt && grep -c train_embedding_tags train_sft.py && python3 -c 'import ast,sys; [ast.parse(open(f).read()) for f in (\"train_sft.py\",\"growth.py\",\"tags.py\",\"languages.py\")]' && echo deployed_ok"
