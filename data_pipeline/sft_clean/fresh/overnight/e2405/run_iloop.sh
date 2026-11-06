cd /workspace/e2405; E=/workspace/eval_385m; L=yor,hau,ibo,swh,amh,zul,xho,kin,som,wol,lug,sna,twi,ewe,lin,tir,orm,nso,tsn,afr; V=$E/venv_comet/bin/python
$V e_iloop.py /dev/shm/evid_e20/sft.npz base/sft.npz $E/out_m385/predictions.jsonl --langs $L &
$V e_iloop.py /dev/shm/evid_e20/stageA.npz base/stageA.npz $E/out_stageA/predictions.jsonl $E/out_stageA_x/predictions.jsonl --langs $L &
$V e_iloop.py /dev/shm/evid_e20/b1.npz base/b1best.npz $E/out_b1best/predictions.jsonl $E/out_b1best_x/predictions.jsonl --langs $L &
$V e_iloop.py /dev/shm/evid_e20/b1final.npz base/b1final.npz $E/out_b1final/predictions.jsonl --langs $L &
$V e_iloop.py /dev/shm/evid_e20/mixed.npz base/mixed.npz $E/out_wk_mixedbest/predictions.jsonl --langs $L &
wait
