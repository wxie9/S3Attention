function runexp {

gpu=${1}
task=${2}
model=${3}
chkpath=${4}
layers=${5}
lms=${6}          # lms (landmarks) is r in paper
k_conv=${7}
wsize=${8}        # wsize is w in paper
lr=${9}
wd=${10}
seed=${11}
flags=${12}

flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=${task}-${model}-l_${layers}-lr_${lr}-wd_${wd}-lms_${lms}-winsize_${wsize}-k_conv_${k_conv}-seed_${seed}${flags_print}-chkpath_${chkpath}-padzero-resv2

cmd="
 /home/pai/bin/python -m torch.distributed.launch --nproc_per_node=4  run_tasks.py --model ${model} --task ${task}
    --num_landmarks ${lms}   --conv_kernel_size ${k_conv}  --window_size ${wsize} ${flags}
    --weight_decay 1e-2 --num_layers ${layers}  --chk_path ${chkpath}
       --dropout_prob 0.0 --attention_dropout 0.0 --learning_rate 1e-4 --rnn_dropout 0.0
    --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096
    --num_train_steps 1000000  --num_eval_steps 565 --eval_frequency 3000 --dp_rank 8 --batch_size 8
    --seed ${seed} --epoch 50
"

debug=1
if [ ${debug} -eq 0 ]; then
cmd="${cmd} --logging --expname ${expname}  > logs/${expname}.log 2>&1 &"
else
cmd="${cmd} "
fi

echo logs/${expname}.log

eval ${cmd}

}

# The following hyperparameters correspond to Transformer-LS (w,r = 8,32) in the paper.
# One can change it to Transformer-LS (best) with lms = 254, win_size = 1
# runexp  gpu     task         model                        chkpath           layers  lms  k_conv  win_size lr   wd   seed   flags
#runexp   3      retrieval  dynamicsoftmax_v9   "LRA_chks/retrieval_dynamicsoftmax_v9_softmax_r=8_totalstep=100000"   2    32    -1      8     1e-4  0.01 4096
runexp   1     retrieval  attention_SKT   "LRA_chks/retrieval_dynamic_v77_softmax_d_sigmoid_n_1_1017"   2    32    -1      8     1e-4  0.01 4096
#runexp   1      retrieval  dynamicsoftmax_v11   "LRA_chks/retrieval_dynamicsoftmax_v11_softmax_r=8_totalstep=100000"   2    32    -1      8     1e-4  0.01 4096
#runexp   1      retrieval  dynamicsoftmax_v7   "LRA_chks/retrieval_dynamicsoftmax_v7_r=8_add_diag_totalstep=100000"   2    32    -1      8     1e-4  0.01 4096
