function runexp {

gpu=${1}
task=${2}
model=${3}
chkpath=${4}
layers=${5}
lms=${6}         # lms (landmarks) is r in paper
k_conv=${7}
wsize=${8}       # wsize is w in paper
lr=${9}
wd=${10}
seed=${11}
flags=${12}

flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=${task}-${model}-l_${layers}-lr_${lr}-wd_${wd}-lms_${lms}-winsize_${wsize}-k_conv_${k_conv}-seed_${seed}${flags_print}-chkpath_${chkpath}

cmd="
/home/pai/bin/python -m torch.distributed.launch --nproc_per_node=4 run_tasks.py --model ${model} --task ${task}
    --num_landmarks ${lms}   --conv_kernel_size ${k_conv}  --window_size ${wsize} ${flags}
    --weight_decay 1e-2  --num_layers ${layers}  --chk_path ${chkpath} --max_seq_len 1024
    --dropout_prob 0.0 --attention_dropout 0.0 --learning_rate 1e-4 --rnn_dropout 0.5
    --num_train_steps 624000000 --num_eval_steps 312 --eval_frequency 624 --batch_size 64 --warmup 624
    --n_train_samples 159999 --n_dev_samples 20000 --n_test_samples 20000 --global_step 128 --num_classes 2 --dp_rank 64
    --seed ${seed} --epoch 120
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
# One can change it to Tra:nsformer-LS (best) with lms=2, win_size=16
# runexp  gpu   task    model               chkpath                                         layers  lms  k_conv  win_size lr   wd   seed   flags    
#runexp    4    pathfinder32-curv_contour_length_14  dynamicsoftmax_v9    "LRA_chks/path_dynamicsoftmax_v9_r=16_softmax_dropout=0.1_lr=1e-4_step=62400"    2     512    -1  8     1e-4   0.0 1234
runexp    0   pathfinder32-curv_contour_length_14  attention_SKT    "LRA_chks/path_dynamic_v77_softmax_d_softmax_n_1_1017"    2   64    -1      8     1e-2   0.0 1234 
#runexp    6    pathfinder32-curv_contour_length_14  dynamicsoftmax_v11    "LRA_chks_best/path_dynamicsoftmax_11_r=32_dropout=0.1_lr=1e-4_warmup=6240_step=62400"    2     512    -1      8     1e-4   0.0 1234



# /home/pai/bin/python -m torch.distributed.launch --nproc_per_node=4 run_tasks.py --model ${model} --task ${task}
#     --num_landmarks ${lms}   --conv_kernel_size ${k_conv}  --window_size ${wsize} ${flags}
#     --weight_decay 1e-2  --num_layers ${layers}  --chk_path ${chkpath} --max_seq_len 1024
#     --dropout_prob 0.0 --attention_dropout 0.0 --learning_rate 1e-4 --rnn_dropout 0.5
#     --num_train_steps 624000000 --num_eval_steps 312 --eval_frequency 624 --batch_size 64 --warmup 624
#     --n_train_samples 159999 --n_dev_samples 20000 --n_test_samples 20000 --global_step 128 --num_classes 2 --dp_rank 64
#     --seed ${seed} --epoch 120
# "