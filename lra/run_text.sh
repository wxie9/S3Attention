function runexp {

gpu=${1}
task=${2}
model=${3}
chkpath=${4}
layers=${5}
lms=${6}      # lms (landmarks) is r in paper
k_conv=${7}
wsize=${8}    # wsize is w in paper
lr=${9}
wd=${10}
seed=${11}
flags=${12}

flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=${task}-newdata-${model}-l_${layers}-lr_${lr}-wd_${wd}-lms_${lms}-winsize_${wsize}-k_conv_${k_conv}-seed_${seed}${flags_print}-chkpath_${chkpath}

cmd="
 /home/pai/bin/python -m torch.distributed.launch --nproc_per_node=1 run_tasks.py --model ${model} --task ${task}
    --num_landmarks ${lms}   --conv_kernel_size ${k_conv}  --window_size ${wsize} ${flags}
    --weight_decay 0.01  --num_layers ${layers}  --chk_path ${chkpath}  --max_seq_len 1096 --batch_size 32 --learning_rate 1e-4
    --dropout_prob 0.1 --attention_dropout 0.1  --rnn_dropout 0.5 --warmup 2000  --num_train_steps 4000000  --dp_rank 8 --eval_frequency 781 --num_eval_steps 780
    --seed ${seed}
    --epoch 60 
"

# --embedding_dim 512 --transformer_dim 512 --head_dim 64 --num_head 8

debug=1
if [ ${debug} -eq 0 ]; then
cmd="${cmd} --logging --expname ${expname}  > logs/${expname}.log 2>&1 &"
else
cmd="${cmd} "
fi

echo logs/${expname}.log

eval ${cmd}

}
#attention_SKT
# The following hyperparameters correspond to Transformer-LS (w,r = 8,32) in the paper.
# One can change it to Transformer-LS (best) with lms = 1, win_size = 1
#runexp  gpu   task  model       chkpath   layers  lms  k_conv  win_size lr   wd   seed   flags
#runexp   7    text  linformer      "LRA_chks/linformer_softmax_text_base"    2       32    -1      8     1e-4  0.01 1234
#runexp   1    text  dynamicsoftmax_v7      "LRA_chks_best/text_dynamicsoftmax_v7"    2       32    -1      8     1e-4  0.01 1234
runexp   1    text  softmax      "LRA_chks/text_dynamic_v77_softmax_d_sigmoid_n_1_1015"    2       64    -1      8     1e-4  0.01 1234
