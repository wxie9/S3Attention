
clear
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
    --weight_decay 0.01 --num_layers ${layers}  --chk_path ${chkpath} --max_seq_len 1024
    --dropout_prob 0.1 --attention_dropout 0.1 --rnn_dropout 0.5
    --num_train_steps 120000 --num_eval_steps 320 --eval_frequency 175 --batch_size 64 --learning_rate 1e-3 --warmup 175 --dp_rank 32
    --n_train_samples 35000 --n_dev_samples 5000 --n_test_samples 10000 --num_classes 10
    --seed ${seed}     --epoch 60  
"
#  --cls_token --embedding_dim 128 --transformer_dim 128 --head_dim 32 --num_head 4
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
# runexp  gpu   task    model               chkpath                                    layers  lms  k_conv  win_size lr   wd   seed   flags    
#runexp   6     image  dynamicsoftmax_v9    "LRA_chks/image_dynamicsoftmax_v9_r=256_softmax_lr=1e-3_step=35000_dropout=0.1"    2     32    -1      8     1e-4   0.0 1234
#runexp   3     image  dynamicsoftmax_v7    "LRA_chks/image_dynamicsoftmax_v7_r=256_sigmoid_lr=1e-3_step=8250_dropout=0.1_0922"    2     32    -1      8     1e-4   0.0 1234
runexp   0     image  attention_SKT    "LRA_chks/image_dynamic_v77_softmax_d_softmax_n_.1.9_warmup=175_r=256_1015"    4     32    -1      8     1e-4 0.0 1234
#runexp   2     image  dynamicsoftmax_v11    "LRA_chks/image_dynamicsoftmax_v11_r=256_sigmoid_lr=1e-3_step=10500_dropout=0.1_0930"    2     32    -1      8     1e-4   0.0 1234attention_SKT


# clear
# function runexp {

# gpu=${1}
# task=${2}
# model=${3}
# chkpath=${4}
# layers=${5}
# lms=${6}         # lms (landmarks) is r in paper
# k_conv=${7}
# wsize=${8}       # wsize is w in paper
# lr=${9}
# wd=${10}
# seed=${11}
# flags=${12}

# flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
# flags_print=${flags_print//--/_}

# expname=${task}-${model}-l_${layers}-lr_${lr}-wd_${wd}-lms_${lms}-winsize_${wsize}-k_conv_${k_conv}-seed_${seed}${flags_print}-chkpath_${chkpath}

# cmd="
#  /home/pai/bin/python -m torch.distributed.launch --nproc_per_node=4 run_tasks.py --model ${model} --task ${task}
#     --num_landmarks ${lms}   --conv_kernel_size ${k_conv}  --window_size ${wsize} ${flags}
#     --weight_decay 0.01 --num_layers ${layers}  --chk_path ${chkpath} --max_seq_len 1024
#     --dropout_prob 0.1 --attention_dropout 0.1 --rnn_dropout 0.5
#     --num_train_steps 120000 --num_eval_steps 320 --eval_frequency 175 --batch_size 64 --learning_rate 1e-3 --warmup 175 --dp_rank 32
#     --n_train_samples 35000 --n_dev_samples 5000 --n_test_samples 10000 --num_classes 10
#     --seed ${seed}     --epoch 60  
# "
# #  --cls_token --embedding_dim 128 --transformer_dim 128 --head_dim 32 --num_head 4
# debug=1
# if [ ${debug} -eq 0 ]; then
# cmd="${cmd} --logging --expname ${expname}  > logs/${expname}.log 2>&1 &"
# else
# cmd="${cmd} "
# fi

# echo logs/${expname}.log

# eval ${cmd}

# }

# # The following hyperparameters correspond to Transformer-LS (w,r = 8,32) in the paper.
# # One can change it to Tra:nsformer-LS (best) with lms=2, win_size=16
# # runexp  gpu   task    model               chkpath                                    layers  lms  k_conv  win_size lr   wd   seed   flags    
# #runexp   6     image  dynamicsoftmax_v9    "LRA_chks/image_dynamicsoftmax_v9_r=256_softmax_lr=1e-3_step=35000_dropout=0.1"    2     32    -1      8     1e-4   0.0 1234
# #runexp   3     image  dynamicsoftmax_v7    "LRA_chks/image_dynamicsoftmax_v7_r=256_sigmoid_lr=1e-3_step=8250_dropout=0.1_0922"    2     32    -1      8     1e-4   0.0 1234
# runexp   0     image  attention_SKT    "LRA_chks/image_dynamic_v77_softmax_d_softmax_n_.1.9_warmup=175_r=256_1015"    4     32    -1      8     1e-4 0.0 1234
# #runexp   2     image  dynamicsoftmax_v11    "LRA_chks/image_dynamicsoftmax_v11_r=256_sigmoid_lr=1e-3_step=10500_dropout=0.1_0930"    2     32    -1      8     1e-4   0.0 1234attention_SKT
