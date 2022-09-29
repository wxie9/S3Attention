"""
"" Adapted from https://github.com/mlpen/Nystromformer
"" Refined by Xue & Jianqing
"""


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model", dest="model", required=True)
parser.add_argument("--task", type=str, help="task", dest="task", required = False)
parser.add_argument("--skip_train", type = int, help = "skip_train", dest = "skip_train", default = 0)
parser.add_argument("--logging", action='store_true', default=False)
parser.add_argument("--expname", type=str, default="default")
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--local_rank", type=int,default=0)



# # Model configs
# parser.add_argument("--attention_grad_checkpointing", default=False, action="store_true")
# parser.add_argument("--num_landmarks", default=128, type=int)
# parser.add_argument("--window_size", default=129, type=int)
# parser.add_argument("--conv_kernel_size", default=-1, type=int)
# parser.add_argument("--learn_pos_emb", default=1, type=int,
#                     help="Use 0 or 1 to represent false and true")
# parser.add_argument("--tied_weights", default=False, action="store_true")
# parser.add_argument("--embedding_dim", default=512, type=int)
# parser.add_argument("--transformer_dim", default=512, type=int)
# parser.add_argument("--transformer_hidden_dim", default=1024, type=int)
# parser.add_argument("--head_dim", default=32, type=int)
# parser.add_argument("--num_head", default=8, type=int)
# parser.add_argument("--num_layers", default=2, type=int)
# parser.add_argument("--vocab_size", default=512, type=int)
# parser.add_argument("--max_seq_len", default=4096, type=int)
# parser.add_argument("--dropout_prob", default=0.1, type=float)
# parser.add_argument("--attention_dropout", default=0.1, type=float)
# parser.add_argument("--pooling_mode", default="MEAN", type=str)
# parser.add_argument("--num_classes", default=2, type=int)
# parser.add_argument("--cls_token", default=False, action='store_true')
# parser.add_argument("--dp_rank", default=8, type=int) ###dynaminc V7

# Model configs
parser.add_argument("--attention_grad_checkpointing", default=False, action="store_true")
parser.add_argument("--num_landmarks", default=128, type=int)
parser.add_argument("--window_size", default=129, type=int)
parser.add_argument("--conv_kernel_size", default=-1, type=int)
parser.add_argument("--learn_pos_emb", default=1, type=int,
                    help="Use 0 or 1 to represent false and true")
parser.add_argument("--tied_weights", default=False, action="store_true")
parser.add_argument("--embedding_dim", default=64, type=int)
parser.add_argument("--transformer_dim", default=64, type=int)
parser.add_argument("--transformer_hidden_dim", default=128, type=int)
parser.add_argument("--head_dim", default=32, type=int)
parser.add_argument("--num_head", default=2, type=int)
parser.add_argument("--pooling_mode", default="MEAN", type=str)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--vocab_size", default=512, type=int)
parser.add_argument("--max_seq_len", default=4096, type=int)
parser.add_argument("--dropout_prob", default=0.1, type=float)
parser.add_argument("--attention_dropout", default=0.1, type=float)
parser.add_argument("--rnn_dropout", default=0.5, type=float)

parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--cls_token", default=False, action='store_true')
parser.add_argument("--dp_rank", default=8, type=int) ###dynaminc V7

# Training configs
parser.add_argument("--batch_size", default=32, type=int) ##32
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--warmup", default=8000, type=int)
parser.add_argument("--lr_decay", default="linear", type=str)
parser.add_argument("--fixed_lr", default=False, action='store_true')
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--adam_eps", default=1e-6, type=float)

parser.add_argument("--eval_frequency", default=500, type=int)
parser.add_argument("--num_train_steps", default=20000, type=int)
parser.add_argument("--num_eval_steps", default=781, type=int)
##############################################################
parser.add_argument("--global_step", default=128, type=int)
##############################################################
parser.add_argument("--fp32_attn", default=False, action='store_true')
parser.add_argument("--conv_zero_init", default=False, action='store_true')

# Dataset Configs
parser.add_argument("--n_train_samples", default=25000, type=int)
parser.add_argument("--n_dev_samples", default=25000, type=int)
parser.add_argument("--n_test_samples", default=25000, type=int)

parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--cls_last_layer", default=False, action='store_true')

parser.add_argument("--seed", default=1234, type=int)

parser.add_argument("--linformer_k", default=256, type=int)
parser.add_argument("--rp_dim", default=256, type=int)
parser.add_argument("--num_hash", default=2, type=int)
parser.add_argument("--chk_path", default="LRA_chks", type=str)
parser.add_argument("--test_flops", default=False, action='store_true')
# args = parser.parse_args()


if __name__ =="__main__":
    print(parser)

