


"""
This file is from https://github.com/mlpen/Nystromformer
"""

import torch
import torch.nn as nn
import math
import numpy as np
import json
from torch.utils.checkpoint import checkpoint
import pdb
import lmu 
import dct
from torch.nn import functional as F

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        self.head_dim = config.head_dim

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        return V


class Attention(nn.Module):
    def __init__(self, config, maybe = 0):
        super().__init__()

        self.grad_checkpointing = config.attention_grad_checkpointing

        self.dim = config.transformer_dim
        self.head_dim = config.head_dim
        self.num_head = config.num_head
        self.seq_len = config.max_seq_len
        self.attn_type = config.attn_type
        self.dp_rank = config.num_landmarks

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
    

        self.lmufft =    lmu.LMUFFT_nccl(self.head_dim * self.num_head,  self.seq_len, config.rnn_dropout, self.num_head, transformer_dim = self.dim)
        self.lmufftq =    lmu.LMUFFT_nccl1(self.head_dim * self.num_head,  self.seq_len, config.rnn_dropout, self.num_head)

 
        
        
        
        self.dconv_fc = None

        if self.attn_type == "softmax":
            self.attn = SoftmaxAttention(config)
        elif self.attn_type == "none":
            self.attn = NoneAttention(config)
        elif self.attn_type.startswith("linformer"):
            from attention_linformer import LinformerAttention
            self.attn = LinformerAttention(config)

        elif self.attn_type.startswith("reformer"):
            from attention_reformer import LSHAttention
            self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
        elif self.attn_type.startswith("nystrom"):
            from attention_nystrom import NystromAttention
            self.attn = NystromAttention(config)
        elif self.attn_type.startswith("performer"):
            from attention_performer import PerformerAttention
            self.attn = PerformerAttention(config)
        elif self.attn_type.startswith("linear"):
            from attention_linear import LinearAttention
            self.attn = LinearAttention(config)
        elif self.attn_type.startswith("attention_SKT"):
            from attention_SKT import DynamicAttention_SKTAttention
            self.attn = DynamicAttention_SKTAttention(config, maybe = maybe)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask, cls_embed=None):

        if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(X.float(), mask.float())
        
        elif self.attn_type.startswith("attention_SKT"):            
            
            
            
            m,m = self.lmufft(X)

 
            Q = self.split_heads(self.W_q(m))

            K = self.split_heads(self.W_k(m))
            V = self.split_heads(self.W_v(m))
 
            
            
            with torch.cuda.amp.autocast(enabled = False):
                if self.grad_checkpointing:
                    attn_out = checkpoint(self.attn,(self.split_heads(X)).float(), Q.float(), K.float(), V.float(), mask.float(),cls_embed = cls_embed)
                else:
                    attn_out = self.attn((self.split_heads(X)).float(), Q.float(), K.float(), V.float(), mask.float(),cls_embed = cls_embed)
            attn_out = self.combine_heads(attn_out)
            
        else:
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            V = self.split_heads(self.W_v(X))

            with torch.cuda.amp.autocast(enabled = False):
                if self.grad_checkpointing:
                    attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                else:
                    attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
            attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X














# """
# This file is from https://github.com/mlpen/Nystromformer
# """

# import torch
# import torch.nn as nn
# import math
# import json
# from torch.utils.checkpoint import checkpoint
# import torch.nn.functional as F
# import pdb
# import dct
# import numpy as np

# class SoftmaxAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         #self.config = config
#         #self.bias = nn.Parameter(torch.zeros(config.num_head, config.max_seq_len, config.max_seq_len)).cuda()
#         #trunc_normal_(self.bias, std=.02) #0.02

#     def forward(self, Q, K, V, mask):
#         dot = torch.matmul(Q, torch.transpose(K, -2, -1))
#         #################  Position 1
#         #dot = dot + torch.transpose(dot,-2,-1) #+ self.bias
#         dot = dot / math.sqrt(2*self.head_dim)
#         #################  Position 2
#         #dot = dot + torch.transpose(dot,-2,-1) + self.bias
#         dot = dot - 1e6 * (1 - mask[:, None, None, :])
#         #################  Position 3
#         #dot = dot + torch.transpose(dot,-2,-1)+self.bias

#         attn = nn.functional.softmax(dot, dim = -1) #+ nn.functional.softmax(torch.transpose(dot, -2, -1), dim = -1)
#         attn = self.drop_attn(attn)
#         #print(mask.shape,mask[:, None, None, :].shape)
#         K1 = K * mask[:, None, :, None]
#         V1 = V * mask[:, None, :, None]
#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K1)
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         attn1 = torch.sigmoid(dot1)
#         attn1 = self.drop_attn(attn1)

#         X = torch.matmul(attn, V) + torch.matmul(V1, attn1)
#         #print("&&&&&&&&&&&&&&&&&&&&&&",self.config)
#         return X

# class FMMAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.config = config
#         self.alpha_1 = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()
#         self.alpha_1.data.fill_(0.)
#         self.alpha_2 = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()
#         self.alpha_2.data.fill_(0.)     
   

#     def forward(self, Q, K, V, mask):
#         #diag = 5
#         #dot = torch.matmul(Q, torch.transpose(K, -2, -1))
#         #dot = dot / math.sqrt(self.head_dim)
#         #dot = dot - torch.tril(dot,-diag) - torch.triu(dot,diag)
#         #dot = dot - 1e6 * (1 - mask[:, None, None, :])
#         band = 3
#         dot = torch.diag_embed(torch.sum(Q*K,dim=-1))
#         for i in range(1,band):
#             dot += torch.diag_embed(torch.sum(Q[:,:,i:,:]*K[:,:,:-i,:],dim=-1),-i)
#             dot += torch.diag_embed(torch.sum(Q[:,:,:-i,:]*K[:,:,i:,:],dim=-1),i)
#         dot = dot / math.sqrt(self.head_dim)
#         dot = dot - 1e6 * (1 - mask[:, None, None, :])
#         attn = nn.functional.softmax(dot, dim = -1)
#         attn = self.drop_attn(attn)
        
#         Q_1 = (F.elu(Q)+1)
#         K_1 = (F.elu(K)+1)*mask[:, None, :,None]
#         V_1 = (F.elu(V)+1)*mask[:, None, :,None]
#         dot_1 = torch.matmul(Q_1,torch.matmul(K_1.transpose(-2,-1),V_1))/(torch.matmul(Q_1,torch.sum(K_1,dim=-2,keepdim=True).transpose(-2,-1)))

#         Q_2 = (F.elu(-Q)+1)
#         K_2 = (F.elu(-K)+1)*mask[:, None, :,None]
#         V_2 = (F.elu(-V)+1)*mask[:, None, :,None]
#         dot_2 = torch.matmul(Q_2,torch.matmul(K_2.transpose(-2,-1),V_2))/(torch.matmul(Q_2,torch.sum(K_2,dim=-2,keepdim=True).transpose(-2,-1)))
#         X = torch.sigmoid(self.alpha_1)*torch.matmul(attn, V) + torch.sigmoid(self.alpha_2)*(dot_1 + dot_2)
#         return X



# class SigmoidAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.head_dim = config.head_dim
#         self.config = config
#         #self.bias = nn.Parameter(torch.zeros(config.num_head, config.max_seq_len, config.max_seq_len)).cuda()
#         #trunc_normal_(self.bias, std=.02) #0.02

#     def forward(self, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:, None, :, None]
#         dot = K*V
#         #dot = torch.matmul(Q, torch.transpose(K, -2, -1))
#         #dot = dot #/ math.sqrt(self.head_dim)
#         #dot = dot - 1e6 * (1 - mask[:, None, None, :])
#         attn = dot.sigmoid()
#         #attn = attn/self.config.max_seq_len
#         attn = self.drop_attn(attn)
#         X = attn * V
#         return X


# class SoftmaxAttention_V2(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.head_dim = config.head_dim
#         self.max_seq_len = config.max_seq_len
#         self.tau = nn.Parameter(torch.ones(config.num_head,1,1)).cuda()
#         #self.config = config
#         #self.bias = nn.Parameter(torch.zeros(config.num_head, config.max_seq_len, config.max_seq_len)).cuda()
#         #trunc_normal_(self.bias, std=.02) #0.02

#     def forward(self, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:, None, :, None]
#         attn1 = Q
#         attn2 = K.transpose(-2,-1)@K
#         attn3 = Q.transpose(-2,-1)/math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(attn1@attn2@attn3,dim=-1)
#         #attn = attn - 1e6 * (1 - mask[:, None, None, :])
#         attn = self.drop_attn(attn)
#        # Q = Q.transpose(-2, -1)
#         #K = K.transpose(-2, -1)
#         #V = V.transpose(-2, -1)

       
#         #attn = (Q @ K.transpose(-2, -1)) * self.tau
#         #attn = attn.softmax(dim=-1)
#         #attn = self.drop_attn(attn)

#         X = torch.matmul(attn,V)
#         #dot = torch.matmul(Q, torch.transpose(K, -2, -1))
#         #dot = dot / math.sqrt(self.head_dim)

#        # attn = nn.functional.softmax(dot, dim = -1) #+ nn.functional.softmax(torch.transpose(dot, -2, -1), dim = -1)
#        # attn = self.drop_attn(attn)
        
#         #dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         #dot1 = dot1 / math.sqrt(self.tau)
#         #attn1 = nn.functional.softmax(dot1, dim = -2)
#         #attn1 = self.drop_attn(attn1)

#         #X = torch.matmul(V, attn1.transpose(-2,-1))#torch.matmul(attn, V) + torch.matmul(V, attn1.transpose(-2,-1))
#         #print("&&&&&&&&&&&&&&&&&&&&&&",self.config)
#         return X

# class SoftmaxAttention_mul(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.head_dim = config.head_dim
#         self.temperature = nn.Parameter(torch.ones(config.num_head, 1, 1)).cuda()
#         #self.config = config
#         #self.bias = nn.Parameter(torch.zeros(config.num_head, config.max_seq_len, config.max_seq_len)).cuda()
#         #trunc_normal_(self.bias, std=.02) #0.02

#     def forward(self, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         dot = torch.matmul( torch.transpose(Q, -2, -1),K)
#         #dot = dot + torch.transpose(dot,-2,-1) + self.bias
#         dot = dot / math.sqrt(self.head_dim)
#         #dot = dot + torch.transpose(dot,-2,-1) + self.bias
#         #dot = dot #- 1e6 * (1 - mask[:, None, None, :])
#         #dot = dot + torch.transpose(dot,-2,-1)#+self.bias

#         attn = nn.functional.softmax(dot, dim = -1) #+ nn.functional.softmax(torch.transpose(dot, -2, -1), dim = -1)
#         attn = self.drop_attn(attn)

#         X = torch.matmul(V, attn)
#         #print("&&&&&&&&&&&&&&&&&&&&&&",self.config)
#         return X

# class DynamicAttention_V5(nn.Module):
#     def __init__(self, config, dp_rank=64):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.to_dynamic_projection = nn.Linear(self.dim, dp_rank * self.num_head)
#         self.to_dynamic_projection_1 = nn.Linear(self.dim, 32 * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.alpha_1 = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()
#         self.alpha_1.data.fill_(0.5)
#         self.alpha_2 = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()
#         self.alpha_2.data.fill_(0.5)

#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]

#         c_scores = self.to_dynamic_projection(X).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         c_scores = c_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms = K.transpose(-2,-1).matmul(c_scores)
#         # b x h x (lw) x r
#         dots = Q @ k_lms
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)
#         #attn_ = nn.functional.softmax(dots.transpose(-2,-1),dim=-1).transpose(-2,-1)
#         attn_ = torch.sigmoid(dots)
#         attn_ = self.drop_attn(attn_)
#         v_lms = c_scores.transpose(-2,-1)@ V

#         c_scores_1 = self.to_dynamic_projection_1(X).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)

#         c_scores_1 = c_scores_1.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms_1 = K.transpose(-2,-1).matmul(c_scores_1)
#         dots_1 = Q @ k_lms_1
#         attn_1 = nn.functional.softmax(dots_1,dim=-1)
#         attn_1 = self.drop_attn(attn_1)
#         #attn_1_ = nn.functional.softmax(dots_1.transpose(-2,-1),dim=-1).transpose(-2,-1)
#         attn_1_ = torch.sigmoid(dots_1)
#         attn_1_ = self.drop_attn(attn_1_)
#         v_lms_1 = c_scores_1.transpose(-2,-1)@ V

#         #alpha_1 = torch.sigmoid(self.alpha_1)
#         #alpha_2 = torch.sigmoid(self.alpha_2)
#         X = self.alpha_1*(attn@v_lms + attn_1@v_lms_1)+ self.alpha_2*(attn_ @ v_lms + attn_1_ @ v_lms_1)
#         return X

# class DynamicAttention_V6(nn.Module):
#     def __init__(self, config, dp_rank=16):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.to_dynamic_projection = nn.Linear(self.dim, dp_rank * self.num_head)
#         self.to_dynamic_projection_1 = nn.Linear(self.dim, 32 * self.num_head)
#         #self.to_dynamic_projection_2 = nn.Linear(self.dim, 8 * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.alpha = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()
#         self.alpha.data.fill_(0)

#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]

#         c_scores = self.to_dynamic_projection(X).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         c_scores = c_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms = K.transpose(-2,-1).matmul(c_scores)
#         # b x h x (lw) x r
#         dots = Q @ k_lms
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)
#         v_lms = c_scores.transpose(-2,-1)@ V

#         c_scores_1 = self.to_dynamic_projection_1(X).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)

#         c_scores_1 = c_scores_1.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms_1 = K.transpose(-2,-1).matmul(c_scores_1)
#         # b x h x (lw) x r
#         dots_1 = Q @ k_lms_1
#         dots_1 = dots_1 / math.sqrt(self.head_dim)
#         attn_1 = nn.functional.softmax(dots_1,dim=-1)
#         attn_1 = self.drop_attn(attn_1)
#         v_lms_1 = c_scores_1.transpose(-2,-1)@ V
 
#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         dot1 = dot1 / math.sqrt(self.head_dim)
#         attn1 = nn.functional.softmax(dot1,dim=-1).transpose(-2,-1)

        
#         X = torch.matmul(attn,v_lms) + torch.matmul(attn_1, v_lms_1) +torch.matmul(V, attn1)
#         return X


# class DynamicAttention_V4(nn.Module):
#     def __init__(self, config, dp_rank=16):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.to_dynamic_projection = nn.Linear(self.dim, dp_rank * self.num_head)
#         self.to_dynamic_projection_1 = nn.Linear(self.dim, 32 * self.num_head)
#         #self.to_dynamic_projection_2 = nn.Linear(self.dim, 8 * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.alpha = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()
#         self.alpha.data.fill_(0)

#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]

#         c_scores = self.to_dynamic_projection(X).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         c_scores = c_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms = K.transpose(-2,-1).matmul(c_scores)
#         # b x h x (lw) x r
#         dots = Q @ k_lms
#         #dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)
#         v_lms = c_scores.transpose(-2,-1)@ V

#         c_scores_1 = self.to_dynamic_projection_1(X).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)

#         c_scores_1 = c_scores_1.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms_1 = K.transpose(-2,-1).matmul(c_scores_1)
#         # b x h x (lw) x r
#         dots_1 = Q @ k_lms_1
#         #dots_1 = dots_1 / math.sqrt(self.head_dim)
#         attn_1 = nn.functional.softmax(dots_1,dim=-1)
#         attn_1 = self.drop_attn(attn_1)
#         v_lms_1 = c_scores_1.transpose(-2,-1)@ V

#         #c_scores_2 = self.to_dynamic_projection_2(X).contiguous().view(
#         #        X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         #c_scores_2 = c_scores_2.softmax(dim=-1, dtype=torch.float32).to(X)
#         #k_lms_2 = K.transpose(-2,-1).matmul(c_scores_2)
#         # b x h x (lw) x r
#         #dots_2 = Q @ k_lms_2
#         #dots = dots / math.sqrt(self.head_dim)
#         #attn_2 = nn.functional.softmax(dots_2,dim=-1)
#         #attn_2 = self.drop_attn(attn_2)
#         #v_lms_2 = c_scores_2.transpose(-2,-1)@ V

#         #alpha = torch.sigmoid(self.alpha)
#         X = torch.matmul(attn,v_lms) + torch.matmul(attn_1, v_lms_1) #+torch.matmul(attn_2, v_lms_2)
#         return X



# class DynamicAttention_V8(nn.Module):
#     def __init__(self, config, dp_rank=8):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.to_dynamic_projection_k = nn.Linear(self.dim, dp_rank * self.num_head)
#         self.to_dynamic_projection_v = nn.Linear(self.dim, dp_rank * self.num_head)

#         self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
#         self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
#         self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X


#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]

#         k_scores = self.to_dynamic_projection_k(self.combine_heads(K)).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         k_scores = k_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms = K.transpose(-2,-1).matmul(k_scores)
#         v_scores = self.to_dynamic_projection_v(self.combine_heads(V)).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         v_scores = v_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         v_lms = v_scores.transpose(-2,-1).matmul(V)
#         dots = Q @ k_lms
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)

        
#         Q1 = self.split_heads(self.W_q(X))
#         K1 = self.split_heads(self.W_k(X))
#         V1 = self.split_heads(self.W_v(X))

#         K1 = K1 * mask[:, None, :, None]
#         V1 = V1 * mask[:,None, :, None]

#         dot1 = torch.matmul( torch.transpose(Q1, -2, -1),K1)
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         attn1 = torch.sigmoid(dot1)
#         attn1 = self.drop_attn(attn1)
#         X = torch.matmul(attn,v_lms) + torch.matmul(V1, attn1)
#         return X

    
# class DynamicAttention_V7(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.dp_rank = config.dp_rank
#         self.to_dynamic_projection_k = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.to_dynamic_projection_v = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         #self.ln_1 = nn.Sequential(nn.LayerNorm(dp_rank*self.num_head),nn.GELU())#nn.LayerNorm(dp_rank)
#         #self.ln_2 = nn.Sequential(nn.LayerNorm(dp_rank*self.num_head),nn.GELU())#nn.LayerNorm(dp_rank)
#         #self.to_dynamic_projection_2 = nn.Linear(self.dim, 8 * self.num_head)
#         self.drop_attn_1 = torch.nn.Dropout(p=config.attention_dropout)
#         self.drop_attn_2 = torch.nn.Dropout(p=config.attention_dropout)
#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X


#     def forward(self,X, Q, K, V, mask):
        
#         Q = self.split_heads(dct.dct_real(self.combine_heads(Q),dim=1))
#         K = self.split_heads(dct.dct_real(self.combine_heads(K),dim=1))
#         V = self.split_heads(dct.dct_real(self.combine_heads(V),dim=1))
#         K = K * mask[:, None, :, None]
#         VV = V * mask[:,None, :, None]

#         k_scores = self.to_dynamic_projection_k(self.combine_heads(K)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         k_scores = k_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         #k_scores = self.ln_1(k_scores).contiguous().view(
#         #        X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)#.sigmoid()
#         k_lms = K.transpose(-2,-1).matmul(k_scores)


#         v_scores = self.to_dynamic_projection_v(self.combine_heads(VV)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         v_scores = v_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         #v_scores = self.ln_2(v_scores).contiguous().view(
#         #        X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)#.sigmoid()
#         v_lms = v_scores.transpose(-2,-1).matmul(VV)

#         dots = Q @ k_lms
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         #attn = torch.sigmoid(dots)
#         attn = self.drop_attn_1(attn)


# #         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
# # #         dot1 = dot1 / math.sqrt(self.seq_len)#math.sqrt(self.seq_len)#head_dim or seq_len
# # #         attn1 = torch.sigmoid(dot1)
# #         attn1 = nn.functional.softmax(dot1,dim=-1).transpose(-2,-1)
# #         attn1 = self.drop_attn_2(attn1)
#         #X = torch.max(torch.matmul(attn,v_lms), torch.matmul(V, attn1)) #+ #attn2*V#+torch.matmul(attn_2, v_lms_2)
#         X = torch.matmul(attn,v_lms) #+ torch.matmul(V, attn1)
#         return X


# class DynamicAttention_V77(nn.Module):
#     def __init__(self, config, maybe = 0):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.dp_rank = config.dp_rank
#         self.to_dynamic_projection_k = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.to_dynamic_projection_v = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.ln_1 = nn.LayerNorm(self.num_head * self.head_dim)
#         self.ln_2 = nn.LayerNorm(self.num_head * self.head_dim)
#         self.ln_3 = nn.LayerNorm(self.num_head * self.head_dim)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        
        
        
#         self.maybe = maybe


#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X

#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]
        
#         # Q =  self.split_heads(dct.dct_real(self.combine_heads(Q),dim=1))
#         # K = self.split_heads(dct.dct_real(self.combine_heads(K),dim=1))
#         # V = self.split_heads(dct.dct_real(self.combine_heads(V),dim=1))
#         # K  = Q
#         # V = Q

# #         k_scores = self.to_dynamic_projection_k(self.combine_heads(K)).contiguous().view(
# #                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
# #         k_scores = k_scores.softmax(dim=-1, dtype=torch.float32).to(X)
        
        
# #         k_lms = K.transpose(-2,-1).matmul(k_scores)
        
# #         print('k_scores shape: ',  k_scores.shape)
        
# #         print('k_lms shape: ',  k_lms.shape)
        
# #         print('K shape: ',  K.shape)
        
# #         print('self.dp_rank: ', self.dp_rank)
#         # break
#         # print((K[:,:,-self.dp_rank//2:-1,:].transpose(-2,-1)).shape, (K[:,:,:self.dp_rank//2,:].transpose(-2,-1)).shape)
#         k_lms =K[:,:, self.maybe : self.maybe + self.dp_rank, :].transpose(-2,-1)# K[:,:,-self.dp_rank//2-1:-1,:].transpose(-2,-1) + K[:,:,:self.dp_rank//2,:].transpose(-2,-1)
#         # print('k_lms shape: ',  k_lms.shape)
#         # k_lms = k_lms.transpose(-2,-1)
#         # print('k_lms shape: ',  k_lms.shape)
# #         
#         # print('-'*50)

#         # v_scores = self.to_dynamic_projection_v(self.combine_heads(V)).contiguous().view(
#         #        X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         # v_scores = v_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         # v_lms = (v_scores).transpose(-2,-1).matmul(V)
        
# #         print('V shape: ', V.shape)
# #         print('v_lms shape: ', v_lms.shape)
        
# #         print('-'*50)
#         v_lms  =  V[:,:,self.maybe : self.maybe + self.dp_rank,:] #V[:,:,:self.dp_rank//2,:]+  V[:,:,-self.dp_rank//2-1:-1,:]
#         dots = Q @ k_lms #/ math.sqrt(self.dim)
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         #attn = torch.sigmoid(dots)
#         attn = self.drop_attn(attn)
        
        
#         if self.dp_rank//self.num_head <= self.head_dim:
        
        
#             K = K[:,:,:,:self.dp_rank//self.num_head]
#             V_1 = V[:,:,:,:self.dp_rank//self.num_head]
#         else:
#             K = K
#             V_1  = V
#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         # # attn1 = torch.sigmoid(dot1)
#         attn1 = nn.functional.softmax(dot1,dim=-1).transpose(-2,-1)
#         # attn1 = self.drop_attn(attn1)
#         # X = self.split_heads(X) + self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,v_lms))))/2 + self.split_heads(self.ln_2(self.combine_heads(torch.matmul(V_1, attn1))))/2
#         # X =self.split_heads(self.ln_3(X)) -  self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,v_lms))))/2 - self.split_heads(self.ln_2(self.combine_heads(torch.matmul(V_1, attn1))))/2
#         # X =self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,v_lms))))
#         # X = self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,v_lms))))/
#         # X = self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,v_lms))))/2 + self.split_heads(self.ln_2(self.combine_heads(torch.matmul(V_1, attn1))))/2
#         # X = self.split_heads(X) - torch.matmul(attn,v_lms)/2 - torch.matmul(V_1, attn1)/2
#         X =  torch.matmul(attn,v_lms)/2 + torch.matmul(V_1, attn1)/2
#         # X = self.split_heads(self.ln_2(self.combine_heads(torch.matmul(X, attn1))))
#         # X = self.split_heads(dct.idct_real(self.combine_heads(X),dim=1))
        
#         # X = torch.matmul(attn,v_lms)
#         # X = self.split_heads(dct.idct_real(self.combine_heads(X),dim=1))
        
        
        
#         # X = dct.idct_real(torch.matmul(attn,v_lms),dim =-2)/2 + dct.idct_real(torch.matmul(V_1, attn1),dim =-2)/2
#         # X = self.combine_heads(X)
#         return X

    
    
# class DynamicAttention_SKTAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.dp_rank = config.dp_rank
#         self.to_dynamic_projection_k = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.to_dynamic_projection_v = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.ln_1 = nn.LayerNorm(self.num_head*self.head_dim)
#         self.ln_2 = nn.LayerNorm(self.num_head*self.head_dim) 
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        
        
#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X
    
#     def combine_heads_right(self, X):
#         # X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X
#     def split_heads_right(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         # X = X.transpose(1, 2)
#         return X
        
#     def forward(self,X, Q, K, V, mask):
#         #K = K * mask[:, None, :, None]
#         #V = V * mask[:,None, :, None]

#         k_scores = self.to_dynamic_projection_k(self.combine_heads(K)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         k_scores = k_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms = (K*mask[:, None, :, None]).transpose(-2,-1).matmul(k_scores*mask[:, None, :, None])
        

#         v_scores = self.to_dynamic_projection_v(self.combine_heads(V)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         v_scores = v_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         v_lms = (v_scores*mask[:, None, :, None]).transpose(-2,-1).matmul(V*mask[:, None, :, None])
        
#         dots = Q @ k_lms
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         #attn = torch.sigmoid(dots)
#         attn = self.drop_attn(attn)
        

#         dot1 = torch.matmul( torch.transpose(Q*mask[:, None, :, None], -2, -1),K*mask[:, None, :, None])
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         #attn1 = torch.sigmoid(dot1)
#         attn1 = nn.functional.softmax(dot1,dim=-1).transpose(-2,-1)
#         attn1 = self.drop_attn(attn1)
# #         X = 0.9*torch.matmul(attn,v_lms) + 0.1*torch.matmul(V, attn1)
#         V_tmp = torch.matmul(attn,v_lms)
#         X = torch.matmul(V_tmp, attn1)
#         return X


# class DynamicAttention_V12(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.dp_rank = config.dp_rank
#         self.to_dynamic_projection_k = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.to_dynamic_projection_v = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X

#     def forward(self,X, Q, K, V, mask,W_q_1,W_k_1,W_v_1):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]

#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         #attn1 = torch.sigmoid(dot1)
#         attn1 = nn.functional.softmax(dot1,dim=-1).transpose(-2,-1)
#         attn1 = self.drop_attn(attn1)
#         VV = torch.matmul(V, attn1)
        
#         VV = self.combine_heads(VV)
        
#         QQ = self.split_heads(W_q_1(VV))
#         KK = self.split_heads(W_k_1(VV))
#         VV = self.split_heads(W_v_1(VV))
        

#         k_scores = self.to_dynamic_projection_k(self.combine_heads(KK)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         k_scores = k_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms = K.transpose(-2,-1).matmul(k_scores)


#         v_scores = self.to_dynamic_projection_v(self.combine_heads(VV)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         v_scores = v_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         v_lms = v_scores.transpose(-2,-1).matmul(V)

#         dots = Q @ k_lms
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         #attn = torch.sigmoid(dots)
#         attn = self.drop_attn(attn)
#         X = torch.matmul(attn,v_lms)
#         return X

# class DynamicAttention_V11(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.dp_rank = config.dp_rank
#         self.to_dynamic_projection_k = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.to_dynamic_projection_v = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         #self.ln_1 = nn.Sequential(nn.LayerNorm(dp_rank*self.num_head),nn.GELU())#nn.LayerNorm(dp_rank)
#         #self.ln_2 = nn.Sequential(nn.LayerNorm(dp_rank*self.num_head),nn.GELU())#nn.LayerNorm(dp_rank)
#         #self.to_dynamic_projection_2 = nn.Linear(self.dim, 8 * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X


#     def forward(self,X, Q, K, V, mask):
#         KK = K * mask[:, None, :, None]
#         VV = V * mask[:,None, :, None]

#         k_scores = self.to_dynamic_projection_k(self.combine_heads(KK)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         k_scores = k_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms = KK.transpose(-2,-1).matmul(k_scores)


#         v_scores = self.to_dynamic_projection_v(self.combine_heads(VV)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         v_scores = v_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         #v_scores = self.ln_2(v_scores).contiguous().view(
#         #        X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)#.sigmoid()
#         v_lms = v_scores.transpose(-2,-1).matmul(VV)
#         dots = Q @ k_lms
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         #attn = torch.sigmoid(dots)
#         attn = self.drop_attn(attn)


#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),KK)
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         attn1 = torch.sigmoid(dot1)
#         attn1 = self.drop_attn(attn1)
#         X = torch.matmul(attn,v_lms) + torch.matmul(V, attn1)
#         return X
  

# class DynamicAttention_V10(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.dp_rank = config.dp_rank
#         self.to_dynamic_projection_k = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.to_dynamic_projection_v = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         #self.ln_1 = nn.Sequential(nn.LayerNorm(dp_rank*self.num_head),nn.GELU())#nn.LayerNorm(dp_rank)
#         #self.ln_2 = nn.Sequential(nn.LayerNorm(dp_rank*self.num_head),nn.GELU())#nn.LayerNorm(dp_rank)
#         #self.to_dynamic_projection_2 = nn.Linear(self.dim, 8 * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X


#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]

#         k_scores = self.to_dynamic_projection_k(self.combine_heads(Q)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         k_scores = k_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         k_lms = K.transpose(-2,-1).matmul(k_scores)


#         v_scores = self.to_dynamic_projection_v(self.combine_heads(Q)).contiguous().view(
#                X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         v_scores = v_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         #v_scores = self.ln_2(v_scores).contiguous().view(
#         #        X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)#.sigmoid()
#         v_lms = v_scores.transpose(-2,-1).matmul(V)
#         dots = Q @ k_lms
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         #attn = torch.sigmoid(dots)
#         attn = self.drop_attn(attn)


#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         #attn1 = nn.functional.softmax(dot1,dim=-1)#.transpose(-2,-1)
#         attn1 = torch.sigmoid(dot1)
#         attn1 = self.drop_attn(attn1)
#         #attn1 = nn.functional.softmax(dot1,dim=-1).transpose(-2,-1)
#         #attn2 = torch.sum(Q*K,dim=-1,keepdim=True)
#         X = torch.matmul(attn,v_lms) + torch.matmul(V, attn1) #+ #attn2*V#+torch.matmul(attn_2, v_lms_2)
#         return X



# class DynamicAttention_V9(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.dp_rank = config.dp_rank
#         self.pool = nn.AdaptiveAvgPool2d((self.dp_rank,self.head_dim))
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X


#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]

#         #P = self.combine_heads(Q)[:,:self.dp_rank,:].contiguous().view(
#         #       K.shape[0], self.dp_rank, self.num_head,-1).transpose(1,2)
#         P = self.pool(Q)

#         P_K = torch.sigmoid(P@K.transpose(-2,-1))#.softmax(dim=-1, dtype=torch.float32).to(X)
#         P_V = torch.sigmoid(P@V.transpose(-2,-1))#.softmax(dim=-1, dtype=torch.float32).to(X)
#         #print(self.combine_heads(Q).shape,P.shape,K.shape,P_K.shape)
#         k_lms = P_K.matmul(K)
#         v_lms = P_V.matmul(V)
#         dots = Q @ k_lms.transpose(-2,-1)
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)

#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         #attn1 = torch.sigmoid(dot1)
#         attn1 = nn.functional.softmax(dot1,dim=-1).transpose(-2,-1)
#         attn1 = self.drop_attn(attn1)
#         X = torch.matmul(attn,v_lms) + torch.matmul(V, attn1)
#         return X


# class DynamicAttention_V3(nn.Module):
#     def __init__(self, config, dp_rank=16):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.to_dynamic_projection = nn.Linear(self.dim, dp_rank * self.num_head)
#         #self.to_dynamic_projection = nn.Linear(self.dim, 8 * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.alpha = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()
#         self.alpha.data.fill_(0.)
        
#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]
#         #print(K.shape)
#         #print(X.shape,self.to_dynamic_projection)
#         c_scores = self.to_dynamic_projection(X).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)

#         c_scores = c_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         #c_scores = c_scores.sigmoid()
#         k_lms = K.transpose(-2,-1).matmul(c_scores)
#         # b x h x (lw) x r
#         dots = Q @ k_lms
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)
#         v_lms = c_scores.transpose(-2,-1)@ V

#         #attn_1 = nn.functional.softmax(dots.transpose(-2,-1),dim=-1).transpose(-2,-1)
#         #attn_1 = self.drop_attn(attn_1)
#         alpha = torch.sigmoid(self.alpha)
#         X = alpha*torch.matmul(attn,v_lms) #+ (1-alpha)*torch.matmul(attn_1, v_lms)
     
#         return X
       


# class DynamicAttention(nn.Module):
#     def __init__(self, config, dp_rank=16):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.to_dynamic_projection = nn.Linear(self.dim, dp_rank * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.alpha_1 = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()#torch.nn.Parameter(torch.randn(1)).sigmoid().cuda()
#         self.alpha_2 = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()#torch.nn.Parameter(torch.randn(1)).sigmoid().cuda()
#         self.alpha_1.data.fill_(1)
#         self.alpha_2.data.fill_(1)
#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]
#         #print(K.shape)
#         #print(X.shape,self.to_dynamic_projection)
#         c_scores = self.to_dynamic_projection(X).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)

#         c_scores = c_scores.softmax(dim=-1, dtype=torch.float32).to(X)
#         #c_scores = c_scores.sigmoid()
#         k_lms = K.transpose(-2,-1).matmul(c_scores)
#         # b x h x (lw) x r
#         dots = Q @ k_lms
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)
#         v_lms = c_scores.transpose(-2,-1)@ V
#         #X = torch.matmul(attn,v_lms)

#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         dot1 = dot1 / math.sqrt(self.head_dim)
#         attn1 = nn.functional.softmax(dot1, dim = -2).transpose(-2,-1)
#         attn1 = self.drop_attn(attn1)
#         X = torch.matmul(attn,v_lms) + torch.matmul(V, attn1)
#         return X

# class DynamicAttention_V2(nn.Module):
#     def __init__(self, config, dp_rank=16):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.to_dynamic_projection = nn.Linear(self.dim, dp_rank * self.num_head)
#         self.to_dynamic_projection_1 = nn.Linear(self.dim, dp_rank * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.alpha_1 = torch.nn.Parameter(torch.randn(1)).sigmoid().cuda()
#         self.alpha_2 = torch.nn.Parameter(torch.randn(1)).sigmoid().cuda()

#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def forward(self,X, Q, K, V, mask):
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]
#         #print(K.shape)
#         #print(X.shape,self.to_dynamic_projection)
#         c_scores = self.to_dynamic_projection(self.combine_heads(K)).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)

#         c_scores = c_scores.softmax(dim=-2, dtype=torch.float32).to(K)

#         c_scores_1 = self.to_dynamic_projection_1(self.combine_heads(V)).contiguous().view(
#                 X.shape[0], X.shape[1], self.num_head,-1).transpose(1,2)
#         c_scores_1 = c_scores_1.softmax(dim=-2, dtype=torch.float32).to(X)
         
#         k_lms = K.transpose(-2,-1).matmul(c_scores)
#         # b x h x (lw) x r
#         dots = Q @ k_lms
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)
#         v_lms = c_scores_1.transpose(-2,-1)@ V

#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         dot1 = dot1 / math.sqrt(self.head_dim)
#         attn1 = nn.functional.softmax(dot1, dim = -2).transpose(-2,-1)
#         attn1 = self.drop_attn(attn1)
#         X = torch.matmul(attn,v_lms) + torch.matmul(V, attn1)
#         #X = torch.matmul(attn,v_lms)
        
#         #q_lms_2 = Q.transpose(-2,-1).matmul(c_scores_1)
#         #print(k_lms_2.shape,Q.shape)
#         #k_lms_2 = c_scores_1.transpose(-2,-1)@ K
#         #dots_2 = q_lms_2 @ k_lms_2
#         #attn_2 = nn.functional.softmax(dots_2,dim=-1).transpose(-2,-1)
#         #v_lms_2 = c_scores_1.transpose(-2,-1)@ V
#         #X = X+ torch.matmul(V,attn_2)
#         #Q1 = self.to_dynamic_projection_1(torch.transpose(X,-2,-1))  ##
#         #dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         #dot1 = dot1 / math.sqrt(self.head_dim)
#         #attn1 = nn.functional.softmax(dot1, dim = -2).transpose(-2,-1)
#         #X = X+torch.matmul(V, attn1)
#         return X

# class SymmetricAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
#         self.head_dim = config.head_dim
#         #self.config = config
#         #self.bias = nn.Parameter(torch.zeros(config.num_head, config.max_seq_len, config.max_seq_len)).cuda()
#         #trunc_normal_(self.bias, std=.02) #0.02

#     def forward(self, Q, K, V, mask):
#         dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        
#         attn1 = dot / math.sqrt(self.head_dim)
#         attn1 = attn1 - 1e6 * (1 - mask[:, None, None, :])
#         attn1 = nn.functional.softmax(attn1, dim = -1) + nn.functional.sigmoid(attn1)
#         attn1 = self.drop_attn(attn1)
        
#         #attn2 = torch.transpose(dot, -2, -1)/ math.sqrt(self.head_dim)
#         #attn2 = attn2 - 1e6 * (1 - mask[:, None, None, :])
#         #attn2 = nn.functional.softmax(attn2, dim = -1) #+ nn.functional.softmax(torch.transpose(dot, -2, -1), dim = -1)
#         #attn2 = nn.functional.softmax(torch.transpose(attn2,-2,-1),dim=-1)
#         #attn2 = self.drop_attn(attn2)

#         X = torch.matmul(attn1, V) #torch.transpose(torch.matmul(torch.transpose(V,-2,-1), attn2),-2,-1)) #torch.matmul(attn2,V)
#         #print("&&&&&&&&&&&&&&&&&&&&&&",self.config)
#         return X

# class NoneAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#     def forward(self, Q, K, V, mask):
#         return V


# class Attention_Part1(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.dp_rank = config.dp_rank
#         self.to_dynamic_projection_k = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.to_dynamic_projection_v = nn.Linear(self.dim, self.dp_rank * self.num_head)
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        
#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X
    
#     def split_heads_sub(self, X, block_size = 4):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim//block_size, block_size)
#         X = X.transpose(1, 2)
#         return X
    
#     def combine_heads_sub(self, X, block_size = 4):
# # #         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         return X
        
#     def forward(self, Q, K, V, mask):
#         #print("part1 ###############################")
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]

#         k_scores = self.to_dynamic_projection_k(self.combine_heads(K)).contiguous().view(
#                Q.shape[0], -1, self.num_head, self.dp_rank).transpose(1,2)
#         k_scores = k_scores.softmax(dim=-1, dtype=torch.float32).to(Q)
#         #print(K.shape,k_scores.shape)
#         k_lms = K.transpose(-2,-1).matmul(k_scores)
        
#         v_scores = self.to_dynamic_projection_v(self.combine_heads(V)).contiguous().view(
#                Q.shape[0], -1, self.num_head,self.dp_rank).transpose(1,2)
#         v_scores = v_scores.softmax(dim=-1, dtype=torch.float32).to(Q)
#         v_lms = v_scores.transpose(-2,-1).matmul(V)
        
#         dots = Q @ k_lms
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         #attn = torch.sigmoid(dots)
#         attn = self.drop_attn(attn)
        
#         X = torch.matmul(attn,v_lms)
#         return X

# class Attention_Part2(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_head = config.num_head
#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.seq_len = config.max_seq_len
#         self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        
#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
#         X = X.transpose(1, 2)
#         return X
#     def forward(self, Q, K, V, mask):
#         #print("part2 *************************")
#         K = K * mask[:, None, :, None]
#         V = V * mask[:,None, :, None]
#         dot1 = torch.matmul( torch.transpose(Q, -2, -1),K)
#         dot1 = dot1 / math.sqrt(self.seq_len)
#         #attn1 = torch.sigmoid(dot1)
#         attn1 = nn.functional.softmax(dot1,dim=-1).transpose(-2,-1)
#         attn1 = self.drop_attn(attn1)
#         X = torch.matmul(V, attn1) #+ #attn2*V#+torch.matmul(attn_2, v_lms_2)
#         return X



# class Attention(nn.Module):
#     def __init__(self, config,maybe = 0):
#         super().__init__()
        
        
        
#         self.maybe = maybe
        
    
#         self.grad_checkpointing = config.attention_grad_checkpointing

#         self.dim = config.transformer_dim
#         self.head_dim = config.head_dim
#         self.num_head = config.num_head

#         self.attn_type = config.attn_type

#         self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
#         self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
#         self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

#         self.dconv_fc = None

#         if self.attn_type == "softmax":
#             self.attn = SoftmaxAttention(config)
#         elif self.attn_type == "fmmattention":
#             self.attn = FMMAttention(config)
#         elif self.attn_type == "sigmoid":
#             self.attn = SigmoidAttention(config)
#         elif self.attn_type == "softmax_v2":
#             self.attn = SoftmaxAttention_V2(config)
#         elif self.attn_type == "softmax_mul":
#             self.attn = SoftmaxAttention_mul(config)
#         elif self.attn_type == "symmetricsoftmax":
#             self.attn = SymmetricAttention(config)
#         elif self.attn_type == "dynamicsoftmax":
#             self.attn = DynamicAttention(config)
#         elif self.attn_type == "dynamicsoftmax_v2":
#             self.attn = DynamicAttention_V2(config)
#         elif self.attn_type == "dynamicsoftmax_v3":
#             self.attn = DynamicAttention_V3(config)
#         elif self.attn_type == "dynamicsoftmax_v4":
#             self.attn = DynamicAttention_V4(config)
#         elif self.attn_type == "dynamicsoftmax_v5":
#             self.attn = DynamicAttention_V5(config)
#         elif self.attn_type == "dynamicsoftmax_v6":
#             self.attn = DynamicAttention_V6(config)
#         elif self.attn_type == "dynamicsoftmax_v7":
#             self.attn = DynamicAttention_V7(config)
#         elif self.attn_type == "dynamicsoftmax_v77":
#             self.attn = DynamicAttention_V77(config, maybe = maybe)
#             # self.attn1 = DynamicAttention_V77(config, maybe = 256)
#             # self.attn2 = DynamicAttention_V77(config, maybe = 512)
#             # self.attn3 = DynamicAttention_V77(config, maybe = 768)
#         elif self.attn_type == "DynamicAttention_SKTAttention":
#             self.attn = DynamicAttention_SKTAttention(config)
#         elif self.attn_type == "dynamicsoftmax_v8":
#             self.attn = DynamicAttention_V8(config)
#         elif self.attn_type == "dynamicsoftmax_v9":
#             self.attn = DynamicAttention_V9(config)
#         elif self.attn_type == "dynamicsoftmax_v10":
#             self.attn = DynamicAttention_V10(config)
#         elif self.attn_type == "dynamicsoftmax_v11":
#             self.attn = DynamicAttention_V11(config)
#         elif self.attn_type == "dynamicsoftmax_v12":
#             self.attn = DynamicAttention_V12(config)
#             self.W_q_1 = nn.Linear(self.dim, self.num_head * self.head_dim)
#             self.W_k_1 = nn.Linear(self.dim, self.num_head * self.head_dim)
#             self.W_v_1 = nn.Linear(self.dim, self.num_head * self.head_dim)
        
#         elif self.attn_type == "attention_part1":
#             self.attn = Attention_Part1(config)
#         elif self.attn_type == "attention_part2":
#             self.attn = Attention_Part2(config)

#         elif self.attn_type == "none":
#             self.attn = NoneAttention(config)
#         elif self.attn_type.startswith("linformer"):
#             from attention_linformer import LinformerAttention
#             self.attn = LinformerAttention(config)

#         elif self.attn_type.startswith("reformer"):
#             from attention_reformer import LSHAttention
#             self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
#         elif self.attn_type.startswith("nystrom"):
#             from attention_nystrom import NystromAttention
#             self.attn = NystromAttention(config)
#         elif self.attn_type.startswith("performer"):
#             from attention_performer import PerformerAttention
#             self.attn = PerformerAttention(config)
#         elif self.attn_type.startswith("linear"):
#             from attention_linear import LinearAttention
#             self.attn = LinearAttention(config)

#         self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)
        
        
#         self.prenorm = nn.LayerNorm(self.num_head * self.head_dim)
#         self.dctnorm_q = nn.LayerNorm(self.head_dim)
#         self.dctnorm_k = nn.LayerNorm(self.head_dim)
#         self.dctnorm_v = nn.LayerNorm(self.head_dim)
# #         self.prenorm = nn.LayerNorm(self.num_head * self.head_dim)
#         self.lambda_prameter = nn.Parameter(torch.ones(1))
#         self.m = torch.nn.AvgPool1d(33, stride=1, padding=16, ceil_mode=False, count_include_pad=True)
#     def shrink(self,X):
#         sign_X = X.sign()
#         X = sign_X*torch.maximum(torch.abs(X)-1e-1*torch.abs(self.lambda_prameter),torch.zeros(X.shape).cuda())
#         return X
        
#     def split_heads_sub(self, X, block_size = 512):
#         X = X.reshape(X.size(0), block_size, X.size(1)//block_size, self.num_head, self.head_dim)
#         X = X.transpose(1, 3)
# #         (X.size(0),self.num_head , block_size, X.size(1)//block_size, self.head_dim)
#         X = X.transpose(3, 4)
# #         (X.size(0),self.num_head , block_size,  self.head_dim, X.size(1)//block_size)
#         return X
    
#     def combine_heads_sub(self, X, block_size = 512):
# # #         X = X.transpose(1, 2)
# #         X = X.transpose(1, 3)
#         X = X.transpose(3, 4)
#         X = X.reshape(X.size(0), self.num_head, -1, self.head_dim)
#         return X
    
#     def fft_self(self, X , dim = -1):
  
#         X = torch.fft.fft(X.float(),dim = dim, norm = 'ortho')
#         X = X.real - X.imag    
#         return X
        
 
#     def forward(self, X, mask):
#         dct_scalar = 1e2
# #         print('X shape:', X.shape)
#         # X = dct.dct_real(X,dim = 1)
#         # print((self.split_heads(X)).shape,X.shape)
#         # X = dct.dct_real(self.split_heads(X),dim =-1)
#         # X = self.combine_heads(X)
#         # X = dct.dct_real(X,dim = 2)
# #         print(torch.sum(torch.abs(X)))
# #         print('X new shape:', X.shape)
#         # XX = X.clone()
#         # X = dct.dct_real(dct.dct_real(X,dim=2)/dct_scalar,dim=1)/dct_scalar
#         if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
#             with torch.cuda.amp.autocast(enabled = False):
#                 attn_out = self.attn(X.float(), mask.float())
#         else:
            
            
            
# #             Q0 = dct.idct_real(dct.idct_real(self.W_q(X),dim = 1),dim = 2)
# #             Q = self.split_heads(Q0)
            
# #             K0 = dct.idct_real(dct.idct_real(self.W_k(X),dim = 1),dim = 2)
# #             K = self.split_heads(K0)
            
# #             V0 = dct.idct_real(dct.idct_real(self.W_v(X),dim = 1),dim = 2)
# #             V = self.split_heads(V0)
 
#             # X = torch.fft.fft(X.float(),dim = 1, norm = 'ortho')
#             # X = X.real - X.imag     
            
            
#             Q = self.split_heads(self.W_q(X))
          
#             # import time
#             # t0 = time.time()
#             # X = dct.dct_real(X,dim = 1)
#             # X = dct.dct_real(X,dim = 2)
#             X = dct.idct_real(X,dim = 2)
#             X = dct.idct_real(self.split_heads(X),dim =-1)
#             X = self.combine_heads(X)
#             # print(X.shape)
#             K = self.split_heads(self.W_k(X))
 
#             V = self.split_heads(self.W_v(X))
 
#             # print('time: ', time.time()-t0)
#             # print("*"*100)



#             # K = self.fft_self(K, dim = 1)
#             # Q = self.fft_self(Q, dim = 1)
#             # V = self.fft_self(V, dim = 1)
            
# #             if self.maybe == 0:

                

# # #                 # print(X.shape)
# #                 X = torch.fft.fft(X.float(),dim = 1, norm = 'ortho')
# #                 X = X.real - X.imag   
               
# # #                 # X = dct.dct_real(X,dim = 1)
# # #                 # X = dct.dct_real(self.split_heads(X),dim =-1)
# # #                 # X = self.combine_heads(X)

# #                 K = self.split_heads(self.W_k(X))
# #                 V = self.split_heads(self.W_v(X))
# #                 # X = dct.dct_real(X,dim = 2)
# #                 Q = self.split_heads(self.W_q(X))
# # #             elif self.maybe == 512:
# # #                 K = self.split_heads(self.W_k(X))
# # #                 V = self.split_heads(self.W_v(X))
# # #                 # X = dct.idct_real(self.split_heads(X),dim =-1)
# # #                 # X = dct.idct_real(X,dim = 1)
# # #                 # X = self.combine_heads(X)
# # #                 X = dct.idct_real(X,dim = 2)
# # #                 Q = self.split_heads(self.W_q(X))
# #             else:
# #                 K = self.split_heads(self.W_k(X))
# #                 V = self.split_heads(self.W_v(X))
# # #                 # X = dct.dct_real(self.split_heads(X),dim =-1)
# # #                 # X = dct.idct_real(self.split_heads(X),dim =-1)
# # #                 # X = dct.idct_real(X,dim = 1)
# # #                 # X = self.combine_heads(X)
# # #                 X = dct.idct_real(X,dim = 2)
# #                 Q = self.split_heads(self.W_q(X))
#                 # X = self.combine_heads(X)
            
#             # X = self.prenorm(X)
# #             X = self.shrink(X)

#             # Q0 = self.split_heads(self.W_q(X))
#             # K0 = self.split_heads(self.W_k(X))
#             # V0 = self.split_heads(self.W_v(X))
            
            
                       
#             # Q1 = self.split_heads_sub(self.W_q(X))
#             # K1 = self.split_heads_sub(self.W_k(X))
#             # V1 = self.split_heads_sub(self.W_v(X))
            
            
            
# #             Q_init = self.W_q(X)
# #             Q_smooth = self.m(Q_init.permute(0,2,1)).permute(0,2,1)
# #             Q_sparse = self.split_heads(Q_init-Q_smooth)
            
# #             K_init = self.W_k(X)
# #             K_smooth = self.m(K_init.permute(0,2,1)).permute(0,2,1)
# #             K_sparse = self.split_heads(K_init-K_smooth)
# #             V_init = self.W_v(X)
# #             V_smooth = self.m(V_init.permute(0,2,1)).permute(0,2,1)
# #             V_sparse = self.split_heads(V_init-V_smooth)
 
                       
#             # Q1 = self.split_heads_sub(Q_smooth)
#             # K1 = self.split_heads_sub(K_smooth)
#             # V1 = self.split_heads_sub(V_smooth)
# #             print(Q1.shape)
# #             Q1[:,0,:,:,:] = (dct.dct_real(dct.dct_real(Q1[:,0,:,:,:],dim=2),dim=3))#/dct_scalar
# #             Q1 = dct.idct_real(dct.idct_real(Q1,dim=4),dim=3)#/dct_scalar
            
# # #             Q1 = dct.dct_real(dct.dct_real(Q1,dim=3),dim=4)#,dim = 2)#/dct_scalar#             Q1[:,0,:,:,:] = torch.where(torch.abs(Q1[:,0,:,:,:])>1e2,Q1[:,0,:,:,:],Q1[:,0,:,:,:]*0)
# # #             Q1 = (dct.dct_real(dct.dct_real(Q1,dim=4),dim=2))#/dct_scalar
    
# # #             Q1[:,0,:,:,:] = torch.log( Q1[:,0,:,:,:]*Q1[:,0,:,:,:])#*torch.sign( Q1[:,0,:,:,:])
# # #             Q = dct.dct_real(dct.dct_real(dct.dct_real(Q1,dim=4)/dct_scalar,dim=3)/dct_scalar,dim = 2)/dct_scalar
# # #             Q = dct.dct_real(Q1,dim=3)/dct_scalar
# #             Q = self.combine_heads_sub(Q1)
# #             Q = self.dctnorm_q(Q)
# #             # print(Q0.shape)
# #             Q = dct.idct_real(dct.idct_real(Q0,dim=4),dim=3)
# # # #             print(Q.shape)
# #             K1 =dct.idct_real(dct.idct_real(K1,dim=4),dim=3)#/dct_scalar
# # #             K1  =dct.dct_real(dct.dct_real(K1 ,dim=3),dim=4)#/dct_scalar
# # #             K1[:,0,:,:,:] = torch.where(torch.abs(K1[:,0,:,:,:])>1e2,K1[:,0,:,:,:],K1[:,0,:,:,:]*0)
# # #             K[:,0,:,:] =  dct.dct_real(dct.dct_real(dct.dct_real(K1,dim=4)/dct_scalar,dim=3)/dct_scalar,dim = 2)/dct_scalar
# # #             K1[:,0,:,:,:] = torch.log( torch.abs(K1[:,0,:,:,:])+1.0)*torch.sign( K1[:,0,:,:,:])
# # #             K = dct.dct_real(K1,dim=3)/dct_scalar
# #             K = self.combine_heads_sub(K1)
# #             K = self.dctnorm_k(K)
# #             V1 = dct.idct_real(dct.idct_real(V1,dim=4),dim=3)#/dct_scalar
# # #             V1  =dct.dct_real(dct.dct_real(V1,dim=3),dim=4)#/dct_scalar
# # #             V1[:,0,:,:,:] = torch.where(torch.abs(V1[:,0,:,:,:])>1e2,V1[:,0,:,:,:],V1[:,0,:,:,:]*0)
# # #             V1[:,0,:,:,:] = torch.log( torch.abs(V1[:,0,:,:,:])+1.0)*torch.sign( V1[:,0,:,:,:])
# # #             V =  dct.dct_real(dct.dct_real(dct.dct_real(V1,dim=4)/dct_scalar,dim=3)/dct_scalar,dim = 2)/dct_scalar
# # #             V = dct.dct_real(V1,dim=3)/dct_scalar
# #             V = self.combine_heads_sub(V1)
# #             V = self.dctnorm_v(V)
# #             print(Q.shape)
# #             for i in range(Q0.shape[1]):
# #                 Q[:,i,:,:] = dct.dct_real(dct.dct_real((Q0[:,i,:,:]),dim=2)/dct_scalar,dim=1)/dct_scalar
# #             Q = self.shrink(self.prenorm(Q.permute(0,1,3,2))).permute(0,1,3,2)
# #             K = self.split_heads(dct.dct_real(dct.dct_real((self.W_k(X)),dim=2)/dct_scalar,dim=1)/dct_scalar)
# # #             K = self.shrink(self.prenorm(K.permute(0,1,3,2))).permute(0,1,3,2)
# #             for i in range(Q0.shape[1]):
# #                 K[:,i,:,:] = dct.dct_real(dct.dct_real((K0[:,i,:,:]),dim=2)/dct_scalar,dim=1)/dct_scalar
# #             V = self.split_heads(dct.dct_real(dct.dct_real((self.W_v(X)),dim=2)/dct_scalar,dim=1)/dct_scalar)
# #             for i in range(Q0.shape[1]):
# #                 V[:,i,:,:] = dct.dct_real(dct.dct_real((V0[:,i,:,:]),dim=2)/dct_scalar,dim=1)/dct_scalar
# # #             V = self.shrink(self.prenorm(V.permute(0,1,3,2))).permute(0,1,3,2)
# #             Q =  self.split_heads(self.prenorm(dct.dct_real(dct.dct_real(self.W_q(X),dim=2),dim=1)))
# #             K =  self.split_heads(self.prenorm(dct.dct_real(dct.dct_real(self.W_k(X),dim=2),dim=1)))
# #             V =  self.split_heads(self.prenorm(dct.dct_real(dct.dct_real(self.W_v(X),dim=2),dim=1)))
# #             Q = self.split_heads(dct.dct_real(self.W_q(X),dim=1)/dct_scalar)
                
# #             K = self.split_heads(dct.dct_real(self.W_k(X),dim=1)/dct_scalar)
# #             V = self.split_heads(dct.dct_real(self.W_v(X),dim=1)/dct_scalar)
# #             print(X.shape)
# #             Q = self.split_heads(dct.dct_real(self.W_q(X),dim=1)/dct_scalar)
            
# #             K = self.split_heads(dct.dct_real(self.W_k(X),dim=1)/dct_scalar)
# #             V = self.split_heads(dct.dct_real(self.W_v(X),dim=1)/dct_scalar)

# #             import pdb;pdb.set_trace()

# #             np.save('dct_K.npy',K[0,0,:,:].cpu().detach().numpy())
# #             np.save('dct_Q.npy',Q[0,0,:,:].cpu().detach().numpy())
# #             np.save('ori_Q.npy',Q0[0,0,:,:].cpu().detach().numpy())
# #             np.save('ori_K.npy',K0[0,0,:,:].cpu().detach().numpy())
            
# #             Q = Q0
# #             K = K0
# #             V = V0
            
# #             np.save('dct_K.npy',K[0,0,:,:].cpu().detach().numpy())
# #             np.save('dct_Q.npy',Q[0,0,:,:].cpu().detach().numpy())
# #             np.save('dct_K_1.npy',K[0,1,:,:].cpu().detach().numpy())
# #             np.save('dct_Q_1.npy',Q[0,1,:,:].cpu().detach().numpy())
            
#             #with torch.cuda.amp.autocast(enabled = False):
#             #    if self.grad_checkpointing:
#             #        attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
#             #    else:
#             #        attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())

#             if self.attn_type.startswith("dynamicsoftmax"):
#                 if self.attn_type == "dynamicsoftmax_v12":
#                     #Q1 = self.split_heads(self.W_q_1(X))
#                     #K1 = self.split_heads(self.W_k_1(X))
#                     #V1 = self.split_heads(self.W_v_1(X))
#                     with torch.cuda.amp.autocast(enabled = False):
#                         if self.grad_checkpointing:
#                             attn_out = checkpoint(X.float(), self.attn, Q.float(), K.float(), V.float(), mask.float(), self.W_q_1,self.W_k_1,self.W_v_1)
#                         else:
#                             attn_out = self.attn( X.float(), Q.float(), K.float(), V.float(), mask.float(), self.W_q_1,self.W_k_1,self.W_v_1)
#                 else:
#                     with torch.cuda.amp.autocast(enabled = False):
#                         if self.grad_checkpointing:
#                             attn_out = checkpoint(self.attn, X.float(), Q.float(), K.float(), V.float(), mask.float())
#                         else:
#                             attn_out = self.attn(X.float(), Q.float(), K.float(), V.float(), mask.float())
#                 # attn_out = dct.idct(attn_out)
                
                
      

#             else:
#                 with torch.cuda.amp.autocast(enabled = False):
#                     if self.grad_checkpointing:
#                         attn_out = checkpoint(self.attn,Q.float(), K.float(), V.float(), mask.float())
#                     else:
#                         attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())

#             # attn_out = dct.dct_real(attn_out,dim = -1)
#         # print((self.split_heads(X)).shape,X.shape)
#             # attn_out = self.split_heads(dct.idct_real(self.combine_heads(attn_out),dim =1))
#         # X = self.combine_heads(X)

#             attn_out = self.combine_heads(attn_out)
# #         print('asdfasdfasdf', attn_out.shape)
# #         attn_out = dct.idct_real(attn_out*dct_scalar,dim = 1)
# #         attn_out = dct.idct_real(dct.idct_real(attn_out*dct_scalar,dim = 1)*dct_scalar,dim = 2)
# #         attn_out = dct.idct_real(dct.idct_real(attn_out,dim = 1),dim = 2)
# #         print('hello world')




# #             Q1 = self.split_heads_sub(self.W_q(X))
# #             K1 = self.split_heads_sub(self.W_k(X))
# #             V1 = self.split_heads_sub(self.W_v(X))
# #         attn_out = self.split_heads_sub(attn_out)
# #             print(Q1.shape)
#         # attn_out = dct.dct_real(dct.dct_real(attn_out,dim=2),dim=1)
# #             Q = self.combine_heads_sub(Q)
# #             print(Q.shape)
# #             K = dct.dct_real(dct.dct_real(K1,dim=2)/dct_scalar,dim=3)/dct_scalar
# #             K = self.combine_heads_sub(K)
#             # V = dct.dct_real(dct.dct_real(V1,dim=2)/dct_scalar,dim=3)/dct_scalar
# #         attn_out = self.combine_heads_sub(attn_out)
# #         attn_out = self.combine_heads(attn_out)
#         out = self.ff(attn_out)

#         return out


#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_head , self.head_dim)
#         X = X.transpose(1, 2)
#         return X

# import torch
# import math
# import warnings


# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     # Cut & paste from PyTorch official master until it's in a few official releases - RW
#     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     def norm_cdf(x):
#         # Computes standard normal cumulative distribution function
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.

#     if (mean < a - 2 * std) or (mean > b + 2 * std):
#         warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#                       "The distribution of values may be incorrect.",
#                       stacklevel=2)

#     with torch.no_grad():
#         # Values are generated by using a truncated uniform distribution and
#         # then using the inverse CDF for the normal distribution.
#         # Get upper and lower cdf values
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)

#         # Uniformly fill tensor with values from [l, u], then translate to
#         # [2l-1, 2u-1].
#         tensor.uniform_(2 * l - 1, 2 * u - 1)

#         # Use inverse cdf transform for normal distribution to get truncated
#         # standard normal
#         tensor.erfinv_()

#         # Transform to proper mean, std
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)

#         # Clamp to ensure it's in the proper range
#         tensor.clamp_(min=a, max=b)
#         return tensor


# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     # type: (Tensor, float, float, float, float) -> Tensor
#     """Fills the input Tensor with values drawn from a truncated
#     normal distribution. The values are effectively drawn from the
#     normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
#     with values outside :math:`[a, b]` redrawn until they are within
#     the bounds. The method used for generating the random values works
#     best when :math:`a \leq \text{mean} \leq b`.
#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         mean: the mean of the normal distribution
#         std: the standard deviation of the normal distribution
#         a: the minimum cutoff value
#         b: the maximum cutoff value
#     Examples:
#         >>> w = torch.empty(3, 5)
#         >>> nn.init.trunc_normal_(w)
#     """
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)

    