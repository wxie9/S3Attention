"""
--------------------------------------
"""

import torch
import torch.nn as nn
import math
import pdb

import dct
import lmu 

class DynamicAttention_SKTAttention(nn.Module):
    def __init__(self, config, maybe = 0):
        super().__init__()
        
        
        assert not (config.pooling_mode.lower() == 'cls' and config.cls_token)
        self.cls_from_seq = config.pooling_mode.lower() == 'cls'


        self.num_head = config.num_head
        self.dim = config.transformer_dim
        self.head_dim = config.head_dim
        self.seq_len = config.max_seq_len
        self.dp_rank = config.num_landmarks
        
        
        self.ln_1 = nn.LayerNorm(self.num_head * self.head_dim)
        self.ln_2 = nn.LayerNorm(self.num_head * self.head_dim)
      
        
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        
        

        self.index_set_right =   torch.randperm(self.head_dim)
        self.index_set_right = self.index_set_right[:self.dp_rank] 
        
     
        
        self.index_set_left =   torch.randperm(self.seq_len)
        self.index_set_left = self.index_set_left[:self.dp_rank]
        

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

       
        
    def forward(self,X, Q, K, V, mask,cls_embed=None):
                
        
        K = K * mask[:, None, :, None]
        V = V * mask[:,None, :, None]
                
        if cls_embed is not None:
            Q = torch.cat([self.split_heads(cls_embed),Q],dim = 2)
            K = torch.cat([self.split_heads(cls_embed),K],dim = 2)
            V = torch.cat([self.split_heads(cls_embed),V],dim = 2)
        
        
        if self.dp_rank <= self.seq_len:
            K1 = K[:,:,self.index_set_left,:]
            V1 = V[:,:,self.index_set_left,:]
        else:
            K1 = K
            V1 = V

            
        # batch, head_number, seq_len, hidden_dim
        dots = Q @ K1.transpose(-1,-2)  
        # batch, head_number, seq_len, sub_seq_len
        # batch, head_number, sub_seq_len, hiddem_dim
        dots = dots / math.sqrt(self.head_dim)
        attn = nn.functional.softmax(dots,dim=-1)
        attn = self.drop_attn(attn)
        
        #### right part ####        
        Q2 = Q.transpose(-1,-2)
     
        if self.dp_rank <= self.head_dim:

            K2 = K[:,:,:,self.index_set_right]
            V2 = V[:,:,:,self.index_set_right]
        else:
            K2 = K
            V2 = V
    
        dots_r = Q2 @ K2
        dots_r = dots_r / math.sqrt(self.seq_len)
        attn_r = nn.functional.softmax(dots_r,dim=-1).transpose(-1,-2)
        attn_r = self.drop_attn(attn_r)

        X =self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,V1))))/2 + self.split_heads(self.ln_2(self.combine_heads(torch.matmul(V2,attn_r))))/2
        
        
        return X


