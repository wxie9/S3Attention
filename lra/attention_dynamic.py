import torch
import torch.nn as nn
class DynamicAttention(nn.Module):
    def __init__(self, config, dp_rank=3):
        super().__init__()
          
        self.num_head = config.num_head
        self.dim = config.transformer_dim
        self.head_dim = config.head_dim
        self.seq_len = config.max_seq_len
        self.to_dynamic_projection = nn.Linear(self.dim, dp_rank * self.num_head)
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

    def forward(self,x, Q, K, V, mask):
        c_scores = self.to_dynamic_projection(x).contiguous().view(
                x.shape[0], x.shape[1], self.num_head,-1).transpose(1,2)
        
        c_scores = c_scores.softmax(dim=-1, dtype=torch.float32).to(x)
        
        k_lms = K.transpose(-2,-1).matmul(c_scores)
        # b x h x (lw) x r
        dots_all = Q @ k_lms
        attn = nn.functional.softmax(dots,dim=-1)
        attn = self.drop_attn(attn)
        v_lms = c_scores.transpose(-2,-1)@ V
        X = torch.matmul(attn,V)
        return X
