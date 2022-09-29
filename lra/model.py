"""
This file is from https://github.com/mlpen/Nystromformer
"""

import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from attention import Attention
from attention_transformer_ls import AttentionLS
# from h_transformer_1d import HAttention1D
import pdb

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embedding_dim == config.transformer_dim

        self.dim = config.embedding_dim

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        torch.nn.init.normal_(self.word_embeddings.weight, std=0.02)

        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embedding_dim)
        torch.nn.init.normal_(self.position_embeddings.weight, std=0.02)

        if config.debug:
            self.word_embeddings.weight[-1].data[:] = 0
            self.position_embeddings.weight[0].data[:] = 0

        self.dropout = torch.nn.Dropout(p=config.dropout_prob)

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device=device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

class Transformer(nn.Module):
    def __init__(self, config, maybe = 0):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.transformer_dim)
        if config.attn_type == 'lsta':
            self.mha = AttentionLS(config)
        elif config.attn_type == 'htransformer':
            self.mha = HAttention1D(config)
        else:
            self.mha = Attention(config, maybe = maybe)
        self.dropout1 = torch.nn.Dropout(p=config.dropout_prob)
        self.norm2 = nn.LayerNorm(config.transformer_dim)
        self.debug = config.debug

        self.mlpblock = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=config.dropout_prob),
            nn.Linear(config.transformer_hidden_dim, config.transformer_dim),
            torch.nn.Dropout(p=config.dropout_prob)
        )

    def forward(self, X, mask, cls_embed=None):
        if self.config.attn_type == 'htransformer':
            mask = mask>0
        #print(X.shape,mask.shape)
        if cls_embed is None:
            X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        else:
            if cls_embed.shape[0] == 1:
                cls_embed = cls_embed.expand(X.shape[0], -1, -1)
            X_prepend = torch.cat([cls_embed, X], dim=1)
            if self.debug:
                cls_embed = self.norm1(cls_embed)
            #print(X.shape,mask.shape,cls_embed)
            # print(X,shape)
            # X = dct.dct(X)
            # print((self.mha(self.norm1(X), mask, cls_embed)).shape,X_prepend.shape,cls_embed.shape,X.shape)
            X = self.dropout1(self.mha(self.norm1(X), mask, cls_embed)) + X_prepend
        X = self.mlpblock(self.norm2(X)) + X
        return X


class Transformer_1(nn.Module):
    def __init__(self, config,maybe = 0):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.transformer_dim)
        if config.attn_type == 'lsta':
            self.mha = AttentionLS(config)
        else:
            config.attn_type = 'attention_part1'
            self.mha_1 = Attention(config, maybe = maybe)
            config.attn_type = 'attention_part2'
            self.mha_2 = Attention(config, maybe = maybe)
        self.dropout1 = torch.nn.Dropout(p=config.dropout_prob)
        self.norm2 = nn.LayerNorm(config.transformer_dim)
        self.norm3 = nn.LayerNorm(config.transformer_dim)
        self.norm4 = nn.LayerNorm(config.transformer_dim)
        self.debug = config.debug

        self.mlpblock_1 = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=config.dropout_prob),
            nn.Linear(config.transformer_hidden_dim, config.transformer_dim),
            torch.nn.Dropout(p=config.dropout_prob)
        )
        self.mlpblock_2 = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=config.dropout_prob),
            nn.Linear(config.transformer_hidden_dim, config.transformer_dim),
            torch.nn.Dropout(p=config.dropout_prob)
        )

    def forward(self, X, mask, cls_embed=None):
        if cls_embed is None:
            X = self.dropout1(self.mha_1(self.norm1(X), mask)) + X
            X = self.mlpblock_1(self.norm3(X)) + X
            X = self.dropout1(self.mha_2(self.norm2(X), mask)) + X
            X = self.mlpblock_2(self.norm4(X)) + X
        else:
            if cls_embed.shape[0] == 1:
                cls_embed = cls_embed.expand(X.shape[0], -1, -1)
            X_prepend = torch.cat([cls_embed, X], dim=1)
            if self.debug:
                cls_embed = self.norm1(cls_embed)
            #print(X.shape,mask.shape,cls_embed)
            print(X.shape)
            X = self.dropout1(self.mha_2(self.norm1(X), mask, cls_embed)) + X_prepend
            X = self.mlpblock_1(self.norm3(X)) + X
            X = self.dropout1(self.mha_1(self.norm2(X), mask, cls_embed)) + X
            #X = self.mlpblock_1(self.norm3(X)) + X
            X = self.mlpblock_2(self.norm4(X)) + X
        return X


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config.num_layers
        self.tied_weights = config.tied_weights
        self.cls_last_layer = config.cls_last_layer

        self.embeddings = Embeddings(config)

        if config.cls_token or self.cls_last_layer:
            self.cls_embed = nn.Parameter(torch.zeros(1, 1, config.transformer_dim))
        else:
            self.cls_embed = None

        if self.tied_weights:
            if config.attn_type == 'attention_part':
                self.transformer = Transformer_1(config)
            else:
                self.transformer = Transformer(config)
        else:
            foo = config.num_landmarks * 1
            if config.attn_type == 'attention_part':
                for idx in range(self.num_layers):
                    setattr(self, f"transformer_{idx}", Transformer_1(config,maybe = foo*idx))
                    print('maybe is: ', foo*idx)
            else:
                for idx in range(self.num_layers):
                    setattr(self, f"transformer_{idx}", Transformer(config,maybe = foo*idx))
                    print('maybe is: ', foo*idx)

        self.norm = nn.LayerNorm(config.transformer_dim)

    def forward(self, input_ids, mask=None):
        X = self.embeddings(input_ids)
        cls_embed = self.cls_embed if not self.cls_last_layer else None

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            for idx in range(self.num_layers):
                if self.cls_last_layer and idx == self.num_layers - 1:
                    cls_embed = self.cls_embed
                X = self.transformer(X, mask, cls_embed)
                if cls_embed is not None:
                    # We always prepend the cls token into the first token
                    cls_embed = X[:, :1]
                    X = X[:, 1:]
        else:
            for idx in range(self.num_layers):
                if self.cls_last_layer and idx == self.num_layers - 1:
                    cls_embed = self.cls_embed
                X = getattr(self, f"transformer_{idx}")(X, mask, cls_embed)
                if cls_embed is not None:
                    # We always prepend the cls token into the first token
                    cls_embed = X[:, :1]
                    X = X[:, 1:]

        if cls_embed is not None:
            cls_embed = self.norm(cls_embed)
            return cls_embed
        else:
            X = self.norm(X) * mask[:, :, None]
            return X
