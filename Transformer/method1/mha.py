#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Multi-Headed Attention (MHA)

import math
from typing import Optional, List

import torch
from torch import nn as nn

from labml import tracker
from labml_helpers.module import Module


# In[ ]:


class PrepareForMultiHeadAttention(Module):
    
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)   # 将最后一个维度线性投影到heads*d_k维度
        self.heads = heads
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]   # 保存前两个维度的信息
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)   # 在最后一个维度进行分解
        return x   # x的形状为（seq_len, batch_size, heads, d_k）


# In[ ]:


class MultiHeadAttention(Module):
    
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None
        
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibhd,jbhd->ijbh', query, key)   # 求QK^T,形状为（seq_len, seq_len, batch_size, heads）
    
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):   # mask的形状为（seq_len_q, seq_len_k, batch_size）
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)   # 在最后一个维度增加一个维度，即head维度
        return mask
    
    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):   # query, key, value的形状都是（seq_len, batch_size, d_model）seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = self.get_scores(query, key)
        scores *= self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))   # 将mask的0的位置用负无穷代替
        attn = self.softmax(scores)
        tracker.debug('attn', attn)
        attn = self.dropout(attn)
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        self.attn = attn.detach()   # detach之后的attn永远不求梯度
        x = x.reshape(seq_len, batch_size, -1)
        return self.output(x)

