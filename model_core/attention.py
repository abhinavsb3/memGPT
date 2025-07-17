import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import faiss
from einops import rearrange, einsum
from dataclasses import dataclass
import inspect
import os

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024, base=10000): 
        super().__init__()
        assert dim % 2 == 0

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [dim//2]
        self.register_buffer('inv_freq', inv_freq)

        self._cached_freqs = None
        self._cached_seq_len = 0

    def _get_freqs(self, seq_len, device):
        if self._cached_freqs is None or seq_len > self._cached_seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)  # [seq_len]
            freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
            cos = freqs.cos()  # [seq_len, dim//2]
            sin = freqs.sin()
            self._cached_freqs = (cos, sin)
            self._cached_seq_len = seq_len
        return self._cached_freqs[0][:seq_len], self._cached_freqs[1][:seq_len]

    def apply_rotary_pos_emb(self, q, k, seq_len):
        assert q.shape[-1] == self.dim, f"Expected q.shape[-1] == {self.dim}, got {q.shape[-1]}"
        assert k.shape[-1] == self.dim, f"Expected k.shape[-1] == {self.dim}, got {k.shape[-1]}"

        device = q.device
        cos, sin = self._get_freqs(seq_len, device)  # both [seq_len, dim//2]

        # Expand to match q/k: [1, 1, seq_len, dim//2]
        cos = cos[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)
        sin = sin[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)

        def apply(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            x_rotated_even = x1 * cos - x2 * sin
            x_rotated_odd = x1 * sin + x2 * cos
            return torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2)

        q_rot = apply(q)
        k_rot = apply(k)
        return q_rot, k_rot
    
class KNN():
    def __init__(self, dim, max_memories, process_rank=0):
        self.dim = dim
        self.max_memories = max_memories
        self.shape = (max_memories, 2, dim)
        self.db_offset = 0
        self.db_filepath = f"./memory_rank_{process_rank}.memmap"
        self.db = np.memmap(self.db_filepath, mode='w+', dtype=np.float32, shape=self.shape)
        self.index = faiss.IndexFlatL2(dim)
        self.process_rank = process_rank

    def add_to_db(self, new_data):
        new_data_len = new_data.shape[0] # B*T
        ids = (np.arange(new_data_len) + self.db_offset) % self.max_memories
        self.db[ids] = new_data.detach().cpu().numpy()
        self.db_offset = (self.db_offset + new_data_len) % self.max_memories
        self.db.flush()

    def search_and_retrieve(self, query_vecs, topk):
        distances, indices = self.index.search(query_vecs, topk)
        kvs = self.db[indices]
        return kvs

    def add(self, new_data):
        new_data = new_data.flatten(0, 1) #(B,T,2,C) --> (B*T,2,C)
        self.add_to_db(new_data)
        keys, vals = new_data.unbind(dim=-2) #(B,T,C)
        keys = keys.detach().cpu().numpy()
        keys = np.ascontiguousarray(keys)
        self.index.add(keys)

    def search(self, query_vecs, topk):
        query_batch_size, query_seq_len = query_vecs.shape[0], query_vecs.shape[1]
        query_vecs = query_vecs.flatten(0, 1) #(B,T,C) --> (B*T,C)
        kvs = self.search_and_retrieve(np.ascontiguousarray(query_vecs.detach().cpu().numpy()), topk)
        kvs = torch.tensor(kvs) #(B*T,TOPK,2,C)
        kvs = torch.unflatten(kvs, 0, (query_batch_size, query_seq_len)) #(B*T,TOPK,2,C) --> (B,T,TOPK,2,C)
        return kvs

    def clear(self):
        self.index.reset()
        self.db[:] = 0
        self.db_offset = 0

    def cleanup(self):
        #call it after all training completed
        try:
            if os.path.exists(self.db_filepath):
                os.remove(self.db_filepath)
        except:
            pass
        
        
        

    































# import torch.nn as nn
# from torch.nn import functional as F


# class CasualSelfAttention(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
#         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd)
#         self.c_proj.NANOGPT_SCALE_INIT = 1
#         self.n_head = config.n_head
#         self.n_embd = config.n_embd

#     def forward(self, x):
#         B, T, C = x.size()
#         qkv = self.c_attn(x)
#         q, k, v = qkv.split(self.n_embd, dim=2)
#         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
#         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
#         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

#         y = F.scaled_dot_product_attention(q, k, v, is_causal=True) #flash attention

#         y = y.transpose(1,2).contiguous().view(B, T, C) # (B, T, C) 
#         y = self.c_proj(y)
#         return y