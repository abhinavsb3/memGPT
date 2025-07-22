import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import os
import torch._dynamo

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000): 
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

    def apply_rotary_pos_emb(self, q, k):
        q_len = q.shape[2]
        k_len = k.shape[2]
        assert q.shape[-1] == self.dim, f"Expected q.shape[-1] == {self.dim}, got {q.shape[-1]}"
        assert k.shape[-1] == self.dim, f"Expected k.shape[-1] == {self.dim}, got {k.shape[-1]}"
        assert q_len <= self.max_seq_len, f"seq_len {q_len} exceeds max_seq_len {self.max_seq_len}"
        assert k_len <= self.max_seq_len, f"seq_len {k_len} exceeds max_seq_len {self.max_seq_len}"

        device = q.device
        cos_q, sin_q = self._get_freqs(q_len, device)  # both [seq_len, dim//2]
        cos_k, sin_k = self._get_freqs(k_len, device)  # both [seq_len, dim//2]
        

        # Expand to match q/k: [1, 1, seq_len, dim//2]
        cos_q = cos_q[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)
        sin_q = sin_q[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)
        cos_k = cos_k[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)
        sin_k = sin_k[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)

        def apply(x,cos, sin):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
    
            x_rotated_even = x1 * cos - x2 * sin
            x_rotated_odd = x1 * sin + x2 * cos
            return torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2)

        q_rot = apply(q, cos_q, sin_q)
        k_rot = apply(k, cos_k, sin_k)
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

        self.index_keys = np.zeros((max_memories, dim), dtype=np.float32)  
        self.index_size = 0  
        self.index_offset = 0  

    def add_to_db(self, new_data):
        new_data_len = new_data.shape[0] # B*T
        ids = (np.arange(new_data_len) + self.db_offset) % self.max_memories
        self.db[ids] = new_data.detach().cpu().numpy()
        self.db_offset = (self.db_offset + new_data_len) % self.max_memories
        self.db.flush()

    def add_keys_to_ring_buffer(self, keys):
        keys_len = keys.shape[0]
        
        for i in range(keys_len):
            pos = (self.index_offset + i) % self.max_memories
            
            self.index_keys[pos] = keys[i]
        
        self.index_offset = (self.index_offset + keys_len) % self.max_memories
        self.index_size = min(self.index_size + keys_len, self.max_memories)
        
        self.sync_faiss_with_ring_buffer()

    def sync_faiss_with_ring_buffer(self):
        self.index.reset()
        
        if self.index_size > 0:
            current_keys = self.index_keys[:self.index_size].copy()
            current_keys = np.ascontiguousarray(current_keys)
            self.index.add(current_keys)

    def search_and_retrieve(self, query_vecs, topk):
        distances, indices = self.index.search(query_vecs, topk)
        kvs = self.db[indices]
        return kvs

    def add(self, new_data):
        new_data = new_data.flatten(0, 1) #(B,T,2,C) --> (B*T,2,C)
        self.add_to_db(new_data)
        keys, vals = new_data.unbind(dim=-2) #(B*T,C)
        keys = keys.detach().cpu().numpy()
        keys = np.ascontiguousarray(keys)
        self.add_keys_to_ring_buffer(keys)

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
        self.index_keys[:] = 0
        self.index_size = 0
        self.index_offset = 0

    def cleanup(self):
        #call it after all training completed
        try:
            if os.path.exists(self.db_filepath):
                os.remove(self.db_filepath)
        except:
            pass

class XLAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, 'n_kv_head', config.n_head)  
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_head_dim = config.n_embd // self.n_kv_head  
        self.group_size = self.n_head // self.n_kv_head  
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.0)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.kv_head_dim)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.kv_head_dim)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.MEMGPT_SCALE_INIT = 1

        self.rope = RotaryPositionalEncoding(self.head_dim)

    def forward(self, x, xl_memory=None):
        B, T, C = x.size()

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, n_kv_head * kv_head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * kv_head_dim)

        # Handle XL memory
        if xl_memory is not None:
            k_xl, v_xl = xl_memory.unbind(dim=-2)
            k = torch.cat((k_xl, k), dim=1)
            v = torch.cat((v_xl, v), dim=1)
            xl_seq_len = k_xl.shape[1]

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, -1, self.n_kv_head, self.kv_head_dim).transpose(1, 2)  # (B, n_kv_head, T+xl, kv_head_dim) # GQAchange
        v = v.view(B, -1, self.n_kv_head, self.kv_head_dim).transpose(1, 2)  # (B, n_kv_head, T+xl, kv_head_dim) # GQAchange

        # Apply rotary positional encoding
        seq_len = k.shape[2]
        q, k = self.rope.apply_rotary_pos_emb(q, k)

        k = k.repeat_interleave(self.group_size, dim=1)  # (B, n_head, T+xl, kv_head_dim)
        v = v.repeat_interleave(self.group_size, dim=1)  # (B, n_head, T+xl, kv_head_dim)

        # Attention computation
        att = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        mask = torch.tril(torch.ones(T, seq_len, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, n_head, T, kv_head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        y = self.c_proj(y)

        # Prepare new XL memories - store original KV dimensions
        k_orig = k[:, ::self.group_size]  
        v_orig = v[:, ::self.group_size]  
        k_orig = k_orig.transpose(1, 2).contiguous().view(B, -1, self.n_kv_head * self.kv_head_dim) 
        v_orig = v_orig.transpose(1, 2).contiguous().view(B, -1, self.n_kv_head * self.kv_head_dim)  
        kv_memories = torch.stack((k_orig, v_orig), dim=-2)

        if xl_memory is not None:
            current_kv = kv_memories[:, -xl_seq_len:] #(B,T,2,C)
        else:
            current_kv = kv_memories #(B,T,2,C)

        return y, current_kv #(B,T,C),(B,T,2,C)

class KNNAttention(nn.Module):
    def __init__(self, config, knn, topk_retrieved_memories=3):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, 'n_kv_head', config.n_head)  
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_head_dim = config.n_embd // self.n_kv_head  
        self.group_size = self.n_head // self.n_kv_head  
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.0)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.kv_head_dim)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.kv_head_dim)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.MEMGPT_SCALE_INIT = 1

        self.gate_bias = nn.Parameter(torch.randn(self.n_head, 1, 1))
        self.topk_retrieved_memories = topk_retrieved_memories
        self.knn = knn

        self.rope = RotaryPositionalEncoding(self.head_dim)
    
    def forward(self, x, xl_memory=None):
        B, T, C = x.size()

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, n_kv_head * kv_head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * kv_head_dim)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Handle XL memory
        if xl_memory is not None:
            k_xl, v_xl = xl_memory.unbind(dim=-2)
            k = torch.cat((k_xl, k), dim=1)
            v = torch.cat((v_xl, v), dim=1)
            xl_seq_len = k_xl.shape[1]

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, -1, self.n_kv_head, self.kv_head_dim).transpose(1, 2)  # (B, n_kv_head, seq_len, kv_head_dim) # GQAchange
        v = v.view(B, -1, self.n_kv_head, self.kv_head_dim).transpose(1, 2)  # (B, n_kv_head, seq_len, kv_head_dim) # GQAchange

        seq_len = k.shape[2]
        q, k = self.rope.apply_rotary_pos_emb(q, k)

        k_expanded = k.repeat_interleave(self.group_size, dim=1)  # (B, n_head, seq_len, kv_head_dim)
        v_expanded = v.repeat_interleave(self.group_size, dim=1)  # (B, n_head, seq_len, kv_head_dim)

        # LOCAL ATTENTION
        att = (q @ k_expanded.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, seq_len, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        local_out = att @ v_expanded

        # KNN ATTENTION 
        if self.knn.index.ntotal > 0:
            q_search = q.transpose(1, 2).contiguous().view(B, T, C)
            mem_kv = self.knn.search(q_search, topk=self.topk_retrieved_memories)
            mem_k, mem_v = mem_kv.unbind(dim=-2)

            # Reshape memory K,V according to KV head structure
            mem_k = mem_k.view(B, T, self.topk_retrieved_memories, self.n_kv_head, self.kv_head_dim)
            mem_k = mem_k.permute(0, 3, 1, 2, 4)  # (B, n_kv_head, T, k, kv_head_dim)
            mem_v = mem_v.view(B, T, self.topk_retrieved_memories, self.n_kv_head, self.kv_head_dim)
            mem_v = mem_v.permute(0, 3, 1, 2, 4)  # (B, n_kv_head, T, k, kv_head_dim)
            mem_k = mem_k.to(q.device)
            mem_v = mem_v.to(q.device)

            # Expand memory K,V to match query heads
            mem_k_expanded = mem_k.repeat_interleave(self.group_size, dim=1)  # (B, n_head, T, k, kv_head_dim)
            mem_v_expanded = mem_v.repeat_interleave(self.group_size, dim=1)  # (B, n_head, T, k, kv_head_dim)

            mem_att = (q.unsqueeze(-2) @ mem_k_expanded.transpose(-2, -1)).squeeze(-2) * self.scale
            mem_att = F.softmax(mem_att, dim=-1)
            mem_att = self.dropout(mem_att)
            mem_out = (mem_att.unsqueeze(-2) @ mem_v_expanded).squeeze(-2)

            # Combine local and memory attention
            y = mem_out * self.gate_bias + local_out * (1 - self.gate_bias)
        else:
            y = local_out

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y) #(B,T,C)

        # Prepare new memories - store original KV dimensions
        k_orig = k.transpose(1, 2).contiguous().view(B, -1, self.n_kv_head * self.kv_head_dim) 
        v_orig = v.transpose(1, 2).contiguous().view(B, -1, self.n_kv_head * self.kv_head_dim)  
        kv_memories = torch.stack((k_orig, v_orig), dim=-2)

        if xl_memory is not None:
            current_kv = kv_memories[:, -xl_seq_len:] #(B,T,2,n_kv_head * kv_head_dim) # GQAchange
        else:
            current_kv = kv_memories #(B,T,2,C) 

        self.knn.add(current_kv)

        return y, current_kv #(B,T,C),(B,T,2,C)
    
    
        
        
        

    































