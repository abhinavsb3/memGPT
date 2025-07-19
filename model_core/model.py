import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect
from .attention import CasualSelfAttention
from attention import KNN,KNNAttention,XLAttention

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.MEMGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config, attention_layer):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = attention_layer
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, xl_memory=None):
        attn_out, new_xl_memory = self.attn(self.ln_1(x), xl_memory)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_xl_memory

@dataclass
class GPTConfig:
    block_size: int = 1024 
    vocab_size: int = 50257  
    n_layer: int = 12  
    n_head: int = 12  
    n_embd: int = 768 
    dropout: float = 0.0
    max_knn_memories: int = 81920
    topk_retrieved_memories: int = 3
    knn_layer_idx: int = 10  # which layer to use KNN attention (default: second last)

class GPT(nn.Module):
    def __init__(self, config, process_rank=0):
        super().__init__()
        self.config = config
        self.process_rank = process_rank

        # Initialize KNN memory
        self.knn = KNN(config.n_embd, config.max_knn_memories, process_rank)

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # Create blocks with different attention types
        for i in range(config.n_layer):
            if i == config.knn_layer_idx:
                attention_layer = KNNAttention(config, self.knn, config.topk_retrieved_memories)
            else:
                attention_layer = XLAttention(config)

            self.transformer.h.append(Block(config, attention_layer))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'MEMGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, xl_memories=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        x = self.transformer.wte(idx)

        if xl_memories is None:
            xl_memories = [None] * self.config.n_layer

        new_xl_memories = []

        for i, (block, xl_mem) in enumerate(zip(self.transformer.h, xl_memories)):
            x, new_xl_mem = block(x, xl_mem)
            if new_xl_mem is not None:
                new_xl_memories.append(new_xl_mem.detach())

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if len(new_xl_memories) > 0:
            return logits, loss, new_xl_memories
        else:
          return logits, loss
        
    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        # Get all parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optimization groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        if master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer    

    def clear_knn_memory(self):
        """Clear KNN memory (useful when starting a new document/sequence)"""
        self.knn.clear()

    def cleanup_memory_files(self):
        """Clean up memory files when training is done"""
        self.knn.cleanup()




























































# class MLP(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
#         self.gelu = nn.GELU(approximate='tanh')
#         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
#         self.c_proj.NANOGPT_SCALE_INIT = 1

#     def forward(self, x):
#         x = self.c_fc(x)
#         x = self.gelu(x)
#         x = self.c_proj(x)
#         return x
    

# class Block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = nn.LayerNorm(config.n_embd)
#         self.attn = CasualSelfAttention(config)
#         self.ln_2 = nn.LayerNorm(config.n_embd)
#         self.mlp = MLP(config)

#     def forward(self, x):
#         x = x + self.attn(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x


# @dataclass
# class GPTConfig:
#     block_size: int = 1024 #max sequence length
#     vocab_size: int = 50257 #number of tokens: 50000 BPE merges + 256 byte tokens +1 special token which is endoftext
#     n_layer: int = 12 #number of layers
#     n_head: int = 12 #number of heads
#     n_embd: int = 768 #embedding dimensions


# class GPT(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(config.vocab_size, config.n_embd),
#             wpe = nn.Embedding(config.block_size, config.n_embd),
#             h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
#             ln_f = nn.LayerNorm(config.n_embd),
#         ))

#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

#         #Weight sharing scheme
#         self.transformer.wte.weight = self.lm_head.weight

#         # init params
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             std = 0.02
#             if hasattr(module, 'NANOGPT_SCALE_INIT'):
#                 std *= (2 * self.config.n_layer) ** -0.5
#             torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def forward(self, idx, targets=None):
#         B, T = idx.size()
#         assert T <=self.config.block_size, f"Cannot forward sequence of length {T} ,block size is only {self.config.block_size}"

#         pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
#         pos_emb = self.transformer.wpe(pos)
#         tok_emb = self.transformer.wte(idx)
#         x = tok_emb + pos_emb

#         for block in self.transformer.h:
#             x = block(x)

#         x = self.transformer.ln_f(x)
#         logits = self.lm_head(x) #(B, T, vocab_size)
#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
#         return logits, loss
    
#     def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
#         param_dict = {pn:p for pn, p in self.named_parameters()}
#         param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
#         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#         optim_groups = [{'params':decay_params, ' weight_decay': weight_decay},
#                        {'params':nodecay_params, 'weight_decay': 0.0}
#                        ]
#         num_decay_params = sum(p.numel() for p in decay_params)  
#         num_nodecay_params = sum(p.numel() for p in nodecay_params) 
#         if master_process:
#             print(f"num decayed parameters tensors: {len(decay_params)}, with{num_decay_params}:parameters")
#             print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
#         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
#         use_fused = fused_available and device_type == "cuda"    
#         if master_process:
#             print(f"using fused AdamW: {use_fused}")
#         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
#         return optimizer 
    