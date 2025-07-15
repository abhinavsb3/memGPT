class Dataloader_memT:
    def __init__(self, B, T, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.num_processes = num_processes
        assert split in {'trai', 'val'}

        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shard found for the split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.current_shard = 0
        self.current_position = 0
        self.tokens = 0
        self._load_and_prepare_tokens()

    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def _load_and_prepare_tokens(self):
        all_tokens_in_shard = load_tokens(self.shards[self.current_shard])
        num_tokens_in_shard = all_tokens_in_shard.nelement()
        tokens_per_process = num_tokens_in_shard // self.num_processes
        start = self.process_rank * num_tokens_in_shard
        end = start + tokens_per_process

        process_tokens = all_tokens_in_shard[start:end]

        num_batches_per_process = process_tokens.nelement() // (self.B * self.T)
        
        if num_batches_per_process == 0:
            self.tokens = torch.empty((self.B , 0), dtype=torch.long)

        else:
            num_tokens_to_use = num_batches_per_process * self.B * self.T
            process_tokens = process_tokens[:num_tokens_to_use]
            self.tokens = self.process_tokens.view(self.B, -1)#(B,num_batches_per_process*T)

        self.current_position = 0

    def next_batch():
        B, T = self.B, self.T

        if self.current_position + T + 1 > self.tokens.size(1):
            print("called next shard")
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self._load_and_prepare_tokens()

            if self.tokens.size(1) == 0:
                print(f"WARNING: shard {self.current_shard} is too small for process {self.process_rank}, skipping." )
                return self.next_batch()

        current_shard_num = self.current_shard

        x = self.tokens[:, self.current_position : self.current_position + T]
        y = self.tokens[:,self.current_position + 1 : self.current_position + 1 + T]

        self.current_position += T
        return x, y, current_shard_num

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self._load_and_prepare_tokens()


                


        








































# import os
# import numpy as np
# import torch

# #Data loader
# class DataLoader_1:
#     def __init__(self, B, T, process_rank, num_processes, split, master_process):
#         self.B = B
#         self.T = T
#         self.process_rank = process_rank
#         self.num_processes = num_processes
#         assert split in {'train', 'val'}
        
        
#         data_root = "data/edu_fineweb10B"
#         shards = os.listdir(data_root)
#         shards = [s for s in shards if split in s]
#         shards = sorted(shards)
#         shards = [os.path.join(data_root, s) for s in shards]
#         self.shards = shards
#         assert len(shards)> 0, f"no shards found for split {split}"   
#         if master_process:
#             print(f"found {len(shards)} shards for split {split}")  
#         self.reset()

#     def load_tokens(self, filename):
#         npt = np.load(filename)
#         npt = npt.astype(np.int32)
#         ptt = torch.tensor(npt, dtype=torch.long)
#         return ptt
    

#     def reset(self):
#     #state, init at shard 0
#         self.current_shard = 0
#         self.tokens = self.load_tokens(self.shards[self.current_shard])
#         self.current_position = self.B * self.T * self.process_rank 

#     def next_batch(self):
#         B, T = self.B, self.T
#         buf = self.tokens[self.current_position:self.current_position + B*T+1]
#         x = (buf[:-1]).view(B,T) 
#         y = (buf[1:]).view(B,T) 
        
#         self.current_position += B * T * self.num_processes

#         if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
#             self.current_shard = (self.current_shard + 1) % len(self.shards)
#             self.tokens = self.load_tokens(self.shards[self.current_shard])
#             self.current_position = B * T * self.process_rank
#         return x, y


