import os
import numpy as np
import torch


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.current_shard = 0
        self.tokens = None
        self.current_position = 0
        self._load_and_prepare_tokens()

    def _load_and_prepare_tokens(self):
        """Loads a shard and reshapes it into B parallel streams."""
        # Load all tokens from the current shard file
        all_tokens_in_shard = load_tokens(self.shards[self.current_shard])

        # Distribute the tokens from the shard evenly among processes
        num_tokens_in_shard = all_tokens_in_shard.nelement()
        tokens_per_process = num_tokens_in_shard // self.num_processes
        start = self.process_rank * tokens_per_process
        end = start + tokens_per_process

        # Get the chunk of tokens this specific process is responsible for
        process_tokens = all_tokens_in_shard[start:end]

        # Arrange the data into B parallel streams (the crucial step)
        # We drop the small remainder of tokens that don't fit into a full batch
        num_batches_per_process = process_tokens.nelement() // (self.B * self.T)
        if num_batches_per_process == 0:
             # This can happen if a shard is too small for the given B, T, and num_processes
            self.tokens = torch.empty((self.B, 0), dtype=torch.long)
        else:
            num_tokens_to_use = num_batches_per_process * self.B * self.T
            process_tokens = process_tokens[:num_tokens_to_use]
            # Reshape into (B, num_batches * T)
            self.tokens = process_tokens.view(self.B, -1)#(B,num_batches_per_process*T)

        # Reset position to the start of the new streams
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        # If we've exhausted the tokens in the current shard, move to the next one
        if self.current_position + T + 1 > self.tokens.size(1):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self._load_and_prepare_tokens()
            # If the new shard is also too small, we might need to skip it
            if self.tokens.size(1) == 0:
                # This could be a recursive call or a loop, but for simplicity we'll just return None
                # and expect the training loop to handle it. A more robust implementation might loop
                # until a valid shard is found.
                print(f"WARNING: Shard {self.current_shard} is too small for process {self.process_rank}, skipping.")
                return self.next_batch()

        # Store the current shard number for return
        current_shard_num = self.current_shard

        # Slice the data to create the next batch
        # x is from all B streams, from the current position to T tokens ahead
        x = self.tokens[:, self.current_position : self.current_position + T] #doubt about if condition.and filling current batch
        # y is the target, shifted by one token
        y = self.tokens[:, self.current_position + 1 : self.current_position + T + 1]

        # Advance the position for the next batch
        self.current_position += T

        return x, y, current_shard_num

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self._load_and_prepare_tokens()