
#Setting up DDP
#torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
#run training loop

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import torch
import time
import json
import math
from model import GPT,GPTConfig


def train_memgpt(config_path,dataloader_class=None):    

    with open(config_path,'r') as f:
        cfg = json.load(f)
    
    model_cfg_params = cfg['model']
    train_cfg_params = cfg['training']

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        if master_process:
            print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    

        # --- Use loaded training parameters ---
    total_batch_size = train_cfg_params['total_batch_size']
    B = train_cfg_params['B']
    T = train_cfg_params['T']
    max_steps = train_cfg_params['max_steps']
    log_dir = train_cfg_params['log_dir']
    max_lr = train_cfg_params['max_lr']
    min_lr = train_cfg_params['min_lr']
    warmup_steps = train_cfg_params['warmup_steps']
    weight_decay = train_cfg_params['weight_decay']
    base_learning_rate = train_cfg_params['learning_rate'] 

    # total_batch_size = 524288
    # B = 64
    # T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = dataloader_class(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train",master_process=master_process)
    val_loader = dataloader_class(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val",master_process=master_process)

    torch.set_float32_matmul_precision('high')

    # Create Model
    model = GPT(GPTConfig(**model_cfg_params))
    model.to(device)
    use_compile = True
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=base_learning_rate, device_type=device_type, master_process=master_process)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    with open(log_file, "w") as f:
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step % 350 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

                checkpoint_name = f"model_final.pt" if last_step else f"model_{step:05d}.pt"
                checkpoint_path = os.path.join(log_dir, checkpoint_name)

                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'config': raw_model.config
                }
                torch.save(checkpoint, checkpoint_path)

               
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"Step:{step:5d} | Loss: {loss_accum.item():.6f} | lr: {lr:.4e} | Norm:{norm:.4f} | dt: {dt:.2f}ms | Tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()