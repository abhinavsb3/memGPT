import torch
from ..hellaswag import render_example, iterate_examples, get_most_likely_row
import torch.distributed as dist 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from ..ModelGPT2 import GPT,log_file

ddp = int(os.environ.get('RANK', -1)) != -1 #will be True if ddp run
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #this is the process doing checkpoint,logging,etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #attempt to autodetect the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  #for mac users use apple silicon cpu which allready have gpu.mps is backend for apple silicon
    print(f"Using device: {device}")
# device = "cpu" #OVERRIDE

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


#Creating model by loading the model weights
checkpoint_path = '../log/model_final.pt' 
if master_process:
        print(f"Loading checkpoint from {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)

# Extract config and create model
model_config = checkpoint['config']
model_config.vocab_size = 50304 #for computational effciency(power of 2)
model = GPT(model_config)
# Load model state dict
model.load_state_dict(checkpoint['model'])
model = DDP(model, device_ids=[ddp_local_rank])
model.to(device)


def evaluate_hellaswag(model, device, device_type, ddp, ddp_rank, ddp_world_size, log_file, master_process):
         
    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        # only process example where i % ddp_world_size ==ddp_rank#this is for proper managemnt of which part is deal by which gpu
        if ddp:
            if i % ddp_world_size != ddp_rank:
                continue
        #rendering example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        #get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    #reduce the stats accross all process
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total #accuracy of hellaswag
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"Final Hellaswag accuracy: {acc_norm:.4f}\n")   

evaluate_hellaswag(model, device, device_type, ddp, ddp_rank, ddp_world_size, log_file, master_process)
if ddp:
    destroy_process_group()