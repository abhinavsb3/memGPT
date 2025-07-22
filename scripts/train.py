import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from model_core.training import train_memgpt
#from model_core.dataloader import DataLoader
from model_core.training import train_memgpt
from model_core.dataloader import DataLoader


if __name__ == "__main__":
    config_path = "configs/config.json"
    print("Training starter")
    train_memgpt(config_path=config_path,dataloader_class=DataLoader)

#to run the code using ddp: torchrun --nproc-per-node=1 train.py 