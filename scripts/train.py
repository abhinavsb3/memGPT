#call training code
from model_core.training import train_memgpt
from model_core.dataloader import DataLoaderLite

if __name__ == "__main__":
    config_path = "configs/config.json"
    print("Training starter")
    train_memgpt(config_path=config_path,dataloader_class=DataLoaderLite)