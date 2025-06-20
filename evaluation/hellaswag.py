"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

"""
import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F

DATA_DOWNLOADED_PATH = '"data/hellaswag"'

def download_file(url:str, fname:str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0 ))
    with open(fname, "wb") as file, tqdm(
        desc = fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024
    )as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """Downloads HellaSwag DATA_DOWNLOADED_PATH"""
    os.makedirs(DATA_DOWNLOADED_PATH, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_DOWNLOADED_PATH, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_DOWNLOADED_PATH, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def get_most_likely_row(tokens, mask, logits):
        shift_logits = (logits[..., :-1, :]).contiguous() #this will be x for loss calculation
        shift_tokens = (tokens[..., 1:]).contiguous() #this will be y for loss calculation
        shift_mask = (mask[..., 1:]).contiguous() #shifting same as tokens shifted
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        pred_norm = avg_loss.argmin().item() #taking the index of minimum loss
        return pred_norm   






