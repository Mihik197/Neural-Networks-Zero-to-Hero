"""
FineWeb-Edu dataset
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk
Run simply as:
$ python fineweb.py
Will save shards to local directory "edu_fineweb10B"
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm. library for displaying progress bars

# -------------------------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"  # name of dataset
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards

# create the cache local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)  # combines the directory of the current script with local_dir to create path
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")  # dataset comes pre-split into train and test sets

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a np array of uint16 tokens
    tokens = [eot]  # intialize token list with special <|endoftext|> token, delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))  # doc is a dictionary where the key "text" contains the text of the document
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)  # converts np array to uint16 type
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # writes a numpy array of uint16 tokens to a binary file
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)  # either half the number of available CPU cores or 1 (to avoid overwhelming the system)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # pre-allocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):  # applies tokenize() on elements of fw (16 for each process at a time)
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))

        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"  # create val set from 1st shard
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits into this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with leftovers of current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
