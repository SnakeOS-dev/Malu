import os
import json
import numpy as np
from datasets import load_dataset
from tiktoken import get_encoding
from tqdm import tqdm

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENC = get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab

SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]
SPECIAL_IDS = {tok: VOCAB_SIZE + i for i, tok in enumerate(SPECIAL_TOKENS)}
TOTAL_VOCAB = VOCAB_SIZE + len(SPECIAL_TOKENS)


def tokenize_text(text):
    return ENC.encode_ordinary(text)


def format_conversation(conversation):
    tokens = []
    loss_mask = []
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            tokens.append(SPECIAL_IDS["<|user|>"])
            loss_mask.append(False)
            content_tokens = tokenize_text(content)
            tokens.extend(content_tokens)
            loss_mask.extend([False] * len(content_tokens))
        elif role == "assistant":
            tokens.append(SPECIAL_IDS["<|assistant|>"])
            loss_mask.append(False)
            content_tokens = tokenize_text(content)
            tokens.extend(content_tokens)
            loss_mask.extend([True] * len(content_tokens))
            tokens.append(SPECIAL_IDS["<|end|>"])
            loss_mask.append(True)
    return tokens, loss_mask


def process_split(split_name, output_name):
    print(f"Processando split: {split_name}")
    ds = load_dataset("nicholasKluge/instruct-aira-dataset-v3", split=split_name, trust_remote_code=True)
    print(f"Exemplos: {len(ds)}")
    all_tokens = []
    all_masks = []
    for example in tqdm(ds, desc=split_name):
        conv = example["conversations"]
        tokens, mask = format_conversation(conv)
        all_tokens.extend(tokens)
        all_masks.extend(mask)
    tokens_arr = np.array(all_tokens, dtype=np.uint16)
    masks_arr = np.array(all_masks, dtype=np.uint8)
    print(f"Total tokens: {len(tokens_arr):,}")
    tokens_arr.tofile(os.path.join(OUTPUT_DIR, f"{output_name}_tokens.bin"))
    masks_arr.tofile(os.path.join(OUTPUT_DIR, f"{output_name}_masks.bin"))
    return len(tokens_arr)


if __name__ == "__main__":
    n_pt = process_split("portuguese", "pt")
    n_en = process_split("english", "en")
    info = {
        "vocab_size": VOCAB_SIZE,
        "total_vocab": TOTAL_VOCAB,
        "special_tokens": SPECIAL_TOKENS,
        "special_ids": SPECIAL_IDS,
        "tokens_pt": n_pt,
        "tokens_en": n_en,
    }
    with open(os.path.join(OUTPUT_DIR, "tokenizer_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Pronto.")
