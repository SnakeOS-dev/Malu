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

TARGET_TOKENS = 500_000_000


def tokenize_text(text):
    return ENC.encode_ordinary(text)


def process_dataset(dataset_name, split, output_name, target_tokens=TARGET_TOKENS):
    print(f"Processando: {dataset_name} ({output_name})")
    ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
    all_tokens = []
    all_masks = []
    total = 0
    pbar = tqdm(total=target_tokens, desc=output_name, unit="tok")
    for example in ds:
        text = example["text"]
        tokens = tokenize_text(text)
        if total + len(tokens) > target_tokens:
            tokens = tokens[: target_tokens - total]
        all_tokens.extend(tokens)
        all_masks.extend([True] * len(tokens))
        total += len(tokens)
        pbar.update(len(tokens))
        if total >= target_tokens:
            break
    pbar.close()
    tokens_arr = np.array(all_tokens, dtype=np.uint16)
    masks_arr = np.array(all_masks, dtype=np.uint8)
    print(f"Total tokens: {len(tokens_arr):,}")
    tokens_arr.tofile(os.path.join(OUTPUT_DIR, f"{output_name}_tokens.bin"))
    masks_arr.tofile(os.path.join(OUTPUT_DIR, f"{output_name}_masks.bin"))
    return len(tokens_arr)


if __name__ == "__main__":
    n_pt = process_dataset("Polygl0t/gigaverbo-v2", "train", "pt")
    n_en = process_dataset("HuggingFaceFW/fineweb", "train", "en")
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
