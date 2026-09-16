import argparse
import torch
from malu.config import load_config
from malu.model import GPT
from malu.tokenizer import MaluTokenizer


def run_generate(config_path, prompt, max_new_tokens, temperature, top_k, ckpt_path):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MaluTokenizer("data/processed/tokenizer_info.json")
    model = GPT(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokens = [tokenizer.user_id] + tokenizer.encode(prompt) + [tokenizer.assistant_id]
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    out_ids = out[0].tolist()
    if tokenizer.end_id in out_ids:
        out_ids = out_ids[:out_ids.index(tokenizer.end_id)]
    out_ids = [i for i in out_ids if i != tokenizer.assistant_id and i != tokenizer.user_id]
    return tokenizer.decode(out_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    max_new_tokens = args.max_new_tokens or config.generation.max_new_tokens
    temperature = args.temperature if args.temperature is not None else config.generation.temperature
    top_k = args.top_k if args.top_k is not None else config.generation.top_k

    text = run_generate(args.config, args.prompt, max_new_tokens, temperature, top_k, args.ckpt)
    print(text)
