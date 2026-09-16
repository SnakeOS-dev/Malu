import json
import os
import tiktoken

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
SPECIAL_TOKENS = [USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN]


class MaluTokenizer:
    def __init__(self, info_path=None):
        self.enc = tiktoken.get_encoding("gpt2")
        self.base_vocab = self.enc.n_vocab
        if info_path and os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
            self.special_ids = info["special_ids"]
            self.total_vocab = info["total_vocab"]
        else:
            self.special_ids = {tok: self.base_vocab + i for i, tok in enumerate(SPECIAL_TOKENS)}
            self.total_vocab = self.base_vocab + len(SPECIAL_TOKENS)

    @property
    def vocab_size(self):
        return self.total_vocab

    @property
    def user_id(self):
        return self.special_ids[USER_TOKEN]

    @property
    def assistant_id(self):
        return self.special_ids[ASSISTANT_TOKEN]

    @property
    def end_id(self):
        return self.special_ids[END_TOKEN]

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        ids = [i for i in ids if i < self.base_vocab]
        return self.enc.decode(ids)
