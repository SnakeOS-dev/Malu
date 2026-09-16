import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, data_dir, split, block_size, train_ratio=0.98):
        self.block_size = block_size
        tokens_list = []
        masks_list = []
        for lang in ["pt", "en"]:
            tokens_path = os.path.join(data_dir, f"{lang}_tokens.bin")
            masks_path = os.path.join(data_dir, f"{lang}_masks.bin")
            if not os.path.exists(tokens_path):
                continue
            tokens_list.append(np.fromfile(tokens_path, dtype=np.uint16))
            masks_list.append(np.fromfile(masks_path, dtype=np.uint8))

        if not tokens_list:
            raise FileNotFoundError(f"Nenhum dado encontrado em {data_dir}")

        self.tokens = np.concatenate(tokens_list)
        self.masks = np.concatenate(masks_list)

        n = len(self.tokens)
        split_idx = int(n * train_ratio)
        if split == "train":
            self.tokens = self.tokens[:split_idx]
            self.masks = self.masks[:split_idx]
        else:
            self.tokens = self.tokens[split_idx:]
            self.masks = self.masks[split_idx:]

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1].astype(np.int64)
        mask_chunk = self.masks[idx:idx + self.block_size + 1].astype(bool)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        loss_mask = torch.from_numpy(mask_chunk[1:].copy())
        return x, y, loss_mask


def create_dataloaders(config):
    train_ds = TokenDataset(config.data.data_dir, "train", config.model.block_size)
    val_ds = TokenDataset(config.data.data_dir, "val", config.model.block_size)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
