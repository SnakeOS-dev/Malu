import os
import sys
import time
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from malu.config import load_config
from malu.dataset import TokenDataset
from malu.model import GPT


def get_lr(iter_num, config):
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / (config.warmup_iters + 1)
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, dist.get_world_size()
    return 0, 0, 1


def is_master(rank):
    return rank == 0


def train(config_path):
    rank, local_rank, world_size = setup_ddp()
    config = load_config(config_path)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"

    if is_master(rank):
        os.makedirs(config.training.out_dir, exist_ok=True)

    train_ds = TokenDataset(config.data.data_dir, "train", config.model.block_size)
    val_ds = TokenDataset(config.data.data_dir, "val", config.model.block_size)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = GPT(config.model).to(device)
    optimizer = model.configure_optimizers(
        config.training.weight_decay,
        config.training.learning_rate,
        (config.training.beta1, config.training.beta2),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(config.training.dtype == "float16"))

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if is_master(rank):
        n_params = model.module.get_num_params() if world_size > 1 else model.get_num_params()
        print(f"Modelo com {n_params:,} parametros")

    start_iter = 0
    best_val = float("inf")
    ckpt_path = os.path.join(config.training.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = model.module.state_dict() if world_size > 1 else model.state_dict()
        state.update(ckpt["model"])
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter_num"]
        best_val = ckpt["best_val"]
        if is_master(rank):
            print(f"Retomando do iter {start_iter}")

    model.train()
    iter_num = start_iter
    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    while iter_num < config.training.max_iters:
        lr = get_lr(iter_num, config.training)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.training.gradient_accumulation_steps):
            try:
                x, y, mask = next(train_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(iter_num)
                train_iter = iter(train_loader)
                x, y, mask = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(config.training.dtype == "float16")):
                _, loss = model(x, y, loss_mask=mask)
                loss = loss / config.training.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += loss.item()
            running_count += 1

        if config.training.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if is_master(rank) and iter_num % config.training.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg_loss = running_loss / max(running_count, 1)
            print(f"iter {iter_num} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt:.2f}s")
            running_loss = 0.0
            running_count = 0

        if iter_num > 0 and iter_num % config.training.eval_interval == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for i, (x, y, mask) in enumerate(val_loader):
                    if i >= config.training.eval_iters:
                        break
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(config.training.dtype == "float16")):
                        _, loss = model(x, y, loss_mask=mask)
                    losses.append(loss.item())
            val_loss = sum(losses) / max(len(losses), 1)
            if is_master(rank):
                print(f"eval iter {iter_num} | val loss {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    state = model.module.state_dict() if world_size > 1 else model.state_dict()
                    torch.save({
                        "model": state,
                        "optimizer": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "best_val": best_val,
                    }, ckpt_path)
                    print(f"Salvo checkpoint em {ckpt_path}")
            model.train()

        iter_num += 1

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train(config_path)
