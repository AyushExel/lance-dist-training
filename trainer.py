import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import wandb
import numpy as np
import time
from tqdm import tqdm

from PIL import Image
from lance.torch.data import (
    LanceDataset,
    SafeLanceDataset,
    get_safe_loader,
    ShardedBatchSampler,
    ShardedFragmentSampler,
    FullScanSampler
)

from modelling.get_model_and_loss import get_model_and_loss

# Standard CIFAR normalization
_cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Standard normalization for ImageNet/Places365
_places_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def decode_tensor_image(batch, **kwargs):
    images = []
    labels = []
    for item in batch.to_pylist():
        arr = np.frombuffer(item["image"], dtype=np.uint8).reshape(224, 224, 3)
        img = Image.fromarray(arr)
        tensor = _places_transform(img)
        images.append(tensor)
        labels.append(item["label"])
    batch = {
        "image": torch.stack(images),
        "label": torch.tensor(labels, dtype=torch.long)
    }
    return batch




def get_dataset(dataset_path, use_safe, batch_size):
    return SafeLanceDataset(dataset_path, to_tensor_fn=decode_tensor_image, batch_size=batch_size) if use_safe else LanceDataset(dataset_path, to_tensor_fn=decode_tensor_image, batch_size=batch_size)


def get_sampler(dataset, sampler_type, batch_size, rank, world_size):
    if sampler_type == "sharded_batch":
        return ShardedBatchSampler(dataset, world_size=world_size)
    elif sampler_type == "sharded_fragment":
        return ShardedFragmentSampler(dataset, world_size=world_size)
    elif sampler_type == "full_scan":
        return FullScanSampler(dataset)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


def get_loader(dataset, sampler, use_safe, num_workers):
    if use_safe:
        return get_safe_loader(dataset, sampler=sampler, num_workers=num_workers, batch_size=None)
    if isinstance(dataset, torch.utils.data.IterableDataset):
        return DataLoader(dataset, batch_size=None, num_workers=num_workers)
    if isinstance(sampler, torch.utils.data.BatchSampler):
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, batch_size=None)
    return DataLoader(dataset, sampler=sampler, num_workers=num_workers, batch_size=None)

def train(rank, world_size, args):
    is_distributed = not getattr(args, "no_ddp", False)
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        print("Warning: CUDA not available. Running on CPU. This will be slow.")
        device = torch.device("cpu")

    dataset = get_dataset(args.dataset_path, args.use_safe_loader, args.batch_size)
    sampler = get_sampler(dataset, args.sampler_type, args.batch_size, rank, world_size)
    loader = get_loader(dataset, sampler, args.use_safe_loader, args.num_workers)

    model, loss_fn, eval_fn = get_model_and_loss(args.task_type, args.num_classes)
    model = model.to(device)

    if is_distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[rank])
        else:
            model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if rank == 0:
        wandb.init(project="lance-ddp-generic", config=vars(args))

    total_start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        batch_iter = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        for batch in batch_iter:
            import pdb; pdb.set_trace()
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_iter.set_postfix(loss=loss.item())

        epoch_time = time.time() - epoch_start_time
        if rank == 0 and epoch % 10 == 0:
            val_acc = eval_fn(model, loader, device)
            wandb.log({"val_acc": val_acc, "epoch_time": epoch_time})
            print(f"[Epoch {epoch}] Rank {rank} Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}, Epoch Time: {epoch_time:.2f}s")

        if rank == 0:
            wandb.log({"epoch": epoch, "loss": total_loss})
            print(f"[Epoch {epoch}] Rank {rank} Loss: {total_loss:.4f}")

    total_time = time.time() - total_start_time
    if rank == 0:
        print(f"Total training time: {total_time:.2f} seconds")
        wandb.log({"total_training_time": total_time})

    if is_distributed:
        dist.destroy_process_group()


def launch_training(args):
    if args.no_ddp:
        args.num_workers = 0  # Ensure DataLoader runs in main process for pdb
        train(0, 1, args)
    else:
        if "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            train(rank, world_size, args)
        else:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/places365.lance")
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--num_classes", type=int, default=365)
    parser.add_argument("--sampler_type", type=str, default="sharded_batch")
    parser.add_argument("--use_safe_loader", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=1, help="Number of processes/devices")
    parser.add_argument("--no_ddp", action="store_true", help="Run in non-distributed (debug) mode")

    args = parser.parse_args()

    launch_training(args)
