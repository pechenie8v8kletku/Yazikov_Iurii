from datasets import load_dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16,ViT_B_16_Weights
from torchvision import transforms
import torch.nn.functional as F

def landmark_error(pred, target):
    pred = pred.view(-1, 68, 2)
    target = target.view(-1, 68, 2)

    return torch.norm(pred - target, dim=2).mean()
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_epoch(model, loader, optimizer, criterion, device, epoch, writer):
    model.train()
    total_loss = 0
    total = 0
    total_metric=0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        x, y = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        batch_metric = landmark_error(logits, y).item()
        total_metric+=batch_metric
        batch_loss = criterion(logits, y)

        batch_loss.backward()
        optimizer.step()

        current_loss_val = batch_loss.item()
        total_loss += current_loss_val
        total+=y.size(0)
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'batch_loss': f"{current_loss_val:.4f}",
            'avg_loss': f"{avg_loss:.4f}"
        })
    avg_loss = total_loss / len(loader)
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    writer.add_scalar("Metric/Train", total_metric / len(loader), epoch)


    return avg_loss,total_metric / len(loader)

@torch.no_grad()
def validate_epoch(model, loader, criterion, device, epoch, writer):
    model.eval()

    total_loss = 0
    total_samples = 0
    total_metric=0

    pbar = tqdm(loader, desc=f"Validation Epoch {epoch}", leave=False)

    for batch in pbar:
        x = batch["image"]
        y = batch["label"]
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_i = criterion(logits, y)
        batch_metric = landmark_error(logits, y).item()
        total_metric += batch_metric

        total_loss += loss_i.item()
        total_samples += y.size(0)
        pbar.set_postfix(loss=f"{loss_i.item():.4f}")
    avg_val_loss = total_loss / len(loader)
    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    writer.add_scalar("Metric/Val", total_metric / len(loader), epoch)
    print(f"\n[Validation Epoch {epoch}] Avg Loss: {avg_val_loss:.4f}")

    return avg_val_loss,total_metric / len(loader)

def save_epoch_checkpoint(model, optimizer, epoch, path):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, file_path)

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

def save_only_model(model, epoch, path):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
    }, file_path)
