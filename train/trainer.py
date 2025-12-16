"""
Code partly inspired from : https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_msg_cls.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import get_config


class TreePointCloudDataset(Dataset):
    def __init__(self, root_dir: Path, split: str):
        self.root_dir = Path(root_dir)
        self.samples_dir = self.root_dir / "samples"
        split_file = self.root_dir / f"{split}.txt"
        ids = split_file.read_text().splitlines()
        self.sample_ids = [s.strip() for s in ids if s.strip()]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        data = np.load(self.samples_dir / f"{sid}.npz")
        points = data["points"].astype(np.float32)
        label = int(data["label"])
        pts = torch.from_numpy(points)
        y = torch.tensor(label, dtype=torch.long)
        return pts, y


class PointNet2Classifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.sa1_conv1 = nn.Conv1d(in_channels, 64, 1)
        self.sa1_bn1 = nn.BatchNorm1d(64)
        self.sa1_conv2 = nn.Conv1d(64, 128, 1)
        self.sa1_bn2 = nn.BatchNorm1d(128)

        self.sa2_conv1 = nn.Conv1d(128, 128, 1)
        self.sa2_bn1 = nn.BatchNorm1d(128)
        self.sa2_conv2 = nn.Conv1d(128, 256, 1)
        self.sa2_bn2 = nn.BatchNorm1d(256)

        self.glob_conv = nn.Conv1d(256, 512, 1)
        self.glob_bn = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.sa1_bn1(self.sa1_conv1(x)))
        x = F.relu(self.sa1_bn2(self.sa1_conv2(x)))
        x = F.relu(self.sa2_bn1(self.sa2_conv1(x)))
        x = F.relu(self.sa2_bn2(self.sa2_conv2(x)))
        x = F.relu(self.glob_bn(self.glob_conv(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(points)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * points.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += points.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        logits = model(points)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * points.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += points.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    cfg = get_config(args.config)

    root_processed = cfg.dataset.processed_dir
    train_ds = TreePointCloudDataset(root_processed, "train")
    val_ds = TreePointCloudDataset(root_processed, "val")

    in_channels = train_ds[0][0].shape[1]
    num_classes = cfg.dataset.num_classes

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    model = PointNet2Classifier(in_channels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs,
    )

    best_val_acc = 0.0
    checkpoint_dir = cfg.training.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.training.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{cfg.training.num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = checkpoint_dir / "pointnet2_best.pth"
            torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()

