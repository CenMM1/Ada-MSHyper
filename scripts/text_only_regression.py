import argparse
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_provider.data_loader import Dataset_PKL_Multimodal_Classification
from utils.metrics import MAE, CORR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextOnlyRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_seq: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        # text_seq: [B, T, D], text_mask: [B, T]
        mask = text_mask.unsqueeze(-1)
        masked = text_seq * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = masked.sum(dim=1) / denom
        return self.net(pooled).squeeze(-1)


def build_loader(args, split: str) -> DataLoader:
    dataset = Dataset_PKL_Multimodal_Classification(
        root_path=args.root_path,
        flag=split,
        args=args,
    )
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds = []
    all_true = []
    for batch in loader:
        text = batch["text_vector"].to(device)
        text_mask = batch["text_mask"].to(device)
        targets = batch["label_reg"].to(device)
        preds = model(text, text_mask)
        all_preds.append(preds.detach().cpu().numpy())
        all_true.append(targets.detach().cpu().numpy())
    pred_np = np.concatenate(all_preds, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    mae = float(MAE(pred_np, true_np))
    corr = float(CORR(pred_np, true_np))
    return mae, corr


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader = build_loader(args, "train")
    val_loader = build_loader(args, "val")
    test_loader = build_loader(args, "test")

    sample_batch = next(iter(train_loader))
    text_dim = sample_batch["text_vector"].shape[-1]

    model = TextOnlyRegressor(text_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=args.huber_delta)

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            text = batch["text_vector"].to(device)
            text_mask = batch["text_mask"].to(device)
            targets = batch["label_reg"].to(device)
            preds = model(text, text_mask)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mae, val_corr = evaluate(model, val_loader, device)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d} | Vali MAE: {val_mae:.4f} | Corr: {val_corr:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mae, test_corr = evaluate(model, test_loader, device)
    print(f"MAE: {test_mae:.4f}")
    print(f"Corr: {test_corr:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text-only regression on MOSI pkl")
    parser.add_argument("--root_path", type=str, default="./datasets")
    parser.add_argument("--pkl_path", type=str, default="./preprocess/mosi_all_feat.pkl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--data_format", type=str, default="pkl")
    parser.add_argument("--label_map", type=int, default=7)
    parser.add_argument("--exclude_oob", type=int, default=1)
    parser.add_argument("--task_mode", type=str, default="regression")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"pkl_path not found: {args.pkl_path}")
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
