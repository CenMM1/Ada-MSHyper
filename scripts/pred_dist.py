import argparse
import ast
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from models.ASHyper import MultimodalClassifier


def parse_namespace_from_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Args: Namespace("):
                content = line.strip().replace("Args: Namespace(", "").rstrip(")")
                pairs = {}
                for match in re.finditer(r"(\w+)=([^,)]*)", content):
                    key = match.group(1)
                    raw = match.group(2).strip()
                    try:
                        val = ast.literal_eval(raw)
                    except Exception:
                        if raw in {"True", "False"}:
                            val = raw == "True"
                        else:
                            val = raw
                    pairs[key] = val
                return pairs
    raise ValueError(f"No Args Namespace found in {log_path}")


def build_model(args, device):
    model = MultimodalClassifier(args).float().to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_log", type=str, required=True, help="path to run_X.log")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint.pth")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_png", type=str, default="./grid_results/pred_dist_run.png")
    parser.add_argument("--out_csv", type=str, default="", help="optional histogram csv output")
    parser.add_argument("--out_pairs_csv", type=str, default="", help="optional csv with per-sample y_pred/y_true")
    parser.add_argument("--bins", type=int, default=61, help="histogram bins for y_pred")
    args_cli = parser.parse_args()

    args_dict = parse_namespace_from_log(args_cli.run_log)
    args = argparse.Namespace(**args_dict)
    args.use_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if args.use_gpu else "cpu")
    model = build_model(args, device)

    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    state = torch.load(args_cli.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _, loader = data_provider(args, args_cli.split)

    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].float().to(device)
            outputs, _ = model(batch)
            if getattr(args, "task_mode", "classification") == "ordinal":
                pred = torch.sigmoid(outputs).sum(dim=1) - 3.0
            elif getattr(args, "task_mode", "classification") == "regression":
                pred = outputs.squeeze(1) if outputs.dim() == 2 else outputs.view(-1)
            else:
                pred = outputs.argmax(dim=1).float()
            preds.append(pred.detach().cpu().numpy())
            if "label_reg" in batch:
                true = batch["label_reg"].view(-1)
                trues.append(true.detach().cpu().numpy())
            elif "label_ordinal" in batch:
                true = batch["label_ordinal"].sum(dim=1) - 3.0
                trues.append(true.detach().cpu().numpy())
            elif "label_multimodal" in batch:
                true = batch["label_multimodal"].view(-1)
                trues.append(true.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    hist_counts, bin_edges = np.histogram(y_pred, bins=args_cli.bins, range=(-3, 3))
    y_true = np.concatenate(trues, axis=0) if trues else None
    true_hist_counts = None
    if y_true is not None:
        true_hist_counts, _ = np.histogram(y_true, bins=bin_edges)

    out_dir = os.path.dirname(args_cli.out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args_cli.out_csv:
        csv_dir = os.path.dirname(args_cli.out_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(args_cli.out_csv, "w", encoding="utf-8") as f:
            if true_hist_counts is None:
                f.write("bin_left,bin_right,y_pred_count\n")
                for i in range(len(hist_counts)):
                    f.write(f"{bin_edges[i]:.6f},{bin_edges[i+1]:.6f},{hist_counts[i]}\n")
            else:
                f.write("bin_left,bin_right,y_pred_count,y_true_count\n")
                for i in range(len(hist_counts)):
                    f.write(
                        f"{bin_edges[i]:.6f},{bin_edges[i+1]:.6f},{hist_counts[i]},{true_hist_counts[i]}\n"
                    )

    if args_cli.out_pairs_csv:
        pairs_path = args_cli.out_pairs_csv
        with open(pairs_path, "w", encoding="utf-8") as f:
            f.write("index,y_pred,y_true\n")
            if y_true is None:
                for i, v in enumerate(y_pred):
                    f.write(f"{i},{v:.6f},\n")
            else:
                for i, (p, t) in enumerate(zip(y_pred, y_true)):
                    f.write(f"{i},{p:.6f},{t:.6f}\n")

    stats = {
        "count": int(y_pred.shape[0]),
        "mean": float(np.mean(y_pred)),
        "std": float(np.std(y_pred)),
        "min": float(np.min(y_pred)),
        "max": float(np.max(y_pred)),
        "p10": float(np.percentile(y_pred, 10)),
        "p50": float(np.percentile(y_pred, 50)),
        "p90": float(np.percentile(y_pred, 90)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if y_true is not None:
        axes[0].scatter(y_true, y_pred, s=8, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle="--", color="gray", linewidth=1)
        axes[0].set_xlabel("y_true")
        axes[0].set_ylabel("y_pred")
        axes[0].set_title("Pred vs True (Scatter)")
    else:
        axes[0].set_title("Pred vs True (Scatter) - y_true not available")

    axes[1].hist(y_pred, bins=bin_edges, alpha=0.6, label="y_pred")
    if y_true is not None:
        axes[1].hist(y_true, bins=bin_edges, alpha=0.6, label="y_true")
    axes[1].set_title("Distribution (Histogram)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(args_cli.out_png, dpi=150)

    print(f"Saved plot to {args_cli.out_png}")
    if args_cli.out_csv:
        print(f"Saved histogram to {args_cli.out_csv}")
    if args_cli.out_pairs_csv:
        print(f"Saved pairs to {pairs_path}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
