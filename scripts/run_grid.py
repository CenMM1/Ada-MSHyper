import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime


def build_cmd(base_args, overrides):
    cmd = ["/opt/miniforge3/bin/conda", "run", "-n", "py385", "python", "run.py"]
    for k, v in base_args.items():
        cmd.extend([f"--{k}", str(v)])
    for k, v in overrides.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def parse_metrics(log_text):
    mae = None
    corr = None
    acc7 = None
    for line in log_text.splitlines():
        line = line.strip()
        if line.startswith("MAE:"):
            mae = float(line.split("MAE:")[1].strip())
        if line.startswith("Corr:"):
            corr = float(line.split("Corr:")[1].strip())
        if line.startswith("ACC7"):
            acc7 = float(line.split(":")[1].strip())
    return mae, corr, acc7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./grid_results")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"grid_{timestamp}.csv")
    md_path = os.path.join(args.out_dir, f"grid_{timestamp}.md")

    base_args = {
        "is_training": 1,
        "root_path": "./datasets",
        "data_format": "pkl",
        "pkl_path": "./preprocess/mosi_all_feat.pkl",
        "label_map": 7,
        "exclude_oob": 1,
        "num_workers": 8,
        "num_classes": 7,
        "task_mode": "ordinal",
        "d_model": 128,
        "dropout": 0.1,
        "k": 3,
        "hyper_heads": 1,
        "hyper_multi_head_attention": 0,
        "seq_len_text": 44,
        "seq_len_audio": 500,
        "seq_len_video": 32,
        "feature_dim_text": 768,
        "feature_dim_audio": 768,
        "feature_dim_video": 512,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "train_epochs": 10,
        "patience": 3,
        "itr": 1,
        "kappa": 0.0001,
    }

    lrs = [3e-3]
    hyper_grid = [
        {"hyper_num_text": 20, "hyper_num_audio": 30, "hyper_num_video": 10},
    ]
    loss_modes = ["coral"]
    reg_loss_weights = [0.05, 0.1]
    reg_loss_types = ["mae", "huber"]
    reg_huber_deltas = [1.0]
    combo_configs = [
        {"combo": "baseline", "use_mosi_ecr": 0},
        {"combo": "ecr_warmup_only", "use_mosi_ecr": 1},
    ]

    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "loss_mode",
            "combo",
            "lr",
            "hyper_num_text",
            "hyper_num_audio",
            "hyper_num_video",
            "reg_loss_weight",
            "reg_loss_type",
            "reg_huber_delta",
            "mae",
            "corr",
            "acc7",
            "status",
        ])

        run_id = 0
        for loss_mode in loss_modes:
            for combo in combo_configs:
                for lr in lrs:
                    for hyper_cfg in hyper_grid:
                        for reg_loss_weight in reg_loss_weights:
                            for reg_loss_type in reg_loss_types:
                                for reg_huber_delta in reg_huber_deltas:
                                    run_id += 1
                                    overrides = {
                                        "loss_mode": loss_mode,
                                        "learning_rate": lr,
                                        **hyper_cfg,
                                        "use_mosi_ecr": combo["use_mosi_ecr"],
                                        "reg_loss_weight": reg_loss_weight,
                                        "reg_loss_type": reg_loss_type,
                                        "reg_huber_delta": reg_huber_delta,
                                    }
                                    overrides["setting_suffix"] = f"run{run_id}"  # unique checkpoint per run
                                    if loss_mode == "coral":
                                        overrides["use_coral"] = 1
                                        overrides["loss_mode"] = "coral"
                                    else:
                                        overrides["use_coral"] = 0
                                    cmd = build_cmd(base_args, overrides)
                                    status = "ok"
                                    mae = corr = acc7 = None

                                    if args.dry_run:
                                        print(" ".join(cmd))
                                        status = "dry_run"
                                    else:
                                        print(f"[Run {run_id}] {json.dumps(overrides)}")
                                        proc = subprocess.run(cmd, capture_output=True, text=True)
                                        log_text = proc.stdout + "\n" + proc.stderr
                                        mae, corr, acc7 = parse_metrics(log_text)
                                        if proc.returncode != 0:
                                            status = f"fail({proc.returncode})"

                                        log_path = os.path.join(args.out_dir, f"run_{run_id}.log")
                                        with open(log_path, "w") as lf:
                                            lf.write(log_text)

                                    writer.writerow([
                                        run_id,
                                        loss_mode,
                                        combo["combo"],
                                        lr,
                                        hyper_cfg["hyper_num_text"],
                                        hyper_cfg["hyper_num_audio"],
                                        hyper_cfg["hyper_num_video"],
                                        reg_loss_weight,
                                        reg_loss_type,
                                        reg_huber_delta,
                                        mae,
                                        corr,
                                        acc7,
                                        status,
                                    ])
                                    rows.append([
                                        run_id,
                                        loss_mode,
                                        combo["combo"],
                                        lr,
                                        hyper_cfg["hyper_num_text"],
                                        hyper_cfg["hyper_num_audio"],
                                        hyper_cfg["hyper_num_video"],
                                        reg_loss_weight,
                                        reg_loss_type,
                                        reg_huber_delta,
                                        mae,
                                        corr,
                                        acc7,
                                        status,
                                    ])

    rows_sorted = sorted(rows, key=lambda r: (-(r[11] or -1e9)))
    with open(md_path, "w") as mf:
        mf.write("| run_id | loss_mode | combo | lr | hyper_num_text | hyper_num_audio | hyper_num_video | reg_loss_weight | reg_loss_type | reg_huber_delta | MAE | Corr | ACC7 | status |\n")
        mf.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for row in rows_sorted:
            mf.write("| " + " | ".join(str(x) for x in row) + " |\n")

    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] MD:  {md_path}")


if __name__ == "__main__":
    main()
