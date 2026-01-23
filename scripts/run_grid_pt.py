import argparse
import csv
import json
import os
import subprocess
from datetime import datetime


def build_cmd(base_args, overrides):
    cmd = ["/opt/miniforge3/bin/conda", "run", "-n", "py385", "python", "run.py"]
    for k, v in base_args.items():
        cmd.extend([f"--{k}", str(v)])
    for k, v in overrides.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def parse_metrics(log_text):
    accuracy = None
    f1_macro = None
    f1_weighted = None
    for line in log_text.splitlines():
        line = line.strip()
        if line.startswith("Accuracy:"):
            accuracy = float(line.split("Accuracy:")[1].strip())
        if line.startswith("F1 Score (Macro):"):
            f1_macro = float(line.split("F1 Score (Macro):")[1].strip())
        if line.startswith("F1 Score (Weighted):"):
            f1_weighted = float(line.split("F1 Score (Weighted):")[1].strip())
    return accuracy, f1_macro, f1_weighted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./grid_results")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"grid_pt_{timestamp}.csv")
    md_path = os.path.join(args.out_dir, f"grid_pt_{timestamp}.md")

    base_args = {
        "is_training": 1,
        "root_path": "./preprocess/processed_data",
        "data_format": "pt",
        "num_workers": 8,
        "num_classes": 7,
        "task_mode": "classification",
        "d_model": 128,
        "dropout": 0.1,
        "k": 3,
        "dynamic_hypergraph": 1,
        "hyper_update_freq": 5,
        "hyper_heads": 1,
        "hyper_multi_head_attention": 0,
        "hyper_num_text": 50,
        "hyper_num_audio": 20,
        "hyper_num_video": 10,
        "seq_len_text": 160,
        "seq_len_audio": 518,
        "seq_len_video": 16,
        "feature_dim_text": 1024,
        "feature_dim_audio": 1024,
        "feature_dim_video": 2048,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "train_epochs": 10,
        "patience": 3,
        "itr": 1,
        "kappa": 0.1,
        "weight_decay": 1e-5,
        "hyper_topk": 3,
        "hyper_tau": 0.5,
    }

    sweep_plan = [
        ("hyperedge_num", [2, 3, 4, 6], "k"),
        ("sparsity_topk", [2, 3, 4], "hyper_topk"),
        ("temperature_tau", [0.3, 0.5, 0.7, 1.0], "hyper_tau"),
        ("ecr_kappa", [0.0, 0.05, 0.1, 0.2], "kappa"),
    ]

    dataset_tag = "pt"
    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "sweep",
            "k",
            "hyper_topk",
            "hyper_tau",
            "kappa",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "status",
        ])

        run_id = 0
        for sweep_name, values, key in sweep_plan:
            for val in values:
                run_id += 1
                overrides = {
                    key: val,
                }
                overrides["setting_suffix"] = f"{dataset_tag}_run{run_id}"  # unique checkpoint per run
                cmd = build_cmd(base_args, overrides)
                status = "ok"
                accuracy = f1_macro = f1_weighted = None

                if args.dry_run:
                    print(" ".join(cmd))
                    status = "dry_run"
                else:
                    print(f"[Run {run_id}] {json.dumps({**base_args, **overrides})}")
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    log_text = proc.stdout + "\n" + proc.stderr
                    accuracy, f1_macro, f1_weighted = parse_metrics(log_text)
                    if proc.returncode != 0:
                        status = f"fail({proc.returncode})"

                    log_path = os.path.join(args.out_dir, f"run_pt_{run_id}.log")
                    with open(log_path, "w") as lf:
                        lf.write(log_text)

                writer.writerow([
                    run_id,
                    sweep_name,
                    overrides.get("k", base_args["k"]),
                    overrides.get("hyper_topk", base_args["hyper_topk"]),
                    overrides.get("hyper_tau", base_args["hyper_tau"]),
                    overrides.get("kappa", base_args["kappa"]),
                    accuracy,
                    f1_macro,
                    f1_weighted,
                    status,
                ])
                rows.append([
                    run_id,
                    sweep_name,
                    overrides.get("k", base_args["k"]),
                    overrides.get("hyper_topk", base_args["hyper_topk"]),
                    overrides.get("hyper_tau", base_args["hyper_tau"]),
                    overrides.get("kappa", base_args["kappa"]),
                    accuracy,
                    f1_macro,
                    f1_weighted,
                    status,
                ])

    rows_sorted = sorted(rows, key=lambda r: (r[1], r[0]))
    with open(md_path, "w") as mf:
        mf.write("| run_id | sweep | k | hyper_topk | hyper_tau | kappa | Accuracy | F1 Macro | F1 Weighted | status |\n")
        mf.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for row in rows_sorted:
            mf.write("| " + " | ".join(str(x) for x in row) + " |\n")

    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] MD:  {md_path}")


if __name__ == "__main__":
    main()
