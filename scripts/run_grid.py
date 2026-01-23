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
        "task_mode": "regression",
        "d_model": 128,
        "dropout": 0.1,
        "k": 3,
        "dynamic_hypergraph": 1,
        "hyper_update_freq": 5,
        "hyper_heads": 1,
        "hyper_multi_head_attention": 0,
        "hyper_topk": 3,
        "hyper_tau": 0.5,
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
        "kappa": 0.0,
        "weight_decay": 1e-5,
        "attn_pooling_dropout": 0.1,
    }

    lrs = [3e-3,1e-3,3e-4]
    hyper_grid = [
        {"hyper_num_text": 20, "hyper_num_audio": 30, "hyper_num_video": 10},
        {"hyper_num_text": 30, "hyper_num_audio": 50, "hyper_num_video": 20},
    ]
    use_head_lns = [0]
    use_modal_gates = [1]
    use_attn_poolings = [0, 1]

    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "lr",
            "use_head_ln",
            "use_modal_gate",
            "use_attn_pooling",
            "attn_pooling_dropout",
            "hyper_num_text",
            "hyper_num_audio",
            "hyper_num_video",
            "mae",
            "corr",
            "acc7",
            "status",
        ])

        run_id = 0
        for lr in lrs:
            for hyper_cfg in hyper_grid:
                for use_head_ln in use_head_lns:
                    for use_modal_gate in use_modal_gates:
                        for use_attn_pooling in use_attn_poolings:
                            run_id += 1
                            overrides = {
                                "learning_rate": lr,
                                **hyper_cfg,
                                "use_head_ln": use_head_ln,
                                "use_modal_gate": use_modal_gate,
                                "use_attn_pooling": use_attn_pooling,
                            }
                            overrides["setting_suffix"] = f"run{run_id}"  # unique checkpoint per run
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
                                lr,
                                use_head_ln,
                                use_modal_gate,
                                use_attn_pooling,
                                base_args["attn_pooling_dropout"],
                                hyper_cfg["hyper_num_text"],
                                hyper_cfg["hyper_num_audio"],
                                hyper_cfg["hyper_num_video"],
                                mae,
                                corr,
                                acc7,
                                status,
                            ])
                            rows.append([
                                run_id,
                                lr,
                                use_head_ln,
                                use_modal_gate,
                                use_attn_pooling,
                                base_args["attn_pooling_dropout"],
                                hyper_cfg["hyper_num_text"],
                                hyper_cfg["hyper_num_audio"],
                                hyper_cfg["hyper_num_video"],
                                mae,
                                corr,
                                acc7,
                                status,
                            ])

    rows_sorted = sorted(rows, key=lambda r: (-(r[8] or -1e9)))
    with open(md_path, "w") as mf:
        mf.write("| run_id | lr | use_head_ln | use_modal_gate | use_attn_pooling | attn_pooling_dropout | hyper_num_text | hyper_num_audio | hyper_num_video | MAE | Corr | ACC7 | status |\n")
        mf.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for row in rows_sorted:
            mf.write("| " + " | ".join(str(x) for x in row) + " |\n")

    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] MD:  {md_path}")


if __name__ == "__main__":
    main()
