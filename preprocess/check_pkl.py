"""
Sanity check for compact MOSI-style pkl dataset with vision features.
"""

import pickle
import numpy as np
from collections import Counter

PKL_PATH = "mosi_compact_with_vision.pkl"
PREVIEW_N = 1


def summarize_array(arr, name, max_print=3):
    """
    Summarize a list/array field without flooding stdout.
    """
    if len(arr) == 0:
        return f"{name}: EMPTY"

    first = arr[0]

    if isinstance(first, np.ndarray):
        shapes = Counter([x.shape for x in arr])
        dtypes = Counter([x.dtype for x in arr])
        return (
            f"{name}:\n"
            f"  type: ndarray\n"
            f"  unique shapes: {dict(shapes)}\n"
            f"  dtypes: {dict(dtypes)}"
        )

    else:
        return (
            f"{name}:\n"
            f"  type: {type(first)}\n"
            f"  example: {first}"
        )


def check_split(split_name, split_data):
    print("=" * 80)
    print(f"SPLIT: {split_name}")
    print("=" * 80)

    # -------------------------
    # Basic info
    # -------------------------
    n_samples = len(split_data["id"])
    print(f"Number of samples: {n_samples}\n")

    # -------------------------
    # Field-level summary
    # -------------------------
    for key, value in split_data.items():
        if isinstance(value, list):
            print(summarize_array(value, key))
        else:
            print(f"{key}: type={type(value)}")
        print()

    # -------------------------
    # Preview first N samples
    # -------------------------
    print("-" * 80)
    print(f"Preview first {PREVIEW_N} samples")
    print("-" * 80)

    for i in range(min(PREVIEW_N, n_samples)):
        print(f"[{i}] id = {split_data['id'][i]}")

        if "text" in split_data:
            t = split_data["text"][i]
            print(f"  text: shape={t.shape}")

        if "audio" in split_data:
            a = split_data["audio"][i]
            al = split_data["audio_lengths"][i]
            print(f"  audio: shape={a.shape}, length={al}")

        if "vision" in split_data:
            v = split_data["vision"][i]
            vl = split_data["vision_lengths"][i]
            print(f"  vision: shape={v.shape}, length={vl}")

        if "regression_labels" in split_data:
            y = split_data["regression_labels"][i]
            print(f"  label (regression): {y}")

        print()


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    print("#" * 80)
    print(f"Loaded dataset: {PKL_PATH}")
    print(f"Splits: {list(data.keys())}")
    print("#" * 80)
    print()

    for split in ["train", "valid", "test"]:
        if split in data:
            check_split(split, data[split])
        else:
            print(f"WARNING: split '{split}' not found!")


if __name__ == "__main__":
    main()
