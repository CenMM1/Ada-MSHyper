"""
Clean and compress a MOSI-style pkl dataset:
- Keep only model-required fields
- Force audio and text features to float16
- Preserve id / lengths / labels
"""

import torch
import numpy as np
import io
import pickle



# =========================
# Paths
# =========================

DST_PKL = "mosei_s0_clean.pkl.gz"    # 输出
SRC_PKL = "mosei_s0.pkl"   # 输入（pickle.dump 保存的）
USE_GZIP = True
# =========================
# Keys to keep
# =========================


BASE_KEEP_KEYS = {
    "id",
    "text",
    "audio",
    "audio_lengths",
    "regression_labels",
}

OPTIONAL_KEYS = {
    "vision",
    "vision_lengths",
}

def force_fp16(x):
    """
    递归强制把可转换的浮点数据转成 float16。
    支持 numpy 数组、list/tuple、dict 的嵌套结构。
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.half()
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float16, copy=False)
    if isinstance(x, list):
        return [force_fp16(v) for v in x]
    if isinstance(x, tuple):
        return tuple(force_fp16(v) for v in x)
    if isinstance(x, dict):
        return {k: force_fp16(v) for k, v in x.items()}
    return x

def process_list(values, key):
    if key in {"audio", "text", "vision"}:
        return force_fp16(values)
    return values

# =========================
# 🔥 正确的 legacy 加载方式
# =========================

with open(SRC_PKL, "rb") as f:
    src_data = pickle.load(f)

# =========================
# 清理数据
# =========================

dst_data = {}

for split in ["train", "valid", "test"]:
    if split not in src_data:
        continue

    dst_data[split] = {}
    split_data = src_data[split]

    keep_keys = BASE_KEEP_KEYS.copy()
    for k in OPTIONAL_KEYS:
        if k in split_data:
            keep_keys.add(k)

    for key in keep_keys:
        values = split_data[key]
        if isinstance(values, list):
            dst_data[split][key] = process_list(values, key)
        else:
            dst_data[split][key] = values

# =========================
# 保存为干净的新格式
# =========================

if USE_GZIP:
    import gzip
    with gzip.open(DST_PKL, "wb") as f:
        pickle.dump(dst_data, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(DST_PKL, "wb") as f:
        pickle.dump(dst_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✅ Cleaned fp16 dataset saved to: {DST_PKL}")
