"""
Build a compact MOSI pkl with utterance-level vision features.
Apply dtype compression to reduce disk size and speed up loading.
"""

import os
import pickle
import numpy as np

# =========================
# Paths
# =========================

SRC_PKL = "mosi.pkl"
VISION_CACHE = "mosi/vision_cache"
DST_PKL = "mosi_compact_with_vision.pkl"

# =========================
# Keys to keep
# =========================

KEEP_KEYS = {
    "id",
    "text",
    "audio",
    "audio_lengths",
    "regression_labels"
}

# =========================
# Load original data
# =========================

with open(SRC_PKL, "rb") as f:
    src_data = pickle.load(f)

dst_data = {}

# =========================
# Helper functions
# =========================

def load_vision(vid_clip):
    """
    vid_clip: video_id$_$clip_id
    returns: features [K, 512] (float16), length K
    """
    video_id, clip_id = vid_clip.split("$_$")
    path = os.path.join(VISION_CACHE, f"{video_id}_{clip_id}.npz")

    cache = np.load(path)
    feat = cache["features"].astype(np.float16)  # ✅ vision → fp16
    return feat, feat.shape[0]


def force_fp16(x):
    """
    Force numeric numpy arrays to float16.
    """
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float16)
    return x


# =========================
# Build new dataset
# =========================

for split in ["train", "valid", "test"]:
    dst_data[split] = {}

    # ---- copy required fields ----
    for key in KEEP_KEYS:
        values = src_data[split][key]

        if key == "audio":
            # 🔥 强制 audio → fp16
            dst_data[split][key] = [force_fp16(v) for v in values]

        elif isinstance(values, list):
            dst_data[split][key] = values  # text 不动

        else:
            dst_data[split][key] = values
    # ---- inject vision ----
    visions = []
    vision_lengths = []

    for vid_clip in src_data[split]["id"]:
        feat, length = load_vision(vid_clip)
        visions.append(feat)
        vision_lengths.append(length)

    dst_data[split]["vision"] = visions
    dst_data[split]["vision_lengths"] = vision_lengths


# =========================
# Save
# =========================

with open(DST_PKL, "wb") as f:
    pickle.dump(dst_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved compact fp16 dataset to {DST_PKL}")
