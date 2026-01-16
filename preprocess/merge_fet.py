"""
Build a compact MOSI pkl with utterance-level vision features.
Only keep model-required fields to reduce file size.
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
# Helper
# =========================

def load_vision(vid_clip):
    """
    vid_clip: video_id$_$clip_id
    returns: features [K, 512], length K
    """
    video_id, clip_id = vid_clip.split("$_$")
    path = os.path.join(VISION_CACHE, f"{video_id}_{clip_id}.npz")

    cache = np.load(path)
    feat = cache["features"]
    return feat, feat.shape[0]


# =========================
# Build new dataset
# =========================

for split in ["train", "valid", "test"]:
    dst_data[split] = {}

    # ---- copy required fields ----
    for key in KEEP_KEYS:
        dst_data[split][key] = src_data[split][key]

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
    pickle.dump(dst_data, f)

print(f"Saved compact dataset to {DST_PKL}")
