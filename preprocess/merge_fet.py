"""
Insert vision features into existing MOSI-style pkl dataset.
"""

import pickle
import numpy as np

PKL_PATH = "mosi_features.pkl"
VISION_CACHE = "preprocess/mosi/vision_cache"
MAX_LEN = 300

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)


def align_clip(vision_feat, timestamps, clip_len):
    T = min(len(vision_feat), MAX_LEN)
    feat = vision_feat[:T]
    mask_len = T
    if T < MAX_LEN:
        feat = np.pad(feat, ((0, MAX_LEN - T), (0, 0)))
    return feat, mask_len


for split in ["train", "valid", "test"]:
    data[split]["vision"] = []
    data[split]["vision_lengths"] = []

    for vid_clip in data[split]["id"]:
        video_id = vid_clip.split("$_$")[0]
        cache = np.load(f"{VISION_CACHE}/{video_id}.npz")
        feat, length = align_clip(
            cache["features"],
            cache["timestamps"],
            None
        )
        data[split]["vision"].append(feat)
        data[split]["vision_lengths"].append(length)


with open("mosi_with_vision.pkl", "wb") as f:
    pickle.dump(data, f)
