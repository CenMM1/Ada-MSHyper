"""
Sequence-level sanity check for vision features (MOSI).
Focus:
1) Self-similarity matrix (can we build hyperedges?)
2) Adjacent-frame cosine similarity & drift vs |label|
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

# =========================
# Config
# =========================

VISION_DIR = "mosi/vision_cache"
LABEL_CSV = "mosi/label.csv"
OUT_DIR = "vision_seq_eval"

K = 32

NUM_CLIPS_SIM_MATRIX = 16     # number of clips for heatmaps
NUM_CLIPS_STATS = 300         # number of clips for statistics

RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "self_similarity"), exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# Load labels
# =========================

labels_df = pd.read_csv(LABEL_CSV)
label_map = {}

for _, row in labels_df.iterrows():
    key = f"{row['video_id']}_{row['clip_id']}"
    label_map[key] = float(row["label"])

# =========================
# Load available clips
# =========================

all_npz = [
    f for f in os.listdir(VISION_DIR)
    if f.endswith(".npz") and f.replace(".npz", "") in label_map
]

print(f"[INFO] Found {len(all_npz)} matched vision clips")

# =========================
# Helper functions
# =========================

def normalize_feats(feats):
    norm = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    return feats / norm


def adjacent_cosine_and_drift(feats):
    """
    feats: [K, D]
    return:
      cosine_sim: [K-1]
      drift: [K-1] = 1 - cosine
    """
    feats = normalize_feats(feats)
    sims = np.sum(feats[:-1] * feats[1:], axis=1)
    drift = 1.0 - sims
    return sims, drift


# =========================
# 1) Self-similarity matrix
# =========================

print("[INFO] Generating self-similarity matrices...")

sample_sim = random.sample(
    all_npz, min(NUM_CLIPS_SIM_MATRIX, len(all_npz))
)

for i, fname in enumerate(sample_sim):
    key = fname.replace(".npz", "")
    feats = np.load(os.path.join(VISION_DIR, fname))["features"]  # [K,512]
    feats = normalize_feats(feats)

    sim = cosine_similarity(feats, feats)  # [K,K]

    plt.figure(figsize=(4, 4))
    plt.imshow(sim, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(label="cosine similarity")
    plt.title(f"Self-similarity\n{key}")
    plt.xlabel("Frame index")
    plt.ylabel("Frame index")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "self_similarity", f"sim_{i:02d}.png")
    )
    plt.close()

# =========================
# 2) Adjacent cosine & drift statistics
# =========================

print("[INFO] Computing adjacent-frame statistics...")

sample_stats = random.sample(
    all_npz, min(NUM_CLIPS_STATS, len(all_npz))
)

all_adj_sims = []
mean_drift = []
max_drift = []
std_drift = []
abs_labels = []

for fname in sample_stats:
    key = fname.replace(".npz", "")
    feats = np.load(os.path.join(VISION_DIR, fname))["features"]

    sims, drift = adjacent_cosine_and_drift(feats)

    all_adj_sims.extend(sims.tolist())
    mean_drift.append(np.mean(drift))
    max_drift.append(np.max(drift))
    std_drift.append(np.std(drift))
    abs_labels.append(abs(label_map[key]))

all_adj_sims = np.array(all_adj_sims)
mean_drift = np.array(mean_drift)
max_drift = np.array(max_drift)
std_drift = np.array(std_drift)
abs_labels = np.array(abs_labels)

# =========================
# Plot 1: adjacent cosine histogram
# =========================

plt.figure(figsize=(6, 5))
plt.hist(all_adj_sims, bins=50, alpha=0.8)
plt.xlabel("Cosine similarity (adjacent frames)")
plt.ylabel("Count")
plt.title("Adjacent-frame cosine similarity distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "adjacent_cosine_hist.png"))
plt.close()

# =========================
# Plot 2: drift vs |label|
# =========================

plt.figure(figsize=(6, 5))
plt.scatter(abs_labels, mean_drift, alpha=0.6, label="mean drift")
plt.scatter(abs_labels, max_drift, alpha=0.6, label="max drift")
plt.xlabel("|Valence label|")
plt.ylabel("Drift (1 - cosine)")
plt.title("Drift vs emotion intensity")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "drift_vs_abslabel.png"))
plt.close()

# =========================
# Correlation statistics
# =========================

rho_mean, p_mean = spearmanr(abs_labels, mean_drift)
rho_max, p_max = spearmanr(abs_labels, max_drift)
rho_std, p_std = spearmanr(abs_labels, std_drift)

with open(os.path.join(OUT_DIR, "drift_stats.txt"), "w") as f:
    f.write("Spearman correlation with |label|\n")
    f.write(f"mean drift: rho={rho_mean:.3f}, p={p_mean:.3g}\n")
    f.write(f"max  drift: rho={rho_max:.3f}, p={p_max:.3g}\n")
    f.write(f"std  drift: rho={rho_std:.3f}, p={p_std:.3g}\n")

print("[RESULT] Drift vs |label| (Spearman)")
print(f"mean drift: rho={rho_mean:.3f}, p={p_mean:.3g}")
print(f"max  drift: rho={rho_max:.3f}, p={p_max:.3g}")
print(f"std  drift: rho={rho_std:.3f}, p={p_std:.3g}")

print(f"[DONE] Outputs saved to: {OUT_DIR}")
