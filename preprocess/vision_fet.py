"""
Fast Fixed-K facial feature extraction for MOSI.
Optimizations:
  1) Run MTCNN only once per clip
  2) Reuse face bounding box
  3) Batch ResNet inference
"""

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from facenet_pytorch import MTCNN
from torchvision.models import resnet18
import torchvision.transforms.functional as TF

# =========================
# Configuration
# =========================

RAW_VIDEO_DIR = "mosi/Raw"
CACHE_DIR = "mosi/vision_cache"

K_FRAMES = 32            # 16 is enough + much faster
FACE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DETECT_FRAMES = 3     # 前三帧
BBOX_EXPAND_RATIO = 0.1   # expand bbox by 10%

FER_RESNET_CKPT = (
    "/workspace/Ada-MSHyper/preprocess/"
    "weights_libreface/Facial_Expression_Recognition/weights/resnet.pt"
)

os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# Models
# =========================

mtcnn = MTCNN(
    image_size=FACE_SIZE,
    margin=0,
    device=DEVICE,
    post_process=False,   # we want raw box
    keep_all=False
)

resnet = resnet18(weights=None)
ckpt = torch.load(FER_RESNET_CKPT, map_location=DEVICE)
resnet.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE).eval()

# =========================
# Utilities
# =========================

def aggregate_bboxes(bboxes):
    """
    bboxes: list of [x1, y1, x2, y2]
    return: median-aggregated bbox
    """
    bboxes = np.array(bboxes)
    return np.median(bboxes, axis=0)


def expand_bbox(box, img_w, img_h, ratio=0.15):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    dx = w * ratio
    dy = h * ratio

    x1 = max(0, int(x1 - dx))
    y1 = max(0, int(y1 - dy))
    x2 = min(img_w, int(x2 + dx))
    y2 = min(img_h, int(y2 + dy))

    return x1, y1, x2, y2


def read_all_frames(cap):
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def sample_fixed_k(frames, k):
    if len(frames) == 0:
        return []
    idx = np.linspace(0, len(frames) - 1, k).astype(int)
    return [frames[i] for i in idx]


def crop_with_box(frame, box):
    """
    box: (x1, y1, x2, y2)
    """
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
    return face


# =========================
# Core extraction
# =========================

def extract_clip_features(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    frames = read_all_frames(cap)
    cap.release()

    sampled = sample_fixed_k(frames, K_FRAMES)
    if len(sampled) == 0:
        return np.zeros((K_FRAMES, 512), dtype=np.float32)

    # =========================
    # Step 1: detect face on first N frames
    # =========================
    bboxes = []

    for frame in sampled[:NUM_DETECT_FRAMES]:
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            bboxes.append(boxes[0])

    if len(bboxes) == 0:
        # no face detected at all
        return np.zeros((K_FRAMES, 512), dtype=np.float32)

    # robust bbox aggregation
    box = aggregate_bboxes(bboxes)

    # expand bbox for head movement tolerance
    h, w, _ = sampled[0].shape
    box = expand_bbox(box, w, h, ratio=BBOX_EXPAND_RATIO)

    # =========================
    # Step 2: crop all frames
    # =========================
    faces = []
    valid_mask = []

    for frame in sampled:
        crop = crop_with_box(frame, box)
        if crop is None:
            faces.append(torch.zeros(3, FACE_SIZE, FACE_SIZE))
            valid_mask.append(False)
        else:
            faces.append(TF.to_tensor(crop))
            valid_mask.append(True)

    faces = torch.stack(faces).to(DEVICE)

    # =========================
    # Step 3: batch ResNet
    # =========================
    with torch.no_grad():
        embs = resnet(faces).cpu().numpy()

    for i, ok in enumerate(valid_mask):
        if not ok:
            embs[i] = 0.0

    return embs



# =========================
# Main loop
# =========================

def main():
    for video_id in tqdm(os.listdir(RAW_VIDEO_DIR), desc="Vision cache"):
        vdir = os.path.join(RAW_VIDEO_DIR, video_id)
        if not os.path.isdir(vdir):
            continue

        for fname in sorted(os.listdir(vdir)):
            if not fname.endswith(".mp4"):
                continue

            clip_id = os.path.splitext(fname)[0]
            mp4_path = os.path.join(vdir, fname)

            feats = extract_clip_features(mp4_path)

            np.savez_compressed(
                os.path.join(CACHE_DIR, f"{video_id}_{clip_id}.npz"),
                features=feats
            )


if __name__ == "__main__":
    main()
