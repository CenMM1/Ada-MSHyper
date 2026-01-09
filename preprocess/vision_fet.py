"""
Extract frame-level facial features for each video folder.
"""

import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import models
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import resnet18

RAW_VIDEO_DIR = "mosi/Raw"
CACHE_DIR = "mosi/vision_cache"

FPS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CACHE_DIR, exist_ok=True)

mtcnn = MTCNN(image_size=224, margin=20, device=DEVICE)

# ------------------
# ResNet backbone
# ------------------
resnet = resnet18(weights=None)

ckpt = torch.load(
    "/workspace/Ada-MSHyper/preprocess/weights_libreface/Facial_Expression_Recognition/weights/resnet.pt",
    map_location=DEVICE
)

if "model" in ckpt:
    resnet.load_state_dict(ckpt["model"], strict=False)
else:
    resnet.load_state_dict(ckpt, strict=False)

resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE).eval()


def extract_video(video_dir):
    feats, times = [], []
    t = 0.0

    for mp4 in sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4")):
        cap = cv2.VideoCapture(os.path.join(video_dir, mp4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(int(fps // FPS), 1)

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = mtcnn(frame)
                if face is not None:
                    with torch.no_grad():
                        emb = resnet(face.unsqueeze(0).to(DEVICE))
                    feats.append(emb.squeeze(0).cpu().numpy())
                    times.append(t)
                t += 1.0 / FPS
            idx += 1
        cap.release()

    return np.array(feats), np.array(times)


def main():
    for vid in tqdm(os.listdir(RAW_VIDEO_DIR), desc="Vision cache"):
        vdir = os.path.join(RAW_VIDEO_DIR, vid)
        if not os.path.isdir(vdir):
            continue
        feats, times = extract_video(vdir)
        np.savez_compressed(
            os.path.join(CACHE_DIR, f"{vid}.npz"),
            features=feats,
            timestamps=times
        )


if __name__ == "__main__":
    main()
