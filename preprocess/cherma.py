import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np


idx_to_label = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


labels_en = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
labels_ch = ["愤怒", "厌恶", "恐惧", "高兴", "平静", "悲伤", "惊奇"]


class CHERMADataset(Dataset):
    def __init__(self, root_dir, stage):
        self.stage = stage
        self.root_dir = root_dir
        self.dataset_path = os.path.join(self.root_dir, f"{self.stage}.json")

        self.filename_label_list = []

        with open(self.dataset_path, encoding="utf-8") as f:
            for example in json.load(f):
                audio_id = example["audio_file"].replace(".wav", "")
                video_id = example["video_file"]
                self.filename_label_list.append(
                    (
                        audio_id,
                        video_id,
                        example["txt_label"],
                        example["audio_label"],
                        example["visual_label"],
                        example["video_label"],
                    )
                )

    def __len__(self):
        return len(self.filename_label_list)

    def __getitem__(self, idx):
        current_filename, current_filename_v, label_t, label_a, label_v, label_m = self.filename_label_list[idx]

        text_vector = np.load(
            os.path.join(self.root_dir, "text", self.stage, f"{current_filename}.npy")
        )
        text_vector = torch.from_numpy(text_vector)

        video_vector = np.load(
            os.path.join(self.root_dir, "visual", self.stage, f"{current_filename}.mp4.npy")
        )
        video_vector = torch.from_numpy(video_vector)

        audio_vector = np.load(
            os.path.join(self.root_dir, "audio", self.stage, f"{current_filename}.npy")
        )
        audio_vector = torch.from_numpy(audio_vector)

        return (
            text_vector,
            audio_vector,
            video_vector,
            labels_ch.index(label_t),
            labels_ch.index(label_a),
            labels_ch.index(label_v),
            labels_ch.index(label_m),
        )

def _get_lengths_and_dims(dataset):
    text_lengths = []
    audio_lengths = []
    video_lengths = []
    text_dim = audio_dim = video_dim = None

    for idx in range(len(dataset)):
        text_vec, audio_vec, video_vec, _, _, _, _ = dataset[idx]
        text_lengths.append(len(text_vec))
        audio_lengths.append(len(audio_vec))
        video_lengths.append(len(video_vec))

        if text_dim is None:
            text_dim = text_vec.shape[1:]
        if audio_dim is None:
            audio_dim = audio_vec.shape[1:]
        if video_dim is None:
            video_dim = video_vec.shape[1:]

    return text_lengths, audio_lengths, video_lengths, text_dim, audio_dim, video_dim


def stage_to_tensors(dataset, dtype=torch.float16):
    text_lengths, audio_lengths, video_lengths, text_dim, audio_dim, video_dim = _get_lengths_and_dims(dataset)

    max_text = max(text_lengths) if text_lengths else 0
    max_audio = max(audio_lengths) if audio_lengths else 0
    max_video = max(video_lengths) if video_lengths else 0

    print(f"Text lengths: min={min(text_lengths)}, max={max_text}")
    print(f"Audio lengths: min={min(audio_lengths)}, max={max_audio}")
    print(f"Video lengths: min={min(video_lengths)}, max={max_video}")

    num_samples = len(dataset)

    text_tensors = torch.zeros((num_samples, max_text, *text_dim), dtype=dtype)
    audio_tensors = torch.zeros((num_samples, max_audio, *audio_dim), dtype=dtype)
    video_tensors = torch.zeros((num_samples, max_video, *video_dim), dtype=dtype)

    labels_t = torch.empty((num_samples,), dtype=torch.long)
    labels_a = torch.empty((num_samples,), dtype=torch.long)
    labels_v = torch.empty((num_samples,), dtype=torch.long)
    labels_m = torch.empty((num_samples,), dtype=torch.long)

    for idx in range(num_samples):
        text_vec, audio_vec, video_vec, label_t, label_a, label_v, label_m = dataset[idx]
        t_len = len(text_vec)
        a_len = len(audio_vec)
        v_len = len(video_vec)

        text_tensors[idx, :t_len] = text_vec.to(dtype)
        audio_tensors[idx, :a_len] = audio_vec.to(dtype)
        video_tensors[idx, :v_len] = video_vec.to(dtype)

        labels_t[idx] = label_t
        labels_a[idx] = label_a
        labels_v[idx] = label_v
        labels_m[idx] = label_m

    return text_tensors, audio_tensors, video_tensors, labels_t, labels_a, labels_v, labels_m


def save_stage_dataset(root_dir, stage, save_path="./processed_data", dtype=torch.float16):
    os.makedirs(save_path, exist_ok=True)
    dataset = CHERMADataset(root_dir, stage)
    print(f"开始处理 {stage}: {len(dataset)} 条")
    stage_tensors = stage_to_tensors(dataset, dtype=dtype)
    out_path = os.path.join(save_path, f"{stage}.pt")
    torch.save(stage_tensors, out_path)
    print(f"{stage} 数据集已保存到 {out_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cherma_root = os.path.join(script_dir, "CHERMA0723")
    output_dir = os.path.join(script_dir, "processed_data")

    # 依次处理完整的 train/dev/test（不抽样、不划分）
    # 默认使用 float16 降低内存占用
    save_stage_dataset(cherma_root, "train", save_path=output_dir, dtype=torch.float16)
    save_stage_dataset(cherma_root, "dev", save_path=output_dir, dtype=torch.float16)
    save_stage_dataset(cherma_root, "test", save_path=output_dir, dtype=torch.float16)

    print("CHERMA 数据处理完成！")
