import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_Multimodal_Classification(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='multimodal',
                 target='multimodal', scale=False, timeenc=0, freq='h', args=None):
        # size [seq_len, label_len, pred_len] - 对于分类任务，seq_len表示时间序列长度
        if size == None:
            self.seq_len = 12  # 默认序列长度
        else:
            self.seq_len = size[0]

        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        # 根据flag选择对应的数据文件
        if self.flag == 'train':
            data_file = os.path.join(self.root_path, 'train.pt')
        elif self.flag == 'val':
            data_file = os.path.join(self.root_path, 'dev.pt')  # dev.pt作为验证集
        else:  # test
            data_file = os.path.join(self.root_path, 'test.pt')

        data = torch.load(data_file)

        # 数据是元组格式，依次为：
        # 0: text_vector, 1: audio_vector, 2: video_vector
        # 3: labels_ch.index(label_t), 4: labels_ch.index(label_a)
        # 5: labels_ch.index(label_v), 6: labels_ch.index(label_m)
        self.text_vectors = data[0]      # [batch_size, seq_len, text_dim]
        self.audio_vectors = data[1]    # [batch_size, seq_len, audio_dim]
        self.video_vectors = data[2]    # [batch_size, seq_len, video_dim]

        # 标签数据
        self.labels_text = data[3]        # [batch_size]
        self.labels_audio = data[4]       # [batch_size]
        self.labels_video = data[5]       # [batch_size]
        self.labels_multimodal = data[6]  # [batch_size]

        # 获取数据维度
        self.num_samples = self.text_vectors.shape[0]

        # 使用全部数据，不再进行分割
        self.indices = list(range(self.num_samples))

    def __getitem__(self, index):
        idx = self.indices[index]

        # 获取多模态特征序列
        text_seq = self.text_vectors[idx]      # [seq_len, text_dim]
        audio_seq = self.audio_vectors[idx]    # [seq_len, audio_dim]
        video_seq = self.video_vectors[idx]    # [seq_len, video_dim]

        # 处理时间序列补零：创建掩码来标识非零元素
        text_mask = (text_seq != 0).any(dim=-1).float()    # [seq_len]
        audio_mask = (audio_seq != 0).any(dim=-1).float()  # [seq_len]
        video_mask = (video_seq != 0).any(dim=-1).float()  # [seq_len]

        # 获取标签
        label_text = self.labels_text[idx]
        label_audio = self.labels_audio[idx]
        label_video = self.labels_video[idx]
        label_multimodal = self.labels_multimodal[idx]

        # 返回多模态特征、掩码和标签
        return {
            'text_vector': text_seq,      # [seq_len, text_dim]
            'audio_vector': audio_seq,    # [seq_len, audio_dim]
            'video_vector': video_seq,    # [seq_len, video_dim]
            'text_mask': text_mask,       # [seq_len]
            'audio_mask': audio_mask,     # [seq_len]
            'video_mask': video_mask,     # [seq_len]
            'label_text': label_text,     # scalar
            'label_audio': label_audio,   # scalar
            'label_video': label_video,   # scalar
            'label_multimodal': label_multimodal  # scalar
        }

    def __len__(self):
        return len(self.indices)


class Dataset_PKL_Multimodal_Classification(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='multimodal',
                 target='multimodal', scale=False, timeenc=0, freq='h', args=None):
        if size is None:
            self.seq_len = 12
        else:
            self.seq_len = size[0]

        assert flag in ['train', 'test', 'val']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.args = args

        self.pkl_path = getattr(args, 'pkl_path', None) if args is not None else None
        self.label_map = int(getattr(args, 'label_map', 7)) if args is not None else 7
        self.exclude_oob = bool(getattr(args, 'exclude_oob', 1)) if args is not None else True

        self.__read_data__()

    def _get_split_name(self):
        if self.flag == 'train':
            return 'train'
        if self.flag == 'val':
            return 'valid'
        return 'test'

    def _map_label(self, y):
        y_val = float(y)

        if self.label_map == 2:
            return 1 if y_val > 0 else 0

        if self.label_map == 5:
            if y_val < -2 or y_val > 2:
                return None
            if y_val < -1:
                return 0
            if y_val < 0:
                return 1
            if y_val < 1:
                return 2
            if y_val <= 2:
                return 3
            return None

        if self.label_map == 7:
            # 7-class mapping with clamped tails: <=-2 and >=2 are edge classes
            bins = [-2, -1, 0, 1, 2]
        else:
            raise ValueError(f"Unsupported label_map: {self.label_map}")

        for idx, edge in enumerate(bins):
            if y_val < edge:
                return idx
        return len(bins)

    def __read_data__(self):
        if self.pkl_path is None:
            raise ValueError('pkl_path is required for pkl data_format')

        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)

        split = self._get_split_name()
        if split not in data:
            raise ValueError(f"Split '{split}' not found in {self.pkl_path}")

        split_data = data[split]

        self.text_list = split_data.get('text', [])
        self.audio_list = split_data.get('audio', [])
        self.video_list = split_data.get('vision', [])

        self.audio_lengths = split_data.get('audio_lengths', [])
        self.video_lengths = split_data.get('vision_lengths', [])
        self.labels_reg = split_data.get('regression_labels', [])

        num_samples = len(split_data.get('id', []))
        self.indices = list(range(num_samples))

        if self.label_map == 5 and self.exclude_oob:
            filtered = []
            for i in self.indices:
                y = self.labels_reg[i]
                mapped = self._map_label(y)
                if mapped is None:
                    continue
                filtered.append(i)
            self.indices = filtered

    def __getitem__(self, index):
        idx = self.indices[index]

        text_seq = self.text_list[idx]
        audio_seq = self.audio_list[idx]
        video_seq = self.video_list[idx]

        text_tensor = torch.as_tensor(text_seq, dtype=torch.float32)
        audio_tensor = torch.as_tensor(audio_seq, dtype=torch.float32)
        video_tensor = torch.as_tensor(video_seq, dtype=torch.float32)

        text_mask = (text_tensor != 0).any(dim=-1).float()

        if len(self.audio_lengths) > idx:
            al = int(self.audio_lengths[idx])
        else:
            al = audio_tensor.shape[0]
        if len(self.video_lengths) > idx:
            vl = int(self.video_lengths[idx])
        else:
            vl = video_tensor.shape[0]

        audio_mask = torch.zeros(audio_tensor.shape[0], dtype=torch.float32)
        video_mask = torch.zeros(video_tensor.shape[0], dtype=torch.float32)
        if al > 0:
            audio_mask[:min(al, audio_mask.shape[0])] = 1.0
        if vl > 0:
            video_mask[:min(vl, video_mask.shape[0])] = 1.0

        label_mapped = self._map_label(self.labels_reg[idx])
        if label_mapped is None:
            label_mapped = 0

        label_multimodal = torch.tensor(label_mapped, dtype=torch.long)

        return {
            'text_vector': text_tensor,
            'audio_vector': audio_tensor,
            'video_vector': video_tensor,
            'text_mask': text_mask,
            'audio_mask': audio_mask,
            'video_mask': video_mask,
            'label_text': label_multimodal,
            'label_audio': label_multimodal,
            'label_video': label_multimodal,
            'label_multimodal': label_multimodal,
        }

    def __len__(self):
        return len(self.indices)






