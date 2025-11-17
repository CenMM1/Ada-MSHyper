import os
import torch
from torch.utils.data import Dataset


class Dataset_Multimodal_Classification(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='multimodal', data_path='train.pt',
                 target='multimodal', scale=False, timeenc=0, freq='h'):
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
        self.data_path = data_path
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









