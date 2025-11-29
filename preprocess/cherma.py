import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


idx_to_label = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'suprise',
}


labels_en = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
labels_ch = ['愤怒', '厌恶', '恐惧', '高兴', '平静', '悲伤', '惊奇']



cuda2 = torch.device('cuda:2')


class MMSAATBaselineDataset(Dataset):
    def __init__(self, stage):
        
        self.stage = stage
        self.dataset_path = r'C:\Users\Jiade\Documents\Github\Ada-MSHyper\dataset\CHERMA20230723_v2\CHERMA0723\\' + self.stage + '.json'

        self.filename_label_list = []

        with open(self.dataset_path, encoding='utf-8') as f:
            for example in json.load(f):
                a = example['audio_file'].replace('.wav', '')
                v = example['video_file']
                self.filename_label_list.append((a, v, example['txt_label'], example['audio_label'], example['visual_label'], example['video_label']))
                
    def __len__(self):
        return len(self.filename_label_list)

    def __getitem__(self, idx):
        current_filename, current_filename_v, label_t, label_a, label_v, label_m = self.filename_label_list[idx]
        
        text_vector = np.load(r'C:\Users\Jiade\Documents\Github\Ada-MSHyper\dataset\CHERMA20230723_v2\CHERMA0723\text\\' + self.stage + '\\' + current_filename + '.npy')
        text_vector = torch.from_numpy(text_vector)

        video_vector = np.load(r'C:\Users\Jiade\Documents\Github\Ada-MSHyper\dataset\CHERMA20230723_v2\CHERMA0723\visual\\' + self.stage + '\\' + current_filename + '.mp4.npy')
        video_vector = torch.from_numpy(video_vector)

        audio_vector = np.load(r'C:\Users\Jiade\Documents\Github\Ada-MSHyper\dataset\CHERMA20230723_v2\CHERMA0723\audio\\' + self.stage + '\\' + current_filename + '.npy')
        audio_vector = torch.from_numpy(audio_vector)

        #return audio_calibrated, audio_vector, text_calibrated, text_vector, labels_ch.index(current_label)
        return  text_vector, audio_vector, video_vector, labels_ch.index(label_t), labels_ch.index(label_a), labels_ch.index(label_v), labels_ch.index(label_m)
        
        

import torch
from data_prepare import MMSAATBaselineDataset
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 加载前5000个样本（分批加载避免内存溢出）
def load_sample_data(max_samples=5000, batch_size=100):
    train_dataset = MMSAATBaselineDataset('train')
    dev_dataset = MMSAATBaselineDataset('dev')
    test_dataset = MMSAATBaselineDataset('test')

    all_data = []

    # 计算每个数据集的样本数
    train_samples = min(len(train_dataset), max_samples // 3)
    dev_samples = min(len(dev_dataset), max_samples // 3)
    test_samples = max_samples - train_samples - dev_samples

    print(f"计划加载: Train {train_samples}, Dev {dev_samples}, Test {test_samples}")

    # 分批加载train数据
    for start_idx in range(0, train_samples, batch_size):
        end_idx = min(start_idx + batch_size, train_samples)
        batch_data = [train_dataset[i] for i in range(start_idx, end_idx)]
        all_data.extend(batch_data)
        print(f"已加载 train {end_idx}/{train_samples}")

    # 分批加载dev数据
    for start_idx in range(0, dev_samples, batch_size):
        end_idx = min(start_idx + batch_size, dev_samples)
        batch_data = [dev_dataset[i] for i in range(start_idx, end_idx)]
        all_data.extend(batch_data)
        print(f"已加载 dev {end_idx}/{dev_samples}")

    # 分批加载test数据
    for start_idx in range(0, test_samples, batch_size):
        end_idx = min(start_idx + batch_size, test_samples)
        batch_data = [test_dataset[i] for i in range(start_idx, end_idx)]
        all_data.extend(batch_data)
        print(f"已加载 test {end_idx}/{test_samples}")

    print(f"总数据量: {len(all_data)}")
    return all_data

# 划分数据集
def split_dataset(all_data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    # 打乱数据
    np.random.seed(42)
    indices = np.random.permutation(len(all_data))
    all_data = [all_data[i] for i in indices]

    n_total = len(all_data)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    n_test = n_total - n_train - n_dev

    train_data = all_data[:n_train]
    dev_data = all_data[n_train:n_train + n_dev]
    test_data = all_data[n_train + n_dev:]

    print(f"划分后 - Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

    return train_data, dev_data, test_data

# 转换为torch tensors（处理变长序列）
def data_to_tensors(data_list):
    text_vectors = []
    audio_vectors = []
    video_vectors = []
    labels_t = []
    labels_a = []
    labels_v = []
    labels_m = []

    for item in data_list:
        text_vec, audio_vec, video_vec, label_t, label_a, label_v, label_m = item
        text_vectors.append(text_vec)
        audio_vectors.append(audio_vec)
        video_vectors.append(video_vec)
        labels_t.append(label_t)
        labels_a.append(label_a)
        labels_v.append(label_v)
        labels_m.append(label_m)

    # 检查序列长度
    text_lengths = [len(vec) for vec in text_vectors]
    audio_lengths = [len(vec) for vec in audio_vectors]
    video_lengths = [len(vec) for vec in video_vectors]

    print(f"Text lengths: min={min(text_lengths)}, max={max(text_lengths)}")
    print(f"Audio lengths: min={min(audio_lengths)}, max={max(audio_lengths)}")
    print(f"Video lengths: min={min(video_lengths)}, max={max(video_lengths)}")

    # 对于变长序列，使用pad_sequence
    from torch.nn.utils.rnn import pad_sequence

    text_tensors = pad_sequence(text_vectors, batch_first=True, padding_value=0.0)
    audio_tensors = pad_sequence(audio_vectors, batch_first=True, padding_value=0.0)
    video_tensors = pad_sequence(video_vectors, batch_first=True, padding_value=0.0)

    labels_t_tensor = torch.tensor(labels_t, dtype=torch.long)
    labels_a_tensor = torch.tensor(labels_a, dtype=torch.long)
    labels_v_tensor = torch.tensor(labels_v, dtype=torch.long)
    labels_m_tensor = torch.tensor(labels_m, dtype=torch.long)

    return text_tensors, audio_tensors, video_tensors, labels_t_tensor, labels_a_tensor, labels_v_tensor, labels_m_tensor

# 保存数据集
def save_datasets(train_data, dev_data, test_data, save_path='./processed_data/'):
    import os
    os.makedirs(save_path, exist_ok=True)

    # 转换为tensors
    train_tensors = data_to_tensors(train_data)
    dev_tensors = data_to_tensors(dev_data)
    test_tensors = data_to_tensors(test_data)

    # 保存
    torch.save(train_tensors, os.path.join(save_path, 'train.pt'))
    torch.save(dev_tensors, os.path.join(save_path, 'dev.pt'))
    torch.save(test_tensors, os.path.join(save_path, 'test.pt'))

    print(f"数据集已保存到 {save_path}")

# 创建DataLoader
def create_dataloaders(train_tensors, dev_tensors, test_tensors, batch_size=32):
    train_dataset = TensorDataset(*train_tensors)
    dev_dataset = TensorDataset(*dev_tensors)
    test_dataset = TensorDataset(*test_tensors)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader

if __name__ == "__main__":
    # 加载前5000个样本并划分数据
    all_data = load_sample_data(max_samples=5000)
    train_data, dev_data, test_data = split_dataset(all_data)

    # 保存为torch格式
    save_datasets(train_data, dev_data, test_data)

    # 示例：创建DataLoader
    train_tensors = data_to_tensors(train_data)
    dev_tensors = data_to_tensors(dev_data)
    test_tensors = data_to_tensors(test_data)

    train_loader, dev_loader, test_loader = create_dataloaders(train_tensors, dev_tensors, test_tensors)

    print("数据处理完成！")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
