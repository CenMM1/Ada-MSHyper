from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.ASHyper import MultimodalClassifier
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import csv
from sklearn.metrics import classification_report, confusion_matrix, f1_score

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # 启用cuDNN优化，提升GPU利用率
        torch.backends.cudnn.benchmark = True

    def _build_model(self):
        model = MultimodalClassifier(self.args).float()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'[Info] Number of parameters: {num_params}')
        return model

    def _get_data(self, flag):
        print(f'[Info] Loading {flag} data...')
        data_set, data_loader = data_provider(self.args, flag)
        print(f'[Info] {flag} data loaded: {len(data_set)} samples')
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_data in vali_loader:
                # 多模态数据是字典格式
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].float().to(self.device)

                outputs, reg_loss = self.model(batch_data)
                # 使用多模态标签作为目标
                targets = batch_data['label_multimodal'].long()

                # 论文目标：L_total = L_ER + κ * L_ECR
                loss = criterion(outputs, targets) + reg_loss
                total_loss.append(loss)

        total_loss = torch.stack(total_loss).mean()
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join('./checkpoints', setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_loss = []


            for i, batch_data in enumerate(train_loader):
                if i % 10 == 0:
                    print(f'[Info] Processing batch {i}/{len(train_loader)}')
                model_optim.zero_grad()
                # 多模态数据是字典格式
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].float().to(self.device)

                outputs, reg_loss = self.model(batch_data)
                # 使用多模态标签作为目标
                targets = batch_data['label_multimodal'].long()

                # 论文目标：L_total = L_ER + κ * L_ECR
                loss = criterion(outputs, targets) + reg_loss

                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 每个epoch结束时记录超图连接数
            # 使用训练数据中的最后一个批次来获取超图结构
            try:
                # 获取训练数据加载器的迭代器
                train_iter = iter(train_loader)
                last_batch = None
                for batch_data in train_iter:
                    last_batch = batch_data

                if last_batch is not None:
                    for key in last_batch:
                        if isinstance(last_batch[key], torch.Tensor):
                            last_batch[key] = last_batch[key].float().to(self.device)

                    # 记录超图结构（在模型前向传播中会生成超图）
                    with torch.no_grad():
                        _ = self.model(last_batch)  # 前向传播会触发超图生成

                    # 记录每个模态的超图连接数
                    for i, modality in enumerate(['text', 'audio', 'video']):
                        try:
                            # 使用相同的输入来重新生成超图进行记录
                            text_features = last_batch['text_vector'][:1]  # 取第一个样本
                            audio_features = last_batch['audio_vector'][:1]
                            video_features = last_batch['video_vector'][:1]
                            text_mask = last_batch['text_mask'][:1]
                            audio_mask = last_batch['audio_mask'][:1]
                            video_mask = last_batch['video_mask'][:1]

                            features = {'text': text_features, 'audio': audio_features, 'video': video_features}[modality]
                            mask = {'text': text_mask, 'audio': audio_mask, 'video': video_mask}[modality]

                            hypergraphs = self.model.hyper_generators[i](features, mask)
                            hypergraph = hypergraphs[0].to(self.device)
                            num_connections = len(hypergraph[0])
                            self.record_hypergraph_connections(epoch, modality, num_connections)
                        except Exception as e:
                            print(f"Warning: Could not record hypergraph for {modality}: {e}")
            except Exception as e:
                print(f"Warning: Could not record hypergraphs at epoch end: {e}")

            print(f"Epoch: {epoch + 1}, Time: {time.time() - epoch_time:.2f}s | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))


        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            path = os.path.join('./checkpoints', setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for batch_data in test_loader:
                # 多模态数据是字典格式
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].float().to(self.device)

                outputs, _ = self.model(batch_data)
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                labels = batch_data['label_multimodal'].detach().cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

            # 计算详细的分类指标
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            accuracy = np.mean(all_preds == all_labels)

            # 计算F1分数
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            f1_weighted = f1_score(all_labels, all_preds, average='weighted')

            print(f'Accuracy: {accuracy:.4f}')
            print(f'F1 Score (Macro): {f1_macro:.4f}')
            print(f'F1 Score (Weighted): {f1_weighted:.4f}')

            # 打印分类报告
            print('\nClassification Report:')
            report = classification_report(all_labels, all_preds, digits=4)
            print(report)

            # 打印混淆矩阵
            print('\nConfusion Matrix:')
            cm = confusion_matrix(all_labels, all_preds)
            print(cm)

            # 保存结果到文件
            with open('test_results.txt', 'a') as f:
                f.write(f'Setting: {setting}\n')
                f.write(f'Accuracy: {accuracy:.4f}\n')
                f.write(f'F1 Score (Macro): {f1_macro:.4f}\n')
                f.write(f'F1 Score (Weighted): {f1_weighted:.4f}\n')
                f.write('\nClassification Report:\n')
                f.write(report)
                f.write('\nConfusion Matrix:\n')
                f.write(str(cm))
                f.write('\n' + '='*50 + '\n')

            return accuracy

    def record_hypergraph_connections(self, epoch, modality, num_connections):
        """记录超图连接数到CSV"""
        csv_file = 'hypergraph_connections.csv'

        # 初始化CSV文件（如果不存在）
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'modality', 'num_connections'])

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, modality, num_connections])
