from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.ASHyper import BimodalClassifier
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

    def _build_model(self):
        if self.args.data == 'multimodal':
            model = BimodalClassifier(self.args).float()
        else:
            raise ValueError("Only multimodal data is supported")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'[Info] Number of parameters: {num_params}')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.data == 'multimodal':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError("Only multimodal data is supported")

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_data in vali_loader:
                # 多模态数据是字典格式
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].float().to(self.device)

                outputs, _ = self.model(batch_data)
                # 使用多模态标签作为目标
                targets = batch_data['label_multimodal'].long()

                loss = criterion(outputs, targets)
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

            # 每个epoch开始时记录超图结构
            # 使用训练数据中的第一个批次来获取超图结构
            try:
                first_batch = next(iter(train_loader))
                for key in first_batch:
                    if isinstance(first_batch[key], torch.Tensor):
                        first_batch[key] = first_batch[key].float().to(self.device)

                # 记录超图结构（在模型前向传播中会生成超图）
                with torch.no_grad():
                    _ = self.model(first_batch)  # 前向传播会触发超图生成

                # 记录每个模态的超图结构
                for i, modality in enumerate(['text', 'audio']):
                    # 从模型中获取最新生成的超图
                    # 注意：这里假设模型在前向传播后保留了最新的超图结构
                    # 如果模型没有保留，我们需要从前向传播过程中捕获
                    try:
                        # 使用相同的输入来重新生成超图进行记录
                        text_features = first_batch['text_vector'][:1]  # 取第一个样本
                        audio_features = first_batch['audio_vector'][:1]
                        text_mask = first_batch['text_mask'][:1]
                        audio_mask = first_batch['audio_mask'][:1]

                        features = text_features if modality == 'text' else audio_features
                        mask = text_mask if modality == 'text' else audio_mask

                        hypergraphs = self.model.hyper_generators[i](features, mask)
                        hypergraph = hypergraphs[0].to(self.device)
                        self.record_hypergraph_svd(
                            hypergraph, modality, stage='epoch_start', epoch=epoch, batch=0,
                            weight_matrix=self.model.hyper_convs[i].weight
                        )
                    except Exception as e:
                        print(f"Warning: Could not record hypergraph for {modality}: {e}")
            except Exception as e:
                print(f"Warning: Could not record hypergraphs at epoch start: {e}")

            for batch_data in train_loader:
                model_optim.zero_grad()
                # 多模态数据是字典格式
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].float().to(self.device)

                outputs, constrain_loss = self.model(batch_data)
                # 使用多模态标签作为目标
                targets = batch_data['label_multimodal'].long()

                loss = criterion(outputs, targets)
                # 添加超图约束损失
                if constrain_loss > 0:
                    loss = loss + 0.1 * constrain_loss

                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 每个epoch结束时记录超图结构
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

                    # 记录每个模态的超图结构
                    for i, modality in enumerate(['text', 'audio']):
                        try:
                            # 使用相同的输入来重新生成超图进行记录
                            text_features = last_batch['text_vector'][:1]  # 取第一个样本
                            audio_features = last_batch['audio_vector'][:1]
                            text_mask = last_batch['text_mask'][:1]
                            audio_mask = last_batch['audio_mask'][:1]

                            features = text_features if modality == 'text' else audio_features
                            mask = text_mask if modality == 'text' else audio_mask

                            hypergraphs = self.model.hyper_generators[i](features, mask)
                            hypergraph = hypergraphs[0].to(self.device)
                            self.record_hypergraph_svd(
                                hypergraph, modality, stage='epoch_end', epoch=epoch, batch=0,
                                weight_matrix=self.model.hyper_convs[i].weight
                            )
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

        # 训练完全结束时记录最终超图结构
        # 使用测试数据中的第一个批次来获取超图结构
        try:
            test_iter = iter(test_loader)
            first_test_batch = next(test_iter)

            for key in first_test_batch:
                if isinstance(first_test_batch[key], torch.Tensor):
                    first_test_batch[key] = first_test_batch[key].float().to(self.device)

            # 记录超图结构（在模型前向传播中会生成超图）
            with torch.no_grad():
                _ = self.model(first_test_batch)  # 前向传播会触发超图生成

            # 记录每个模态的超图结构
            for i, modality in enumerate(['text', 'audio']):
                try:
                    # 使用相同的输入来重新生成超图进行记录
                    text_features = first_test_batch['text_vector'][:1]  # 取第一个样本
                    audio_features = first_test_batch['audio_vector'][:1]
                    text_mask = first_test_batch['text_mask'][:1]
                    audio_mask = first_test_batch['audio_mask'][:1]

                    features = text_features if modality == 'text' else audio_features
                    mask = text_mask if modality == 'text' else audio_mask

                    hypergraphs = self.model.hyper_generators[i](features, mask)
                    hypergraph = hypergraphs[0].to(self.device)
                    self.record_hypergraph_svd(
                        hypergraph, modality, stage='training_complete', epoch=self.args.train_epochs-1, batch=0,
                        weight_matrix=self.model.hyper_convs[i].weight
                    )
                except Exception as e:
                    print(f"Warning: Could not record final hypergraph for {modality}: {e}")
        except Exception as e:
            print(f"Warning: Could not record final hypergraphs: {e}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            path = os.path.join(self.args.checkpoints, setting)
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
            print(classification_report(all_labels, all_preds, digits=4))

            # 打印混淆矩阵
            print('\nConfusion Matrix:')
            cm = confusion_matrix(all_labels, all_preds)
            print(cm)

            return accuracy

    def record_hypergraph_svd(self, hypergraph, modality, stage='unknown', epoch=0, batch=0, weight_matrix=None):
        """记录超图结构和权重矩阵SVD分解到CSV"""
        csv_file = f'hypergraph_{modality}.csv'

        # 初始化CSV文件（如果不存在）
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['stage', 'epoch', 'batch', 'modality', 'seq_len', 'hyper_num', 'num_connections', 'node_list', 'hyperedge_list', 'weight_singular_values', 'weight_rank', 'weight_condition_number'])

        node_list = hypergraph[0].cpu().numpy().tolist() if hasattr(hypergraph, 'cpu') else hypergraph[0].tolist()
        hyperedge_list = hypergraph[1].cpu().numpy().tolist() if hasattr(hypergraph, 'cpu') else hypergraph[1].tolist()

        # 计算权重矩阵的SVD分解信息
        weight_info = self.compute_weight_svd_info(weight_matrix)

        # 获取模态特定的参数
        seq_len = 160 if modality == 'text' else 518
        hyper_num = getattr(self.args, f'hyper_num_{modality}', 50)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                stage, epoch, batch, modality, seq_len, hyper_num,
                len(node_list), str(node_list), str(hyperedge_list),
                weight_info['singular_values'], weight_info['rank'], weight_info['condition_number']
            ])

    def compute_weight_svd_info(self, weight_matrix=None):
        """计算权重矩阵的SVD分解信息"""
        if weight_matrix is None:
            # 如果没有传入权重矩阵，返回默认值
            return {
                'singular_values': 'N/A',
                'rank': 'N/A',
                'condition_number': 'N/A'
            }

        try:
            # 确保权重矩阵在CPU上并分离梯度
            weight = weight_matrix.detach().cpu().numpy()

            # 检查权重矩阵是否有效
            if np.any(np.isnan(weight)):
                return {
                    'singular_values': 'NaN in weights',
                    'rank': 'NaN',
                    'condition_number': 'NaN'
                }

            if np.any(np.isinf(weight)):
                return {
                    'singular_values': 'Inf in weights',
                    'rank': 'Inf',
                    'condition_number': 'Inf'
                }

            # 确保是二维矩阵
            if weight.ndim != 2:
                return {
                    'singular_values': f'Invalid shape: {weight.shape}',
                    'rank': 'Invalid',
                    'condition_number': 'Invalid'
                }

            # 计算SVD
            U, s, Vt = np.linalg.svd(weight, full_matrices=False)

            # 检查SVD结果
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                return {
                    'singular_values': 'NaN/Inf in singular values',
                    'rank': 'NaN/Inf',
                    'condition_number': 'NaN/Inf'
                }

            # 取前10个奇异值
            top_singular_values = s[:min(10, len(s))].tolist()

            # 计算有效秩
            effective_rank = int(np.sum(s > 1e-6))

            # 计算条件数
            if len(s) > 0 and s[-1] > 1e-12:
                condition_number = float(s[0] / s[-1])
            else:
                condition_number = float('inf')

            return {
                'singular_values': str(top_singular_values),
                'rank': effective_rank,
                'condition_number': condition_number
            }
        except Exception as e:
            return {
                'singular_values': f'Error: {str(e)}',
                'rank': 'Error',
                'condition_number': 'Error'
            }
