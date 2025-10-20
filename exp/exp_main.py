from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.ASHyper import Model as ASHyper, BimodalClassifier
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'multimodal':
            model = BimodalClassifier(self.args).float()
        else:
            model = ASHyper(self.args).float()
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
            return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            if self.args.data == 'multimodal':
                for batch_data in vali_loader:
                    # 多模态数据是字典格式
                    for key in batch_data:
                        if isinstance(batch_data[key], torch.Tensor):
                            batch_data[key] = batch_data[key].float().to(self.device)

                    outputs, _ = self.model(batch_data)
                    # 使用多模态标签作为目标 (暂时使用第一个标签)
                    targets = batch_data['label_multimodal'].long()

                    loss = criterion(outputs, targets)
                    total_loss.append(loss)
            else:
                for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    outputs, _ = self.model(batch_x, batch_x_mark)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]

                    loss = criterion(outputs, batch_y)
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

            if self.args.data == 'multimodal':
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
            else:
                for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    outputs, _ = self.model(batch_x, batch_x_mark)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    loss.backward()
                    model_optim.step()

            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

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
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            if self.args.data == 'multimodal':
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
            else:
                for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    outputs, _ = self.model(batch_x, batch_x_mark)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]

                    preds.append(outputs.detach().cpu().numpy())
                    trues.append(batch_y.detach().cpu().numpy())

                preds = np.concatenate(preds, axis=0)
                trues = np.concatenate(trues, axis=0)

                mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
                print(f'mse:{mse}, mae:{mae}')
                return
