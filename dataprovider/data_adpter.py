import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim


from dataprovider.data_factory import data_provider
from unet.model import MultiPeriodUNetInpainter


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping 计数: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'验证集 Loss 下降 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


# ==========================================
# 2. 核心训练与验证逻辑
# ==========================================
def train_and_evaluate(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用计算设备: {device}")

    # 获取数据加载器
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    # 初始化 2D U-Net 掩码重建模型
    # args.enc_in 就是特征/传感器的数量 (ETT 默认为 7)
    model = MultiPeriodUNetInpainter(num_vars=args.enc_in, top_k=args.top_k).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoints, exist_ok=True)
    model_save_path = os.path.join(args.checkpoints, 'unet_inpainter_best.pth')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=model_save_path)

    for epoch in range(args.train_epochs):
        # -------------------------
        # 训练阶段
        # -------------------------
        model.train()
        train_loss = []
        epoch_start_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            B, seq_len, C = batch_x.shape
            pred_len = batch_y.shape[1] - args.label_len

            # 【数据适配器】：拼接完整画布
            future_y = batch_y[:, -pred_len:, :]
            true_total_seq = torch.cat([batch_x, future_y], dim=1)

            # 【数据适配器】：生成掩码 (1 为历史，0 为未来待预测区域)
            mask = torch.ones_like(true_total_seq).to(device)
            mask[:, seq_len:, :] = 0

            # 前向传播：2D U-Net 掩码重建
            predicted_total_seq = model(true_total_seq, mask)

            # 截取未来预测区域，计算 Loss
            future_pred = predicted_total_seq[:, -pred_len:, :]
            loss = criterion(future_pred, future_y)
            train_loss.append(loss.item())

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"\t迭代: {i + 1}, Epoch: {epoch + 1} | 训练 Loss: {loss.item():.7f}")

        train_loss_avg = np.average(train_loss)
        print(f"Epoch: {epoch + 1} 耗时: {time.time() - epoch_start_time:.2f}s")

        # -------------------------
        # 验证阶段
        # -------------------------
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                B, seq_len, C = batch_x.shape
                pred_len = batch_y.shape[1] - args.label_len

                future_y = batch_y[:, -pred_len:, :]
                true_total_seq = torch.cat([batch_x, future_y], dim=1)

                mask = torch.ones_like(true_total_seq).to(device)
                mask[:, seq_len:, :] = 0

                predicted_total_seq = model(true_total_seq, mask)
                future_pred = predicted_total_seq[:, -pred_len:, :]

                loss = criterion(future_pred, future_y)
                val_loss.append(loss.item())

        val_loss_avg = np.average(val_loss)
        print(f"Epoch: {epoch + 1} | 训练集 Loss: {train_loss_avg:.7f} | 验证集 Loss: {val_loss_avg:.7f}")

        # 早停判断
        early_stopping(val_loss_avg, model)
        if early_stopping.early_stop:
            print("触发早停机制，停止训练。")
            break

    # 加载表现最好的模型权重
    model.load_state_dict(torch.load(model_save_path))
    return model


# ==========================================
# 3. 主函数与命令行参数
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D U-Net Masked Reconstruction for Time Series Forecasting')

    # 基础配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the dataset file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='dataset file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')

    # 序列长度与维度参数
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (通道数/变量数)')

    # 针对我们的 U-Net 模型的定制参数
    parser.add_argument('--top_k', type=int, default=2, help='FFT 取前 k 个周期分支 (12G显存建议 1 或 2)')

    # 训练配置 (已针对 12G VRAM 优化)
    parser.add_argument('--num_workers', type=int, default=0, help='dataset loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input dataset')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    # 其他冗余参数 (为了兼容官方 data_provider 不报错)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    args = parser.parse_args()

    print('实验参数配置:')
    print(args)

    # 启动训练
    trained_model = train_and_evaluate(args)
    print("训练全流程结束！")