import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim

# 导入你的数据加载器和模型
# 注意：确保这里导入路径与你的工程目录结构一致
from dataprovider.data_factory import data_provider
from unet.model import MultiPeriodUNetInpainter


# ==========================================
# 1. 实用工具：早停机制 (Early Stopping)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
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

def train_and_evaluate(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用计算设备: {device}")

    # 1. 先获取数据加载器
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    # 2. 然后打印统计量（此时加载器已存在）
    train_batch = next(iter(train_loader))[0]  # batch_x
    val_batch = next(iter(val_loader))[0]
    print(f"训练集 batch 均值: {train_batch.mean().item():.4f}, 标准差: {train_batch.std().item():.4f}")
    print(f"验证集 batch 均值: {val_batch.mean().item():.4f}, 标准差: {val_batch.std().item():.4f}")



    model = MultiPeriodUNetInpainter(num_vars=args.enc_in, top_k=args.top_k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoints, exist_ok=True)
    model_save_path = os.path.join(args.checkpoints, 'unet_inpainter_best.pth')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=model_save_path)

    for epoch in range(args.train_epochs):
        # -------------------- 训练 --------------------
        model.train()
        train_loss = []
        epoch_start_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            B, seq_len, C = batch_x.shape
            # batch_y 的形状为 [B, label_len + pred_len, C]
            label_len = args.label_len
            pred_len = args.pred_len

            # 修正点1：将历史、已知未来段、待预测未来段拼接成完整序列
            # 注意：batch_y 的前 label_len 是已知的未来段，后 pred_len 是待预测目标
            # ===== 原逻辑（删除）=====
            # true_total_seq = torch.cat([batch_x, batch_y], dim=1)
            # mask = torch.ones_like(true_total_seq)
            # mask[:, seq_len + label_len:, :] = 0
            # predicted_total_seq = model(true_total_seq, mask)
            # future_pred = predicted_total_seq[:, -pred_len:, :]
            # future_y = batch_y[:, -pred_len:, :]

            # ===== 新逻辑（替换）=====
            future_y = batch_y[:, -pred_len:, :]  # 监督目标
            future_placeholder = torch.zeros_like(future_y)  # 不喂未来真值
            model_input = torch.cat([batch_x, future_placeholder], dim=1)  # [B, seq_len+pred_len, C]

            mask = torch.ones_like(model_input)
            mask[:, seq_len:, :] = 0  # 未来区域全遮罩

            predicted_total_seq = model(model_input, mask)
            future_pred = predicted_total_seq[:, -pred_len:, :]

            loss = criterion(future_pred, future_y)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"\t迭代: {i + 1}, Epoch: {epoch + 1} | 训练 Loss: {loss.item():.7f}")

        train_loss_avg = np.average(train_loss)
        print(f"Epoch: {epoch + 1} 训练耗时: {time.time() - epoch_start_time:.2f}s")

        # -------------------- 验证 --------------------
        model.eval()
        val_loss = []
        model.eval()
        val_loss, val_mae, val_rmse = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                B, seq_len, C = batch_x.shape
                label_len, pred_len = args.label_len, args.pred_len

                future_y = batch_y[:, -pred_len:, :]  # 监督目标
                future_placeholder = torch.zeros_like(future_y)  # 不喂未来真值
                model_input = torch.cat([batch_x, future_placeholder], dim=1)  # [B, seq_len+pred_len, C]

                mask = torch.ones_like(model_input)
                mask[:, seq_len:, :] = 0  # 未来区域全遮罩

                predicted_total_seq = model(model_input, mask)
                future_pred = predicted_total_seq[:, -pred_len:, :]

                mse = criterion(future_pred, future_y).item()
                mae = torch.mean(torch.abs(future_pred - future_y)).item()
                rmse = torch.sqrt(torch.mean((future_pred - future_y) ** 2)).item()

                val_loss.append(mse)
                val_mae.append(mae)
                val_rmse.append(rmse)

        val_loss_avg = np.average(val_loss)
        val_mae_avg = np.average(val_mae)
        val_rmse_avg = np.average(val_rmse)

        print(f"Epoch: {epoch + 1} | 验证 MSE: {val_loss_avg:.7f} | MAE: {val_mae_avg:.7f} | RMSE: {val_rmse_avg:.7f}")
        early_stopping(val_loss_avg, model)
        if early_stopping.early_stop:
            print("连续多次验证集 Loss 未下降，触发早停机制，停止训练。")
            break

    model.load_state_dict(torch.load(model_save_path))
    return model

# ==========================================
# 3. 主函数与命令行参数配置
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D U-Net Masked Reconstruction for Time Series')

    # 基础配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='数据集存放文件夹')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件名称')
    parser.add_argument('--features', type=str, default='M', help='预测任务 M:多变量预测多变量')
    parser.add_argument('--target', type=str, default='OT', help='目标列 (单变量预测时使用)')
    parser.add_argument('--freq', type=str, default='h', help='时间特征编码的频率')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型权重保存路径')
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码方式')

    # 序列长度与维度参数
    parser.add_argument('--seq_len', type=int, default=96, help='历史序列长度 (如过去96小时)')
    parser.add_argument('--label_len', type=int, default=48, help='提供给解码器的引导长度 (对于我们的重建任务不敏感)')
    parser.add_argument('--pred_len', type=int, default=96, help='要预测的未来长度')
    parser.add_argument('--enc_in', type=int, default=7, help='变量数/特征数 (ETTh1为7)')

    # 针对 U-Net 的定制参数
    # 考虑到你使用的是 12GB 显存，top_k 建议设为 1 或 2，batch_size 设为 16
    parser.add_argument('--top_k', type=int, default=2, help='FFT 选取的显著周期分支数')

    # 训练配置
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 线程数 (Windows建议设0)')
    parser.add_argument('--train_epochs', type=int, default=1090, help='训练总轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--patience', type=int, default=8, help='早停容忍的轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')

    # 冗余参数 (为了兼容官方 data_provider 中存在的调用)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    args = parser.parse_args()

    print('实验参数配置:')
    print(args)

    # 启动训练
    trained_model = train_and_evaluate(args)
    print("训练及验证全流程执行完毕！")