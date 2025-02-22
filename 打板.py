import os
import pathlib
import random
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction


class MyDataset(Dataset):
    def __init__(self, data_dir, time_filter="< 2023-01-01", context_length=60, prediction_length=1):
        self.data_dir = data_dir
        self.time_filter_op = time_filter.split(' ')[0]
        self.time_filter_point = datetime.fromisoformat(
            time_filter.split(' ')[1].strip())
        self.context_length = context_length
        self.prediction_length = prediction_length
        if isinstance(self.data_dir, list):
            temp = []
            for dir in self.data_dir:
                temp = temp + list(pathlib.Path(dir).glob("*.csv"))
            files = temp
        else:
            files = list(pathlib.Path(self.data_dir).glob("*.csv"))
        filtered_files = [f for f in files if self._is_file_valid(f)]

        with Pool(cpu_count()) as p:
            raw_data = list(tqdm(p.imap(self.read_csv, filtered_files), total=len(
                filtered_files), desc="Loading files"))
        self.data = [(features.cuda(), labels.cuda())
                     for features, labels in raw_data]

    def _is_file_valid(self, file_path):
        """检查文件是否符合时间过滤条件"""
        timestamp_str = file_path.stem.split('_')[0]  # 提取时间戳部分
        file_time = datetime.fromtimestamp(int(timestamp_str))
        if self.time_filter_op == "<":
            return file_time < self.time_filter_point
        elif self.time_filter_op == ">":
            return file_time > self.time_filter_point
        else:
            raise ValueError("Unsupported time filter operation")

    @staticmethod
    def read_csv(file_path):
        # 开盘,收盘,最高,最低,ave,ave_2,ave_3,ave_4,ave_5,ave_6,ave_7,ave_8,ave_9,ave_10,ave_11,ave_12,ave_13,ave_14,ave_15,ave_16,ave_17,ave_18,ave_19,ave_20
        df = pd.read_csv(file_path)
        numpy_data = df.values
        tensor_data = torch.tensor(numpy_data, dtype=torch.float32)

        features = tensor_data[:-1, :]
        labels = tensor_data[-1:, :]

        return features, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        past_values, future_values = self.data[index]
        return {"past_values": past_values, "future_values": future_values}

# TODO: 尝试更多情况
class CustomLoss(nn.Module):
    def __init__(self, penalty_up_up=0.8, penalty_up_down=5, penalty_down_up=1, penalty_down_down=0.8, penalty_high_down=8):
        super(CustomLoss, self).__init__()
        self.penalty_up_up = penalty_up_up          # 预测上涨实际上涨
        self.penalty_up_down = penalty_up_down      # 预测上涨实际下跌
        self.penalty_down_up = penalty_down_up      # 预测下跌实际上涨
        self.penalty_down_down = penalty_down_down  # 预测下跌实际下跌
        self.penalty_high_down = penalty_high_down  # 第二天最高价小于昨日收盘价

    def forward(self, outputs, future_values, past_values):
        base_loss = abs(outputs - future_values)

        last_day_past_values = past_values[:, -1:, :5]
        last_day_open, last_day_close, last_day_high, last_day_low, last_day_ave = [
            last_day_past_values[:, :, i] for i in range(5)]

        outputs_open, outputs_close, outputs_high, outputs_low, outputs_ave = [
            outputs[:, :, i] for i in range(5)]
        future_values_open, future_values_close, future_values_high, future_values_low, future_values_ave = [
            future_values[:, :, i] for i in range(5)]

        def calculate_raising(last_day_value, future_value):
            return (future_value - last_day_value) / torch.clamp(last_day_value, min=1e-8)

        actuals = [future_values_open, future_values_close,
                   future_values_high, future_values_low, future_values_ave]
        preds = [outputs_open, outputs_close,
                 outputs_high, outputs_low, outputs_ave]
        lasts = [last_day_open, last_day_close,
                 last_day_high, last_day_low, last_day_ave]

        act_raise = calculate_raising(lasts[1], actuals[1])
        pred_raise = calculate_raising(lasts[1], preds[1])
        high_down = actuals[2] - lasts[1]

        penalty = torch.where((pred_raise >= 0) & (
            act_raise >= 0), self.penalty_up_up, 1)
        penalty = torch.where((pred_raise >= 0) & (
            act_raise < 0), self.penalty_up_down, penalty)
        penalty = torch.where((pred_raise < 0) & (
            act_raise >= 0), self.penalty_down_up, penalty)
        penalty = torch.where((pred_raise < 0) & (
            act_raise < 0), self.penalty_down_down, penalty)
        penalty = torch.where(high_down <= 0, self.penalty_high_down, penalty)

        penaltied_loss = penalty * base_loss

        return penaltied_loss.mean()


def train_model(train_dataset, val_dataset, epoch_num, check_point_path=None):
    train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=192, shuffle=False)

    config = PatchTSMixerConfig(
        context_length=60,
        prediction_length=1,
        patch_length=10,
        num_input_channels=24,
        patch_stride=5,
        d_model=48,
        num_layers=6,
        expansion_factor=2,
        dropout=0.2,
        head_dropout=0.2,
        mode="mix_channel",
        scaling="std",
    )
    model = PatchTSMixerForPrediction(config).cuda()
    model.train()
    if check_point_path != None:
        model.load_state_dict(torch.load(check_point_path))
    loss_fn = CustomLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters())

    save_dir = "./model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_val_losses = [float('inf'), float('inf')]
    best_model_paths = ["", ""]

    for epoch in range(epoch_num):
        print(f"Epoch {epoch + 1}/{epoch_num}")

        # 训练阶段
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(
            train_loader, desc=f'Training Epoch {epoch + 1}', unit='batch')

        for batch in progress_bar:
            past_values = batch["past_values"]
            future_values = batch["future_values"][:, :, :5]
            outputs = model(past_values)['prediction_outputs'][:, :, :5]
            loss = loss_fn(outputs, future_values, past_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            progress_bar.set_postfix({'loss': '{:.4f}'.format(loss.item())})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            progress_bar = tqdm(
                val_loader, desc=f'Evaling Epoch {epoch + 1}', unit='batch')
            for batch in progress_bar:
                past_values = batch["past_values"]
                future_values = batch["future_values"][:, :, :5]
                outputs = model(past_values)['prediction_outputs'][:, :, :5]
                loss = loss_fn(outputs, future_values, past_values)
                total_val_loss += loss.item()
                progress_bar.set_postfix(
                    {'loss': '{:.4f}'.format(loss.item())})

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # 保存最新的模型
        latest_model_path = os.path.join(save_dir, f"model_latest.pth")
        torch.save(model.state_dict(), latest_model_path)

        # 更新最佳模型
        if avg_val_loss < max(best_val_losses):
            worst_best_idx = best_val_losses.index(max(best_val_losses))
            best_val_losses[worst_best_idx] = avg_val_loss
            best_model_path = os.path.join(
                save_dir, f"model_best_{worst_best_idx}.pth")
            torch.save(model.state_dict(), best_model_path)
            best_model_paths[worst_best_idx] = best_model_path

        # 日志记录
        log_str = f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n"
        with open(os.path.join(save_dir, "training_log.txt"), "a") as log_file:
            log_file.write(log_str)


if __name__ == "__main__":
    train_dataset = MyDataset(
        ["./locked_limit_up", "./limit_up_open"], time_filter="< 2024-01-01")
    val_dataset = MyDataset(
        ["./locked_limit_up", "./limit_up_open"], time_filter="> 2024-04-01")
    train_model(train_dataset, val_dataset, epoch_num=100, check_point_path="./model_checkpoints/model_latest.pth")
