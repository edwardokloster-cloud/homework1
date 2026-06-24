# -*- coding: utf-8 -*-
"""
心跳信号分类预测实验代码：PyTorch 1D-CNN

运行示例：
python heartbeat_cnn_pytorch.py --csv_path train.csv --epochs 20 --batch_size 256
python heartbeat_cnn_pytorch.py --csv_path train.csv --epochs 20 --batch_size 256 --device cuda
"""

import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_heartbeat_signal(signal_str):
    return np.array(signal_str.split(','), dtype=np.float32)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'heartbeat_signals' not in df.columns or 'label' not in df.columns:
        raise ValueError('train.csv 必须包含 heartbeat_signals 和 label 两列')

    X = np.stack(df['heartbeat_signals'].apply(parse_heartbeat_signal).values)
    y = df['label'].astype(int).values
    ids = df['id'].values if 'id' in df.columns else np.arange(len(df))
    return df, X, y, ids


def standardize_by_train(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std() + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, float(mean), float(std)


class HeartbeatDataset(Dataset):
    def __init__(self, X, y=None, ids=None):
        # Conv1d 输入格式：[batch_size, channels, signal_length]
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index], self.ids[index]
        return self.X[index], self.y[index]


class HeartbeatCNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * signals.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, macro_f1, np.array(all_labels), np.array(all_preds), all_probs


def plot_curve(history, output_dir):
    output_dir = Path(output_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
    plt.plot(history['epoch'], history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_curve.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['train_macro_f1'], label='Train Macro-F1')
    plt.plot(history['epoch'], history['test_macro_f1'], label='Test Macro-F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1')
    plt.title('Macro-F1 Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'macro_f1_curve.png', dpi=300)
    plt.close()


def plot_confusion_matrix(cm, output_dir):
    output_dir = Path(output_dir)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.colorbar()

    num_classes = cm.shape[0]
    plt.xticks(range(num_classes), range(num_classes))
    plt.yticks(range(num_classes), range(num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PyTorch CNN 心跳信号分类预测实验')
    parser.add_argument('--csv_path', type=str, default='train.csv')
    parser.add_argument('--output_dir', type=str, default='heartbeat_cnn_output')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print('=' * 60)
    print('心跳信号分类预测实验：PyTorch 1D-CNN')
    print('=' * 60)
    print(f'数据文件: {args.csv_path}')
    print(f'输出目录: {output_dir}')
    print(f'运行设备: {device}')

    df, X, y, ids = load_data(args.csv_path)
    num_classes = len(np.unique(y))
    signal_length = X.shape[1]

    print('\n========== 数据基本信息 ==========')
    print(f'样本数量: {len(df)}')
    print(f'心跳信号长度: {signal_length}')
    print(f'类别数量: {num_classes}')
    print('类别分布:')
    print(pd.Series(y).value_counts().sort_index())

    class_distribution = pd.Series(y).value_counts().sort_index()
    class_distribution.to_csv(output_dir / 'class_distribution.csv', header=['count'])

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids,
        test_size=0.2,
        random_state=args.seed,
        stratify=y
    )

    X_train, X_test, train_mean, train_std = standardize_by_train(X_train, X_test)

    pd.DataFrame({'id': id_train, 'label': y_train}).to_csv(output_dir / 'trainset_split.csv', index=False)
    pd.DataFrame({'id': id_test, 'label': y_test}).to_csv(output_dir / 'testset_split.csv', index=False)

    print('\n========== 8:2 数据划分 ==========')
    print(f'训练集数量: {len(X_train)}')
    print(f'测试集数量: {len(X_test)}')

    train_dataset = HeartbeatDataset(X_train, y_train, id_train)
    test_dataset = HeartbeatDataset(X_test, y_test, id_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = class_counts.sum() / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = HeartbeatCNN(num_classes=num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )

    print('\n========== 模型结构 ==========')
    print(model)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_macro_f1': [],
        'test_loss': [],
        'test_acc': [],
        'test_macro_f1': []
    }

    best_f1 = -1.0
    best_model_path = output_dir / 'best_heartbeat_cnn.pt'

    print('\n========== 开始训练 ==========')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_loss, test_acc, test_f1, y_true, y_pred, y_prob = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(test_f1)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_macro_f1'].append(train_f1)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_macro_f1'].append(test_f1)

        print(
            f'Epoch [{epoch:03d}/{args.epochs}] '
            f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} || '
            f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}'
        )

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'num_classes': num_classes,
                    'signal_length': signal_length,
                    'train_mean': train_mean,
                    'train_std': train_std,
                    'args': vars(args)
                },
                best_model_path
            )

    pd.DataFrame(history).to_csv(output_dir / 'training_log.csv', index=False)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_acc, final_f1, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, device
    )

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print('\n========== 最终测试集结果 ==========')
    print(f'Accuracy: {final_acc:.4f}')
    print(f'Macro-F1: {final_f1:.4f}')
    print(f'Loss: {final_loss:.4f}')
    print('\n分类报告:')
    print(report)
    print('混淆矩阵:')
    print(cm)

    prob_columns = [f'label_{i}_prob' for i in range(num_classes)]
    pred_df = pd.DataFrame(y_prob, columns=prob_columns)
    pred_df.insert(0, 'id', id_test)
    pred_df.insert(1, 'true_label', y_true)
    pred_df.insert(2, 'pred_label', y_pred)
    pred_df.to_csv(output_dir / 'testset_prediction_probability.csv', index=False)

    pd.DataFrame(cm).to_csv(output_dir / 'confusion_matrix.csv', index=False)

    with open(output_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    plot_curve(history, output_dir)
    plot_confusion_matrix(cm, output_dir)

    results = {
        'model': 'PyTorch 1D-CNN',
        'loss_function': 'Weighted CrossEntropyLoss',
        'optimizer': 'AdamW',
        'num_samples': int(len(df)),
        'signal_length': int(signal_length),
        'num_classes': int(num_classes),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'accuracy': float(final_acc),
        'macro_f1': float(final_f1),
        'loss': float(final_loss),
        'confusion_matrix': cm.tolist(),
        'class_distribution': {str(k): int(v) for k, v in class_distribution.to_dict().items()},
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'seed': args.seed,
        'device': str(device)
    }

    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print('\n========== 输出文件 ==========' )
    print(f'最佳模型: {best_model_path}')
    print(f'训练日志: {output_dir / "training_log.csv"}')
    print(f'测试集预测概率: {output_dir / "testset_prediction_probability.csv"}')
    print(f'实验结果: {output_dir / "results.json"}')
    print(f'混淆矩阵图: {output_dir / "confusion_matrix.png"}')
    print(f'Loss 曲线: {output_dir / "loss_curve.png"}')
    print(f'Accuracy 曲线: {output_dir / "accuracy_curve.png"}')
    print(f'Macro-F1 曲线: {output_dir / "macro_f1_curve.png"}')


if __name__ == '__main__':
    main()
