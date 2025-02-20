import argparse
import torch
from torch import optim
from data.loader import get_dataloaders
from models.cnn import SimpleCNN
from models.resnet import ResNet18  # 需实现ResNet
import mlflow
import torch.nn as nn
# 在代码中设置环境变量
import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"


def train(args):
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, _ = get_dataloaders(args.batch_size)

    # 选择模型
    if args.model == 'cnn':
        model = SimpleCNN().to(device)
    elif args.model == 'resnet':
        model = ResNet18().to(device)

    # 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # MLflow跟踪
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        best_acc = 0.0
        for epoch in range(args.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证阶段
            model.eval()
            valid_acc = 0.0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    valid_acc += (outputs.argmax(1) == labels).sum().item()

            # 计算指标
            train_loss = train_loss / len(train_loader)
            valid_acc = valid_acc / len(valid_loader.dataset)
            scheduler.step(valid_acc)

            # 记录指标
            mlflow.log_metrics({
                "train_loss": train_loss,
                "valid_acc": valid_acc
            }, step=epoch)

            # 保存最佳模型
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), f"best_{args.model}.pth")

        print(f"Best Validation Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train(args)