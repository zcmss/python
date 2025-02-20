# models/resnet.py
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # 适配单通道输入
        self.model.fc = nn.Linear(512, 10)