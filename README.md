# FashionMNIST分类项目

## 概述
基于PyTorch实现的FashionMNIST图像分类项目，包含以下功能：
- 支持CNN和ResNet模型训练
- 模型性能评估与可视化
- REST API模型部署
- MLflow实验跟踪

## 环境依赖
- Python 3.8+
- PyTorch 2.0+
- Flask 2.0+
- MLflow 2.0+
- scikit-learn
- matplotlib

## 项目结构
    .
    ├── models/
    │   ├── cnn.py
    │   └── resnet.py
    ├── data/
    │   └── loader.py
    ├── app/
    │   └── app.py
    ├── train.py
    ├── evaluate.py
    └── README.md
## 性能指标
| 模型   | 准确率 | 参数量 |
|--------|--------|--------|
| CNN    | 92.3%  | 1.2M   |
| ResNet | 93.8%  | 11.7M  |

## 注意事项
1. 输入图片应为28x28灰度图，背景为黑色
2. 训练前请确保GPU可用（自动检测CUDA）
3. ResNet实现需手动适配单通道输入

## 许可证
Apache 2.0 License

