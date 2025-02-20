import torch
from sklearn.metrics import classification_report, confusion_matrix
from data.loader import get_dataloaders, CLASS_NAMES
import matplotlib.pyplot as plt
import seaborn as sns
from models.cnn import SimpleCNN
from models.resnet import ResNet18

def evaluate(model_path, model_type='cnn'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    if model_type == 'cnn':
        model = SimpleCNN().to(device)
    elif model_type == 'resnet':
        model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载数据
    _, _, test_loader = get_dataloaders()

    # 测试评估
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    # 分类报告
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # 混淆矩阵可视化
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')


if __name__ == "__main__":
    evaluate(model_path="best_cnn.pth", model_type='cnn')