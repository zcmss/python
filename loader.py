from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(batch_size=64, valid_ratio=0.1):
    """返回训练、验证、测试集的DataLoader"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载数据集
    train_data = datasets.FashionMNIST(
        root='data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(
        root='data', train=False, download=True, transform=transform)

    # 划分验证集
    n_valid = int(len(train_data) * valid_ratio)
    train_data, valid_data = random_split(
        train_data, [len(train_data) - n_valid, n_valid])

    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


# 类别标签对应关系
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]