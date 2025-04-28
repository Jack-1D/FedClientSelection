# import torch
import torchvision
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader

np.random.seed(42)

def distribution_shifting_CIFAR10_training(alpha: float = 1, num_clients: int = 10, num_rounds: int = 20, batch_size: int = 64 , strength: float = 1e2, epsilon: float = 1e-8, rescue_ratio: float = 0.05):
    """
    使用 Dirichlet 分配生成分佈
    :param alpha: Dirichlet 分配的參數
    :param num_clients: 客戶端數量
    :param num_rounds: 輪數
    :param batch_size: 批次大小
    :param strength: 分佈強度
    :param epsilon: 最小值
    :param rescue_ratio: 救援比例
    :return: list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions
    """
    # 加載 CIFAR-10 數據集
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_pool = defaultdict(list)
    for idx, (image, label) in enumerate(trainset):
        train_pool[label].append(idx)
    # 初始資料量
    sample_of_each_clients = [np.random.randint(1, 50) for _ in range(num_clients)]
    # 初始資料分布
    init_proportions = np.random.dirichlet([alpha] * num_clients, len(trainset.classes))
    # 每個client每一輪拿到的資料
    client_get_indices = [[[] for _ in range(num_rounds)] for _ in range(num_clients)]
    # 每個client每一輪的dataLoader
    list_of_dataLoaders = [[[] for _ in range(num_rounds)] for _ in range(num_clients)]
    # 每個client每一輪的資料量
    list_of_data_sizes = [[0 for _ in range(num_rounds)] for _ in range(num_clients)]
    # 每一輪每個client的資料分布
    list_of_data_distributions = [[[] for _ in range(num_clients)] for _ in range(num_rounds)]
    new_proportions = init_proportions.copy()
    cumu = [0 for _ in range(num_clients)]
    all_distributions = [init_proportions]
    for round_idx in range(num_rounds):
        print(f"Round {round_idx}, {new_proportions}")
        for client_idx in range(num_clients):
            for class_idx in range(len(trainset.classes)):
                # 隨機選擇資料
                if len(train_pool[class_idx]) > 0:
                    avail_data_size = min(len(train_pool[class_idx]), round(sample_of_each_clients[client_idx]*new_proportions[client_idx][class_idx]))
                    selected_indices = np.random.choice(train_pool[class_idx], size=avail_data_size, replace=False)
                    client_get_indices[client_idx][round_idx].extend(selected_indices)
                    # 更新data pool
                    train_pool[class_idx] = [idx for idx in train_pool[class_idx] if idx not in selected_indices]
                    # 紀錄這一輪的資料量
                    list_of_data_sizes[client_idx][round_idx] += avail_data_size
            # 紀錄這一輪的資料分布
            list_of_data_distributions[round_idx][client_idx].extend(new_proportions[client_idx])
        # 資料量更新
        sample_of_each_clients = [np.clip(client + np.random.randint(-10, 10), 1, 50) for client in sample_of_each_clients]
        # 資料分布更新
        new_proportions = (1 - rescue_ratio) * new_proportions + rescue_ratio * init_proportions
        new_proportions = new_proportions * strength
        new_proportions = np.clip(new_proportions, epsilon, None)
        new_proportions = np.array([np.random.dirichlet(a) for a in new_proportions])
        
        print("proportions:", new_proportions[0])
        all_distributions.append(new_proportions)
    history = [d[0] for d in all_distributions]
    history = np.array(history)
    for i in range(history.shape[1]):
        plt.plot(history[:, i], label=f'class {i}')
    plt.legend()
    plt.title('Class Probability Over Time (with rescue)')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.show()
    plt.savefig('distribution_drift.png')
    cumu = [[] for _ in range(num_clients)]
    for client_idx in range(num_clients):
        for round_idx in range(num_rounds):
            cumu[client_idx].extend(client_get_indices[client_idx][round_idx])
            loader = DataLoader(Subset(trainset, cumu[client_idx]), batch_size=batch_size, shuffle=True)
            list_of_dataLoaders[client_idx][round_idx] = loader

    return list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions

def distribution_shifting_CIFAR10_test():
    """
    測試分佈漂移
    :return: dataLoader
    """
    # 加載 CIFAR-10 數據集
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=testset.data.shape[0], shuffle=False)

if __name__ == "__main__":
    distribution_shifting_CIFAR10_training()
    distribution_shifting_CIFAR10_test()

