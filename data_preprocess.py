import torchvision
import numpy as np
from torch.utils.data import Subset
from collections import defaultdict
from torch.utils.data import DataLoader
from drawer import *

np.random.seed(42)

def distribution_shifting_CIFAR10_training(alpha: float = 1, num_clients: int = 10, num_rounds: int = 400, batch_size: int = 64 , strength: float = 1e2, epsilon: float = 1e-8, rescue_ratio: float = 0.05):
    """
    使用 Dirichlet 分配生成分佈
    :param alpha: Dirichlet 分配的參數
    :param num_clients: 客戶端數量
    :param num_rounds: 輪數
    :param batch_size: 批次大小
    :param strength: 分佈強度
    :param epsilon: 最小值
    :param rescue_ratio: 救援比例
    :return: list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions, num_class, list_of_client_indices_num
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
    # 每一輪每個client拿到的資料
    client_get_indices = [[[] for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的dataLoader
    list_of_dataLoaders = [[0 for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的資料量
    list_of_data_sizes = [[0 for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的資料分布
    list_of_data_distributions = [[[] for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client每個class的資料量
    list_of_client_indices_num = [[[0 for _ in range(len(trainset.classes))] for _ in range(num_clients)] for _ in range(num_rounds)]
    new_proportions = init_proportions.copy()
    cumu = [0 for _ in range(num_clients)]
    all_distributions = [init_proportions]
    for round_idx in range(num_rounds):
        # print(f"Round {round_idx}, {new_proportions}")
        for client_idx in range(num_clients):
            for class_idx in range(len(trainset.classes)):
                # 隨機選擇資料
                if len(train_pool[class_idx]) > 0:
                    avail_data_size = min(len(train_pool[class_idx]), round(sample_of_each_clients[client_idx]*new_proportions[client_idx][class_idx]))
                    selected_indices = np.random.choice(train_pool[class_idx], size=avail_data_size, replace=False)
                    client_get_indices[round_idx][client_idx].extend(selected_indices)
                    # 紀錄這一輪這個client在這個class的資料量
                    list_of_client_indices_num[round_idx][client_idx][class_idx] += avail_data_size
                    # 更新data pool
                    train_pool[class_idx] = [idx for idx in train_pool[class_idx] if idx not in selected_indices]
                    # 紀錄這一輪的資料量
                    list_of_data_sizes[round_idx][client_idx] += avail_data_size
            # 紀錄這一輪的資料分布
            list_of_data_distributions[round_idx][client_idx].extend(new_proportions[client_idx])
        # 資料量更新
        sample_of_each_clients = [np.clip(samples + np.random.randint(-10, 10), 1, 150) for samples in sample_of_each_clients]
        # 資料分布更新
        new_proportions = (1 - rescue_ratio) * new_proportions + rescue_ratio * init_proportions
        new_proportions = new_proportions * strength
        new_proportions = np.clip(new_proportions, epsilon, None)
        new_proportions = np.array([np.random.dirichlet(a) for a in new_proportions])
        
        # print("proportions:", new_proportions[0])
        all_distributions.append(new_proportions)
    # Plot each client's data size per class for each round, separated by client
    draw_client_label_each_round(num_clients, num_rounds, trainset, list_of_client_indices_num)
    # 顯示每一輪資料量消耗的圖
    draw_data_consumption(num_rounds, list_of_data_sizes)
    # 顯示每個label的資料消耗圖
    draw_label_consumption(num_clients, num_rounds, list_of_client_indices_num, trainset)
    # 每個client累積拿到的indices
    cumu = [[] for _ in range(num_clients)]
    for round_idx in range(num_rounds):
        for client_idx in range(num_clients):
            cumu[client_idx].extend(client_get_indices[round_idx][client_idx])
            loader = DataLoader(Subset(trainset, list(cumu[client_idx])), batch_size=batch_size, shuffle=True)
            list_of_dataLoaders[round_idx][client_idx] = loader
    # print(list_of_data_sizes)
    # print(list_of_data_distributions)

    return list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions, len(trainset.classes), list_of_client_indices_num

def class_incremental_CIFAR10_training(alpha: float = 1, num_clients: int = 10, num_rounds: int = 400, batch_size: int = 64, start_class_num: int = 5, increment_period: int = 20):
    """
    使用 CIFAR-10 數據集進行類別增量學習
    :param num_clients: 客戶端數量
    :param num_rounds: 輪數
    :param batch_size: 批次大小
    :return: dataLoader
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
    proportions = np.random.dirichlet([alpha] * num_clients, len(trainset.classes))
    # 每一輪每個client拿到的資料
    client_get_indices = [[[] for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的dataLoader
    list_of_dataLoaders = [[0 for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的資料量
    list_of_data_sizes = [[0 for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client每個class的資料量
    list_of_client_indices_num = [[[0 for _ in range(len(trainset.classes))] for _ in range(num_clients)] for _ in range(num_rounds)]
    cumu = [0 for _ in range(num_clients)]
    # 紀錄新class進入的round index
    round_idx_increment = []
    cur_class_num = start_class_num
    for round_idx in range(num_rounds):
        if (round_idx + 1) % increment_period == 0 and cur_class_num < len(trainset.classes):
            cur_class_num += 1
            round_idx_increment.append(round_idx + 1)
        for client_idx in range(num_clients):
            for class_idx in range(cur_class_num):
                # 隨機選擇資料
                if len(train_pool[class_idx]) > 0:
                    avail_data_size = min(len(train_pool[class_idx]), round(sample_of_each_clients[client_idx]*proportions[client_idx][class_idx]))
                    selected_indices = np.random.choice(train_pool[class_idx], size=avail_data_size, replace=False)
                    client_get_indices[round_idx][client_idx].extend(selected_indices)
                    # 紀錄這一輪這個client在這個class的資料量
                    list_of_client_indices_num[round_idx][client_idx][class_idx] += avail_data_size
                    # 更新data pool
                    train_pool[class_idx] = [idx for idx in train_pool[class_idx] if idx not in selected_indices]
                    # 紀錄這一輪的資料量
                    list_of_data_sizes[round_idx][client_idx] += avail_data_size
        # 資料量更新
        sample_of_each_clients = [np.clip(samples + np.random.randint(-10, 10), 1, 150) for samples in sample_of_each_clients]
    # Plot each client's data size per class for each round, separated by client
    draw_client_label_each_round(num_clients, num_rounds, trainset, list_of_client_indices_num)
    # 顯示每一輪資料量消耗的圖
    draw_data_consumption(num_rounds, list_of_data_sizes)
    # 顯示每個label的資料消耗圖
    draw_label_consumption(num_clients, num_rounds, list_of_client_indices_num, trainset)
    # 每個client累積拿到的indices
    cumu = [[] for _ in range(num_clients)]
    for round_idx in range(num_rounds):
        for client_idx in range(num_clients):
            cumu[client_idx].extend(client_get_indices[round_idx][client_idx])
            loader = DataLoader(Subset(trainset, list(cumu[client_idx])), batch_size=batch_size, shuffle=True)
            list_of_dataLoaders[round_idx][client_idx] = loader
    # print(list_of_data_sizes)

    return list_of_dataLoaders, list_of_data_sizes, len(trainset.classes), list_of_client_indices_num, round_idx_increment

def CIFAR10_test():
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

def distribution_shifting_CIFAR100_training(alpha: float = 1, num_clients: int = 10, num_rounds: int = 1000, batch_size: int = 64 , strength: float = 1e2, epsilon: float = 1e-8, rescue_ratio: float = 0.05):
    """
    使用 Dirichlet 分配生成分佈
    :param alpha: Dirichlet 分配的參數
    :param num_clients: 客戶端數量
    :param num_rounds: 輪數
    :param batch_size: 批次大小
    :param strength: 分佈強度
    :param epsilon: 最小值
    :param rescue_ratio: 救援比例
    :return: list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions, num_class, list_of_client_indices_num
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_pool = defaultdict(list)
    for idx, (image, label) in enumerate(trainset):
        train_pool[label].append(idx)
    # 初始資料量
    sample_of_each_clients = [np.random.randint(1, 50) for _ in range(num_clients)]
    # 初始資料分布
    init_proportions = np.random.dirichlet([alpha] * len(trainset.classes), num_clients)
    # 每一輪每個client拿到的資料
    client_get_indices = [[[] for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的dataLoader
    list_of_dataLoaders = [[0 for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的資料量
    list_of_data_sizes = [[0 for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的資料分布
    list_of_data_distributions = [[[] for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client每個class的資料量
    list_of_client_indices_num = [[[0 for _ in range(len(trainset.classes))] for _ in range(num_clients)] for _ in range(num_rounds)]
    # 確保每個client擁有第client個label的資料
    for client_idx in range(num_clients):
        if len(train_pool[client_idx]) > 0:
            selected_indices = np.random.choice(train_pool[client_idx], size=min(len(train_pool[client_idx]), 1), replace=False)
            client_get_indices[0][client_idx].extend(selected_indices)
            list_of_client_indices_num[0][client_idx][client_idx] += len(selected_indices)
            train_pool[client_idx] = [idx for idx in train_pool[client_idx] if idx not in selected_indices]
            list_of_data_sizes[0][client_idx] += len(selected_indices)

    new_proportions = init_proportions.copy()
    all_distributions = [init_proportions]
    for round_idx in range(num_rounds):
        # print(f"Round {round_idx}, {new_proportions}")
        for client_idx in range(num_clients):
            for class_idx in range(len(trainset.classes)):
                # 隨機選擇資料
                if len(train_pool[class_idx]) > 0:
                    avail_data_size = min(len(train_pool[class_idx]), round(sample_of_each_clients[client_idx]*new_proportions[client_idx][class_idx]))
                    selected_indices = np.random.choice(train_pool[class_idx], size=avail_data_size, replace=False)
                    client_get_indices[round_idx][client_idx].extend(selected_indices)
                    # 紀錄這一輪這個client在這個class的資料量
                    list_of_client_indices_num[round_idx][client_idx][class_idx] += avail_data_size
                    # 更新data pool
                    train_pool[class_idx] = [idx for idx in train_pool[class_idx] if idx not in selected_indices]
                    # 紀錄這一輪的資料量
                    list_of_data_sizes[round_idx][client_idx] += avail_data_size
            # 紀錄這一輪的資料分布
            list_of_data_distributions[round_idx][client_idx].extend(new_proportions[client_idx])
        # 資料量更新
        sample_of_each_clients = [np.clip(samples + np.random.randint(-10, 10), 1, 200) for samples in sample_of_each_clients]
        # 資料分布更新
        new_proportions = (1 - rescue_ratio) * new_proportions + rescue_ratio * init_proportions
        new_proportions = new_proportions * strength
        new_proportions = np.clip(new_proportions, epsilon, None)
        new_proportions = np.array([np.random.dirichlet(a) for a in new_proportions])
        
        # print("proportions:", new_proportions[0])
        all_distributions.append(new_proportions)
    # Plot each client's data size per class for each round, separated by client
    draw_client_label_each_round(num_clients, num_rounds, trainset, list_of_client_indices_num)
    # 顯示每一輪資料量消耗的圖
    draw_data_consumption(num_rounds, list_of_data_sizes)
    # 顯示每個label的資料消耗圖
    draw_label_consumption(num_clients, num_rounds, list_of_client_indices_num, trainset)
    # 每個client累積拿到的indices
    cumu = [[] for _ in range(num_clients)]
    for round_idx in range(num_rounds):
        for client_idx in range(num_clients):
            cumu[client_idx].extend(client_get_indices[round_idx][client_idx])
            loader = DataLoader(Subset(trainset, list(cumu[client_idx])), batch_size=batch_size, shuffle=True)
            list_of_dataLoaders[round_idx][client_idx] = loader
    # print(list_of_data_sizes)
    # print(list_of_data_distributions)

    return list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions, len(trainset.classes), list_of_client_indices_num

if __name__ == "__main__":
    # distribution_shifting_CIFAR10_training(num_rounds=400)
    # CIFAR10_test()
    # class_incremental_CIFAR10_training(num_rounds=500)
    distribution_shifting_CIFAR100_training()

