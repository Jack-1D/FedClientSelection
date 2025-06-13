import torchvision.transforms as transforms
import torchvision
import numpy as np
from torch.utils.data import Subset
from collections import defaultdict
from torch.utils.data import DataLoader
from drawer import *

def get_training_data(trainset: str = "CIFAR10", 
                      distribution_shifting: bool = True,
                      class_increment: bool = True,
                      data_distribution_alpha: float = 1, 
                      num_clients: int = 10, 
                      num_rounds: int = 400, 
                      batch_size: int = 64, 
                      new_distribution_weight: float = 0.1, 
                      data_size_gain_ratio: float = 0.1, 
                      new_data_size_distribution_weight: float = 0.5,
                      rounds_to_get_new_data: int = 100,
                      data_size_alphas: list[float] = [3.7, 8.2, 10.0, 11.0, 3.3, 6.6, 5.5, 7.4, 4.2, 3.1],
                      start_class_num: int = 5,
                      increment_period: int = 20,
                      random_seed: int = 42):
    """
    根據不同的數據集和分佈生成訓練數據
    :param trainset: 數據集名稱，支持 CIFAR10 和 CIFAR100
    :param data_distribution_alpha: Dirichlet 分配的參數
    :param num_clients: 客戶端數量
    :param num_rounds: 輪數
    :param batch_size: 批次大小
    :param strength: 分佈強度
    :param epsilon: 最小值
    :param rescue_ratio: 救援比例
    :return: list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions, num_class, list_of_client_indices_num
    """
    np.random.seed(random_seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    if trainset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif trainset == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Please choose 'CIFAR10' or 'CIFAR100'.")
    
    if distribution_shifting and class_increment:
        return distribution_shifting_class_increment_training(trainset=trainset,
                                                              data_distribution_alpha=data_distribution_alpha, 
                                                              num_clients=num_clients, 
                                                              num_rounds=num_rounds, 
                                                              batch_size=batch_size, 
                                                              new_distribution_weight=new_distribution_weight, 
                                                              data_size_gain_ratio=data_size_gain_ratio, 
                                                              new_data_size_distribution_weight=new_data_size_distribution_weight, 
                                                              rounds_to_get_new_data=rounds_to_get_new_data, 
                                                              data_size_alphas=data_size_alphas,
                                                              start_class_num=start_class_num,
                                                              increment_period=increment_period)
    elif distribution_shifting:
        return distribution_shifting_training(trainset=trainset,
                                       data_distribution_alpha=data_distribution_alpha, 
                                       num_clients=num_clients, 
                                       num_rounds=num_rounds, 
                                       batch_size=batch_size, 
                                       new_distribution_weight=new_distribution_weight, 
                                       data_size_gain_ratio=data_size_gain_ratio, 
                                       new_data_size_distribution_weight=new_data_size_distribution_weight, 
                                       rounds_to_get_new_data=rounds_to_get_new_data, 
                                       data_size_alphas=data_size_alphas)
    elif class_increment:
        return class_increment_training(trainset=trainset,
                                 data_distribution_alpha=data_distribution_alpha,
                                 num_clients=num_clients,
                                 num_rounds=num_rounds,
                                 batch_size=batch_size,
                                 data_size_gain_ratio=data_size_gain_ratio,
                                 new_data_size_distribution_weight=new_data_size_distribution_weight,
                                 rounds_to_get_new_data=rounds_to_get_new_data,
                                 data_size_alphas=data_size_alphas,
                                 start_class_num=start_class_num,
                                 increment_period=increment_period)
    else:
        raise ValueError("At least one of distribution_shifting or class_increment must be True.")

def get_test_data(testset: str = "CIFAR10",
                  batch_size: int = 128,
                  random_seed: int = 42):
    np.random.seed(random_seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    if testset == "CIFAR10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif testset == "CIFAR100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Please choose 'CIFAR10' or 'CIFAR100'.")
    
    return DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    

def distribution_shifting_training(
        trainset: torchvision.datasets.VisionDataset = None,
        data_distribution_alpha: float = 1, 
        num_clients: int = 10, 
        num_rounds: int = 400, 
        batch_size: int = 64,
        new_distribution_weight: float = 0.1,
        data_size_gain_ratio: float = 0.1,
        new_data_size_distribution_weight: float = 0.5,
        rounds_to_get_new_data: int = 100,
        data_size_alphas: list[float] = [3.7, 8.2, 10.0, 11.0, 3.3, 6.6, 5.5, 7.4, 4.2, 3.1]):
    """
    分布轉移的 CIFAR-100 訓練資料分配
    Args:
        data_distribution_alpha (float): Dirichlet 分布的 alpha 參數，用於控制資料分布的多樣性
        num_clients (int): 客戶端數量
        num_rounds (int): 訓練輪數
        batch_size (int): 每個客戶端的批次大小
        new_distribution_weight (float): 新分布與舊分布的權重比例
        data_size_gain_ratio (float): 資料量增長比例
        new_data_size_distribution_weight (float): 新資料量分布與舊資料量分布的權重比例
        rounds_to_get_new_data (int): 每多少輪獲取新的資料量分布
        data_size_alphas (list[float]): 每個客戶端的資料量比例參數
    Returns:
        list_of_dataLoaders (list): 每輪每個客戶端的 DataLoader
        list_of_data_sizes (list): 每輪每個客戶端的資料量
        list_of_data_distributions (list): 每輪每個客戶端的資料分布
        num_classes (int): 資料集的類別數
        list_of_client_indices_num (list): 每輪每個客戶端每個類別的資料量
        trainset: CIFAR-100 訓練集
    """
    # 檢查 data_size_alphas 的長度是否與 num_clients 相符
    if len(data_size_alphas) != num_clients:
        raise ValueError(f"Length of data_size_alphas ({len(data_size_alphas)}) must match num_clients ({num_clients}).")

    train_pool = defaultdict(list)
    for idx, (image, label) in enumerate(trainset):
        train_pool[label].append(idx)
    
    # 第一輪的總資料量
    total_samples_this_round = len(trainset.targets) / rounds_to_get_new_data
    # 第一輪每個client的資料量佔比 [num_clients]
    data_size_clients_proportions = np.random.dirichlet(data_size_alphas)
    # 初始資料分布 [num_clients, num_classes]
    data_distribution_clients_proportions = np.random.dirichlet([data_distribution_alpha] 
                                                                * len(trainset.classes), num_clients)
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

    # 避免dataLoader為空
    for client_idx in range(num_clients):
        random_class = np.random.choice(len(trainset.classes), size=1, replace=False)[0]
        if len(train_pool[random_class]) > 0:
            selected_indices = np.random.choice(train_pool[random_class], size=min(len(train_pool[random_class]), 1), replace=False)
            client_get_indices[0][client_idx].extend(selected_indices)
            list_of_client_indices_num[0][client_idx][random_class] += len(selected_indices)
            train_pool[random_class] = [idx for idx in train_pool[random_class] if idx not in selected_indices]
            list_of_data_sizes[0][client_idx] += len(selected_indices)

    cumu = [0 for _ in range(num_clients)]
    all_distributions = [data_distribution_clients_proportions]
    record_end = -1
    flag = True
    for round_idx in range(num_rounds):
        for client_idx in range(num_clients):
            for class_idx in range(len(trainset.classes)):
                # 隨機選擇資料
                if len(train_pool[class_idx]) > 0:
                    avail_data_size = min(len(train_pool[class_idx]), round(total_samples_this_round*data_size_clients_proportions[client_idx]*data_distribution_clients_proportions[client_idx][class_idx]))
                    selected_indices = np.random.choice(train_pool[class_idx], size=avail_data_size, replace=False)
                    client_get_indices[round_idx][client_idx].extend(selected_indices)
                    # 紀錄這一輪這個client在這個class的資料量
                    list_of_client_indices_num[round_idx][client_idx][class_idx] += avail_data_size
                    # 更新data pool
                    train_pool[class_idx] = [idx for idx in train_pool[class_idx] if idx not in selected_indices]
                    # 紀錄這一輪的資料量
                    list_of_data_sizes[round_idx][client_idx] += avail_data_size
            # 紀錄這一輪的資料分布
            list_of_data_distributions[round_idx][client_idx].extend(data_distribution_clients_proportions[client_idx])
        # 檢查資料是否用完
        remaining_data = sum([len(indices) for indices in train_pool.values()])
        if remaining_data == 0 and flag:
            record_end = round_idx
            flag = False
        # 總資料量更新
        total_samples_this_round = total_samples_this_round * (1 + np.random.uniform(-data_size_gain_ratio, data_size_gain_ratio))
        total_samples_this_round = np.clip(total_samples_this_round,
                            len(trainset.targets) / rounds_to_get_new_data * (1 - data_size_gain_ratio), 
                            len(trainset.targets) / rounds_to_get_new_data * (1 + data_size_gain_ratio))
        # 資料量分布更新
        new_data_size_clients_proportions = np.random.dirichlet(data_size_alphas)
        data_size_clients_proportions = new_data_size_distribution_weight * new_data_size_clients_proportions + (1 - new_data_size_distribution_weight) * data_size_clients_proportions
        # 資料分布更新
        new_proportions = np.random.dirichlet([data_distribution_alpha] * len(trainset.classes), num_clients)
        data_distribution_clients_proportions = new_distribution_weight * new_proportions + (1 - new_distribution_weight) * data_distribution_clients_proportions
        
        all_distributions.append(data_distribution_clients_proportions)
    # Plot each client's data size per class for each round, separated by client
    draw_client_label_each_round(num_clients, num_rounds, trainset, list_of_client_indices_num)
    # 顯示每一輪資料量消耗的圖
    draw_data_consumption(num_rounds, list_of_data_sizes)
    # 顯示每個label的資料消耗圖
    draw_label_consumption(num_clients, num_rounds, list_of_client_indices_num, trainset)
    # 畫每個client每一輪資料量的圖
    draw_client_data_size(num_clients, num_rounds, list_of_data_sizes)
    # 畫每個client每一輪的累積資料量的圖
    draw_cumulative_data_size(num_clients, num_rounds, list_of_data_sizes)
    
    
    # 每個client累積拿到的indices
    cumu = [[] for _ in range(num_clients)]
    for round_idx in range(num_rounds):
        for client_idx in range(num_clients):
            cumu[client_idx].extend(client_get_indices[round_idx][client_idx])
            # client_train_indices_each_round[round_idx][client_idx] = list(cumu[client_idx])
            loader = DataLoader(Subset(trainset, list(cumu[client_idx])), batch_size=batch_size, shuffle=True)
            list_of_dataLoaders[round_idx][client_idx] = loader
    # 檢查資料是否用完
    remaining_data = sum([len(indices) for indices in train_pool.values()])
    if remaining_data != 0:
        print("Data not exhausted, remaining samples:", remaining_data)
        for class_idx, indices in train_pool.items():
            if len(indices) > 0:
                print(f"Class {class_idx} has {len(indices)} samples remaining.")
    else:
        print("All data exhausted, last round:", record_end)
    end_training_exclusive = record_end + 1 if record_end != -1 else num_rounds
    return list_of_dataLoaders, list_of_data_sizes, len(trainset.classes), list_of_client_indices_num, end_training_exclusive, [-1]

def class_increment_training(
        trainset: torchvision.datasets.VisionDataset = None,
        data_distribution_alpha: float = 1, 
        num_clients: int = 10, 
        num_rounds: int = 600, 
        batch_size: int = 64,
        data_size_gain_ratio: float = 0.1,
        new_data_size_distribution_weight: float = 0.5,
        rounds_to_get_new_data: int = 100,
        data_size_alphas: list[float] = [3.7, 8.2, 10.0, 11.0, 3.3, 6.6, 5.5, 7.4, 4.2, 3.1],
        start_class_num: int = 10,
        increment_period: int = 1):
    """
    分布轉移的 CIFAR-100 訓練資料分配
    Args:
        data_distribution_alpha (float): Dirichlet 分布的 alpha 參數，用於控制資料分布的多樣性
        num_clients (int): 客戶端數量
        num_rounds (int): 訓練輪數
        batch_size (int): 每個客戶端的批次大小
        new_distribution_weight (float): 新分布與舊分布的權重比例
        data_size_gain_ratio (float): 資料量增長比例
        new_data_size_distribution_weight (float): 新資料量分布與舊資料量分布的權重比例
        rounds_to_get_new_data (int): 每多少輪獲取新的資料量分布
        data_size_alphas (list[float]): 每個客戶端的資料量比例參數
    Returns:
        list_of_dataLoaders (list): 每輪每個客戶端的 DataLoader
        list_of_data_sizes (list): 每輪每個客戶端的資料量
        list_of_data_distributions (list): 每輪每個客戶端的資料分布
        num_classes (int): 資料集的類別數
        list_of_client_indices_num (list): 每輪每個客戶端每個類別的資料量
        trainset: CIFAR-100 訓練集
    """
    # 檢查 data_size_alphas 的長度是否與 num_clients 相符
    if len(data_size_alphas) != num_clients:
        raise ValueError(f"Length of data_size_alphas ({len(data_size_alphas)}) must match num_clients ({num_clients}).")

    train_pool = defaultdict(list)
    for idx, (image, label) in enumerate(trainset):
        train_pool[label].append(idx)
    
    # 第一輪的總資料量
    total_samples_this_round = len(trainset.targets) / rounds_to_get_new_data
    # 第一輪每個client的資料量佔比 [num_clients]
    data_size_clients_proportions = np.random.dirichlet(data_size_alphas)
    # 初始資料分布 [num_clients, num_classes]
    data_distribution_clients_proportions = np.random.dirichlet([data_distribution_alpha] 
                                                                * len(trainset.classes), num_clients)
    # 每一輪每個client拿到的資料
    client_get_indices = [[[] for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的dataLoader
    list_of_dataLoaders = [[0 for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client的資料量
    list_of_data_sizes = [[0 for _ in range(num_clients)] for _ in range(num_rounds)]
    # 每一輪每個client每個class的資料量
    list_of_client_indices_num = [[[0 for _ in range(len(trainset.classes))] for _ in range(num_clients)] for _ in range(num_rounds)]
    # 紀錄新class進入的round index
    round_idx_increment = []
    cur_class_num = start_class_num

    # 避免dataLoader為空
    for client_idx in range(num_clients):
        random_class = np.random.choice(start_class_num, size=1, replace=False)[0]
        if len(train_pool[random_class]) > 0:
            selected_indices = np.random.choice(train_pool[random_class], size=min(len(train_pool[random_class]), 1), replace=False)
            client_get_indices[0][client_idx].extend(selected_indices)
            list_of_client_indices_num[0][client_idx][random_class] += len(selected_indices)
            train_pool[random_class] = [idx for idx in train_pool[random_class] if idx not in selected_indices]
            list_of_data_sizes[0][client_idx] += len(selected_indices)

    cumu = [0 for _ in range(num_clients)]
    record_end = -1
    flag = True
    for round_idx in range(num_rounds):
        if (round_idx + 1) % increment_period == 0 and cur_class_num < len(trainset.classes):
            cur_class_num += 1
            round_idx_increment.append(round_idx + 1)
        for client_idx in range(num_clients):
            for class_idx in range(cur_class_num):
                # 隨機選擇資料
                if len(train_pool[class_idx]) > 0:
                    avail_data_size = min(len(train_pool[class_idx]), round(total_samples_this_round*data_size_clients_proportions[client_idx]*data_distribution_clients_proportions[client_idx][class_idx]))
                    selected_indices = np.random.choice(train_pool[class_idx], size=avail_data_size, replace=False)
                    client_get_indices[round_idx][client_idx].extend(selected_indices)
                    # 紀錄這一輪這個client在這個class的資料量
                    list_of_client_indices_num[round_idx][client_idx][class_idx] += avail_data_size
                    # 更新data pool
                    train_pool[class_idx] = [idx for idx in train_pool[class_idx] if idx not in selected_indices]
                    # 紀錄這一輪的資料量
                    list_of_data_sizes[round_idx][client_idx] += avail_data_size
        # 檢查資料是否用完
        remaining_data = sum([len(indices) for indices in train_pool.values()])
        if remaining_data == 0 and flag:
            record_end = round_idx
            flag = False
        # 總資料量更新
        total_samples_this_round = total_samples_this_round * (1 + np.random.uniform(-data_size_gain_ratio, data_size_gain_ratio))
        total_samples_this_round = np.clip(total_samples_this_round,
                            len(trainset.targets) / rounds_to_get_new_data * (1 - data_size_gain_ratio), 
                            len(trainset.targets) / rounds_to_get_new_data * (1 + data_size_gain_ratio))
        # 資料量分布更新
        new_data_size_clients_proportions = np.random.dirichlet(data_size_alphas)
        data_size_clients_proportions = new_data_size_distribution_weight * new_data_size_clients_proportions + (1 - new_data_size_distribution_weight) * data_size_clients_proportions
    # Plot each client's data size per class for each round, separated by client
    draw_client_label_each_round(num_clients, num_rounds, trainset, list_of_client_indices_num)
    # 顯示每一輪資料量消耗的圖
    draw_data_consumption(num_rounds, list_of_data_sizes)
    # 顯示每個label的資料消耗圖
    draw_label_consumption(num_clients, num_rounds, list_of_client_indices_num, trainset)
    # 畫每個client每一輪資料量的圖
    draw_client_data_size(num_clients, num_rounds, list_of_data_sizes)
    # 畫每個client每一輪的累積資料量的圖
    draw_cumulative_data_size(num_clients, num_rounds, list_of_data_sizes)
    
    
    # 每個client累積拿到的indices
    cumu = [[] for _ in range(num_clients)]
    for round_idx in range(num_rounds):
        for client_idx in range(num_clients):
            cumu[client_idx].extend(client_get_indices[round_idx][client_idx])
            loader = DataLoader(Subset(trainset, list(cumu[client_idx])), batch_size=batch_size, shuffle=True)
            list_of_dataLoaders[round_idx][client_idx] = loader
    # 檢查資料是否用完
    remaining_data = sum([len(indices) for indices in train_pool.values()])
    if remaining_data != 0:
        print("Data not exhausted, remaining samples:", remaining_data)
        for class_idx, indices in train_pool.items():
            if len(indices) > 0:
                print(f"Class {class_idx} has {len(indices)} samples remaining.")
    else:
        print("All data exhausted, last round:", record_end)
    end_training_exclusive = record_end + 1 if record_end != -1 else num_rounds
    return list_of_dataLoaders, list_of_data_sizes, len(trainset.classes), list_of_client_indices_num, end_training_exclusive, round_idx_increment

def distribution_shifting_class_increment_training(
        trainset: torchvision.datasets.VisionDataset = None,
        data_distribution_alpha: float = 1, 
        num_clients: int = 10, 
        num_rounds: int = 400, 
        batch_size: int = 64,
        new_distribution_weight: float = 0.1,
        data_size_gain_ratio: float = 0.1,
        new_data_size_distribution_weight: float = 0.5,
        rounds_to_get_new_data: int = 100,
        data_size_alphas: list[float] = [3.7, 8.2, 10.0, 11.0, 3.3, 6.6, 5.5, 7.4, 4.2, 3.1],
        start_class_num: int = 10,
        increment_period: int = 1):
    """
    分布轉移的 CIFAR-100 訓練資料分配
    Args:
        data_distribution_alpha (float): Dirichlet 分布的 alpha 參數，用於控制資料分布的多樣性
        num_clients (int): 客戶端數量
        num_rounds (int): 訓練輪數
        batch_size (int): 每個客戶端的批次大小
        new_distribution_weight (float): 新分布與舊分布的權重比例
        data_size_gain_ratio (float): 資料量增長比例
        new_data_size_distribution_weight (float): 新資料量分布與舊資料量分布的權重比例
        rounds_to_get_new_data (int): 每多少輪獲取新的資料量分布
        data_size_alphas (list[float]): 每個客戶端的資料量比例參數
    Returns:
        list_of_dataLoaders (list): 每輪每個客戶端的 DataLoader
        list_of_data_sizes (list): 每輪每個客戶端的資料量
        list_of_data_distributions (list): 每輪每個客戶端的資料分布
        num_classes (int): 資料集的類別數
        list_of_client_indices_num (list): 每輪每個客戶端每個類別的資料量
        trainset: CIFAR-100 訓練集
    """
    # 檢查 data_size_alphas 的長度是否與 num_clients 相符
    if len(data_size_alphas) != num_clients:
        raise ValueError(f"Length of data_size_alphas ({len(data_size_alphas)}) must match num_clients ({num_clients}).")

    train_pool = defaultdict(list)
    for idx, (image, label) in enumerate(trainset):
        train_pool[label].append(idx)
    
    # 第一輪的總資料量
    total_samples_this_round = len(trainset.targets) / rounds_to_get_new_data
    # 第一輪每個client的資料量佔比 [num_clients]
    data_size_clients_proportions = np.random.dirichlet(data_size_alphas)
    # 初始資料分布 [num_clients, num_classes]
    data_distribution_clients_proportions = np.random.dirichlet([data_distribution_alpha] 
                                                                * len(trainset.classes), num_clients)
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
    # 紀錄新class進入的round index
    round_idx_increment = []
    cur_class_num = start_class_num

    # 避免dataLoader為空
    for client_idx in range(num_clients):
        random_class = np.random.choice(start_class_num, size=1, replace=False)[0]
        if len(train_pool[random_class]) > 0:
            selected_indices = np.random.choice(train_pool[random_class], size=min(len(train_pool[random_class]), 1), replace=False)
            client_get_indices[0][client_idx].extend(selected_indices)
            list_of_client_indices_num[0][client_idx][random_class] += len(selected_indices)
            train_pool[random_class] = [idx for idx in train_pool[random_class] if idx not in selected_indices]
            list_of_data_sizes[0][client_idx] += len(selected_indices)

    cumu = [0 for _ in range(num_clients)]
    all_distributions = [data_distribution_clients_proportions]
    record_end = -1
    flag = True
    for round_idx in range(num_rounds):
        if (round_idx + 1) % increment_period == 0 and cur_class_num < len(trainset.classes):
            cur_class_num += 1
            round_idx_increment.append(round_idx + 1)
        for client_idx in range(num_clients):
            for class_idx in range(cur_class_num):
                # 隨機選擇資料
                if len(train_pool[class_idx]) > 0:
                    avail_data_size = min(len(train_pool[class_idx]), round(total_samples_this_round*data_size_clients_proportions[client_idx]*data_distribution_clients_proportions[client_idx][class_idx]))
                    selected_indices = np.random.choice(train_pool[class_idx], size=avail_data_size, replace=False)
                    client_get_indices[round_idx][client_idx].extend(selected_indices)
                    # 紀錄這一輪這個client在這個class的資料量
                    list_of_client_indices_num[round_idx][client_idx][class_idx] += avail_data_size
                    # 更新data pool
                    train_pool[class_idx] = [idx for idx in train_pool[class_idx] if idx not in selected_indices]
                    # 紀錄這一輪的資料量
                    list_of_data_sizes[round_idx][client_idx] += avail_data_size
            # 紀錄這一輪的資料分布
            list_of_data_distributions[round_idx][client_idx].extend(data_distribution_clients_proportions[client_idx])
        # 檢查資料是否用完
        remaining_data = sum([len(indices) for indices in train_pool.values()])
        if remaining_data == 0 and flag:
            record_end = round_idx
            flag = False
        # 總資料量更新
        total_samples_this_round = total_samples_this_round * (1 + np.random.uniform(-data_size_gain_ratio, data_size_gain_ratio))
        total_samples_this_round = np.clip(total_samples_this_round,
                            len(trainset.targets) / rounds_to_get_new_data * (1 - data_size_gain_ratio), 
                            len(trainset.targets) / rounds_to_get_new_data * (1 + data_size_gain_ratio))
        # 資料量分布更新
        new_data_size_clients_proportions = np.random.dirichlet(data_size_alphas)
        data_size_clients_proportions = new_data_size_distribution_weight * new_data_size_clients_proportions + (1 - new_data_size_distribution_weight) * data_size_clients_proportions
        # 資料分布更新
        new_proportions = np.random.dirichlet([data_distribution_alpha] * len(trainset.classes), num_clients)
        data_distribution_clients_proportions = new_distribution_weight * new_proportions + (1 - new_distribution_weight) * data_distribution_clients_proportions
        
        all_distributions.append(data_distribution_clients_proportions)
    # Plot each client's data size per class for each round, separated by client
    draw_client_label_each_round(num_clients, num_rounds, trainset, list_of_client_indices_num)
    # 顯示每一輪資料量消耗的圖
    draw_data_consumption(num_rounds, list_of_data_sizes)
    # 顯示每個label的資料消耗圖
    draw_label_consumption(num_clients, num_rounds, list_of_client_indices_num, trainset)
    # 畫每個client每一輪資料量的圖
    draw_client_data_size(num_clients, num_rounds, list_of_data_sizes)
    # 畫每個client每一輪的累積資料量的圖
    draw_cumulative_data_size(num_clients, num_rounds, list_of_data_sizes)
    
    
    # 每個client累積拿到的indices
    cumu = [[] for _ in range(num_clients)]
    for round_idx in range(num_rounds):
        for client_idx in range(num_clients):
            cumu[client_idx].extend(client_get_indices[round_idx][client_idx])
            loader = DataLoader(Subset(trainset, list(cumu[client_idx])), batch_size=batch_size, shuffle=True)
            list_of_dataLoaders[round_idx][client_idx] = loader
    # 檢查資料是否用完
    remaining_data = sum([len(indices) for indices in train_pool.values()])
    if remaining_data != 0:
        print("Data not exhausted, remaining samples:", remaining_data)
        for class_idx, indices in train_pool.items():
            if len(indices) > 0:
                print(f"Class {class_idx} has {len(indices)} samples remaining.")
    else:
        print("All data exhausted, last round:", record_end)
    end_training_exclusive = record_end + 1 if record_end != -1 else num_rounds
    return list_of_dataLoaders, list_of_data_sizes, len(trainset.classes), list_of_client_indices_num, end_training_exclusive, round_idx_increment

if __name__ == "__main__":
    # Example usage
    train_loaders, data_sizes, num_classes, client_indices_num, end_rounds, round_idx_increment = get_training_data(
        trainset="CIFAR10",
        distribution_shifting=True,
        class_increment=True,
        data_distribution_alpha=0.5,
        num_clients=10,
        num_rounds=400,
        batch_size=64,
        new_distribution_weight=0.1,
        data_size_gain_ratio=0.1,
        new_data_size_distribution_weight=0.5,
        rounds_to_get_new_data=100,
        data_size_alphas=[3.7, 8.2, 10.0, 11.0, 3.3, 6.6, 5.5, 7.4, 4.2, 3.1],
        start_class_num=5,
        increment_period=20
    )
    print("Training data prepared.")

