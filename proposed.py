import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
import os  # 用於檢查文件路徑
import logging
from model import CNN
import torch.nn.functional as F
from data_preprocess import *

num_clients = 10
num_rounds = 5
epochs_per_client = 5
batch_size = 64
participate_ratio = 0.8
random_seed = 42

# 設置隨機種子以確保可重現性
torch.manual_seed(random_seed)
np.random.seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, filename='Log.log', filemode='a')


# 客戶端更新函數
def client_update(model, loader, epochs=1, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

# 伺服器聚合函數（FedAvg）
def server_aggregate(global_model, client_models, client_weights):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])
        for i, client_dict in enumerate(client_models):
            global_dict[key] += client_weights[i] * client_dict[key]
    global_model.load_state_dict(global_dict)
    return global_model

# 測試全局模型
def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    loss = 0.0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    avg_loss = loss / len(testloader)
    return accuracy, avg_loss

# 保存模型函數
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# 加載模型函數
def load_model(model_class, path, device):
    model = model_class().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}")
    return model

def client_count_cs_score(global_model, local_state_dict, dataLoader):
    """
    計算客戶端的 CS Score
    :param global_model: 全局模型
    :param local_state_dict: 客戶端模型的狀態字典
    :param dataLoader: 客戶端數據加載器
    :return: CS Score
    """
    local_model = CNN()
    local_model.load_state_dict(local_state_dict)
    local_model.to(device)

    global_model.eval()
    local_model.eval()
    total_cs = 0.0

    with torch.no_grad():
        for data, _ in dataLoader:
            data = data.to(device)
            global_feature_map = global_model.conv_layers(data)
            local_feature_map = local_model.conv_layers(data)

            global_feature_map_flat = global_feature_map.view(global_feature_map.size(0), -1)
            local_feature_map_flat = local_feature_map.view(local_feature_map.size(0), -1)

            # 計算每個 sample 的 cosine similarity -> shape: [B]
            sim = F.cosine_similarity(global_feature_map_flat, local_feature_map_flat, dim=1)
            print(sim)
            total_cs += sim.cpu().sum().item()  # 把 batch 裡所有 sample 的 loss 加總
            print(total_cs)

    avg_cs_score = total_cs / len(dataLoader.dataset)
    return avg_cs_score

# 主訓練循環
def main():
    # 準備數據
    list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions = distribution_shifting_CIFAR10_training(num_rounds=15)
    testloader = distribution_shifting_CIFAR10_test()
    # print(list_of_data_sizes)
    # print(list_of_data_distributions)
    # for rounds in list_of_dataLoaders:
    #     for client_idx, dataLoader in enumerate(rounds):
    #         print(f"Client {client_idx} has {len(dataLoader.dataset)} samples.")

    # 初始化全局模型
    global_model = CNN().to(device)

    # 根據客戶端數據量計算權重
    # client_sizes = [len(loader.dataset) for loader in client_loaders]
    # total_size = sum(client_sizes)
    # client_weights = [size / total_size for size in client_sizes]

    # 用於記錄每輪的準確率
    accuracies = []

    # 創建保存模型的目錄
    os.makedirs("checkpoints", exist_ok=True)

    participate_client_num = int(num_clients * participate_ratio)
    client_list = [i for i in range(num_clients)]
    probabilities = [1.0 / num_clients for _ in range(num_clients)]

    # 紀錄從前一次signal到目前的資料累積
    data_size_from_last_signal = [0 for _ in range(num_clients)]
    # 紀錄最後一次signal是第幾輪
    last_signal = 0

    # 聯邦學習訓練
    for round in range(num_rounds):
        selected_clients = np.sort(np.random.choice(client_list, size=participate_client_num, p=probabilities, replace=False))
        print(f"Selected clients for round {round + 1}: {selected_clients}")

        client_models = []
        client_cs_loss = []
        for client_idx in selected_clients:
            local_model = copy.deepcopy(global_model).to(device)
            local_state_dict = client_update(local_model, list_of_dataLoaders[round][client_idx], epochs=epochs_per_client)
            client_models.append(local_state_dict)
            # 每個被選到的client計算CS Score
            cs_score = client_count_cs_score(global_model, local_state_dict, list_of_dataLoaders[round][client_idx])
            client_cs_loss.append(cs_score)
            print(f"Client {client_idx} CS Loss: {cs_score:.4f}")
        for client_idx in range(num_clients):
            # 更新data size
            data_size_from_last_signal[client_idx] += list_of_data_sizes[round][client_idx]
            print(f"round: {round}, client: {client_idx}, defference: {list(set(list_of_dataLoaders[round][client_idx].dataset.indices) - set(list_of_dataLoaders[last_signal][client_idx].dataset.indices))}")
        print(data_size_from_last_signal)
        print(np.sum(data_size_from_last_signal))
        print([data_size_from_last_signal[i] / np.sum([data_size_from_last_signal[j] for j in selected_clients]) for i in selected_clients])

        global_model = server_aggregate(global_model, client_models, [data_size_from_last_signal[i] / np.sum([data_size_from_last_signal[j] for j in selected_clients]) for i in selected_clients])


        accuracy, loss = test_model(global_model, testloader)
        accuracies.append(accuracy)
        print(f"Round {round + 1}/{num_rounds}, Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        logging.info(f"Round {round + 1}/{num_rounds}, Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        if round % 10 == 0:
            cur_model_path = f"checkpoints/global_model_{round+1}.pth"
            save_model(global_model, cur_model_path)

    # 保存最終模型
    final_model_path = "checkpoints/global_model_final.pth"
    save_model(global_model, final_model_path)

    # 繪製Round vs Accuracy圖表
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_rounds + 1), accuracies, marker='o', linestyle='-', color='b')
    plt.title('Test Accuracy vs. Communication Round (Dirichlet α=0.1)')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.savefig('accuracy_vs_round_dirichlet.png')
    plt.show()

    # 示例：加載最終模型並測試
    loaded_model = load_model(CNN, final_model_path, device)
    accuracy, loss = test_model(loaded_model, testloader)
    print(f"Loaded Model - Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")

if __name__ == "__main__":
    main()