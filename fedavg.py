import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
import os  # 用於檢查文件路徑
import logging
from model import CNN
from data_preprocess import prepare_data

num_clients = 10
num_rounds = 60
epochs_per_client = 5
batch_size = 64
participate_ratio = 1
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

# 主訓練循環
def main():
    # 準備數據
    client_loaders, testloader = prepare_data(num_clients, batch_size, alpha=1)

    # 初始化全局模型
    global_model = CNN().to(device)

    # 根據客戶端數據量計算權重
    client_sizes = [len(loader.dataset) for loader in client_loaders]
    total_size = sum(client_sizes)
    client_weights = [size / total_size for size in client_sizes]

    # 用於記錄每輪的準確率
    accuracies = []

    # 創建保存模型的目錄
    os.makedirs("checkpoints", exist_ok=True)

    # 聯邦學習訓練
    for round in range(num_rounds):
        if round % 20 == 0:
            epochs_per_client -= 1
        m = max(1, int(participate_ratio * num_clients))
        selected_clients = np.random.choice(range(num_clients), m, replace=False)

        client_models = []
        for client_idx in selected_clients:
            local_model = copy.deepcopy(global_model).to(device)
            local_state_dict = client_update(local_model, client_loaders[client_idx], epochs=epochs_per_client)
            client_models.append(local_state_dict)

        global_model = server_aggregate(global_model, client_models, [client_weights[i] for i in selected_clients])

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