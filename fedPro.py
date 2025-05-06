import torch
from data_preprocess import *
import os
import copy
import logging
from model import CNN
import torch.nn.functional as F
from loss import *
from server import FLServer
from client import FLClient

num_clients = 10
num_rounds = 800
epochs_per_client = 5
batch_size = 64
participate_ratio = 0.8
random_seed = 42

alpha = 1
beta = 1
gamma = 1

temperature = 0.8
cs_threshold = 0.9
kl_threshold = 0.001

# 設置隨機種子以確保可重現性
torch.manual_seed(random_seed)
np.random.seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

# 確保 DataLoader 的隨機性可控
generator = torch.Generator()
generator.manual_seed(random_seed)

# 強制使用確定性操作
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, filename='Log.log', filemode='a')

model_type = CNN().apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)

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

def main():
    # 準備數據
    list_of_dataLoaders, list_of_data_sizes, list_of_data_distributions, num_class, list_of_client_indices_num = distribution_shifting_CIFAR10_training(num_rounds=num_rounds)
    # print(num_class)
    print(list_of_client_indices_num)
    testloader = distribution_shifting_CIFAR10_test()
    # print(list_of_data_sizes)
    # print(list_of_data_distributions)
    # for rounds in list_of_dataLoaders:
    #     for client_idx, dataLoader in enumerate(rounds):
    #         print(f"Client {client_idx} has {len(dataLoader.dataset)} samples.")

    Server = FLServer(copy.deepcopy(model_type).to(device), num_class, num_clients)
    list_of_dataLoaders = list(map(list, zip(*list_of_dataLoaders)))    # [client][round]
    Clients = [FLClient(copy.deepcopy(model_type).to(device), list_of_dataLoaders[i], num_class, num_clients) for i in range(num_clients)]

    # 用於記錄每輪的準確率
    accuracies = []

    # 創建保存模型的目錄
    os.makedirs("checkpoints", exist_ok=True)
    accuracy_log_path = "accuracy_log.txt"

    participate_client_num = int(num_clients * participate_ratio)
    client_list = [i for i in range(num_clients)]
    probabilities = [1.0 / num_clients for _ in range(num_clients)]

    # 紀錄從前一次signal到目前的資料累積
    data_size_from_last_signal = [0 for _ in range(num_clients)]
    # 紀錄從前一次signal到目前的各class的資料累積
    label_size_from_last_signal = [[0 for _ in range(num_class)] for _ in range(num_clients)]
    # 紀錄每個client的local data distribution 的normalized shannon entropy
    clients_nse = [0 for _ in range(num_clients)]
    # 紀錄每個client data size的排名
    data_size_from_last_signal_rank = [0 for _ in range(num_clients)]
    # 紀錄前一輪每個client data size的排名
    prev_data_size_from_last_signal_rank = [0 for _ in range(num_clients)]
    # 紀錄從頭到前一次signal每個class的資料累積
    label_size_to_last_signal = [[0 for _ in range(num_class)] for _ in range(num_clients)]
    # 紀錄從頭到目前每個class的資料累積
    label_size_to_cur = [[0 for _ in range(num_class)] for _ in range(num_clients)]
    # 紀錄最後一次signal是第幾輪
    last_signal = 0
    signal = False

    # client_models_state_dict = [global_model.state_dict() for _ in range(num_clients)]
    client_cs_score = [1.0 for _ in range(num_clients)]

    # 聯邦學習訓練
    for round in range(num_rounds):
        signal = False

        selected_clients = np.sort(np.random.choice(client_list, size=participate_client_num, p=probabilities, replace=False))
        print(f"Selected clients for round {round + 1}: {selected_clients}")

        # 傳model給被選到的client
        Server.send_model(Clients, selected_clients)
        # 每個client進行local training
        for client_idx in selected_clients:
            local_state_dict = Clients[selected_clients].client_update(list_of_dataLoaders[round][client_idx], epochs=epochs_per_client)
            # client_models_state_dict[client_idx] = local_state_dict
            # 每個被選到的client計算CS Score
            Clients[selected_clients].compute_cs_score(round)
            client_cs_score[client_idx] = Clients[selected_clients].cs
            print(f"Client {client_idx} CS Score: {client_cs_score[client_idx]:.4f}")
        for client_idx in range(num_clients):
            # 更新data size
            Clients[client_idx].data_size_from_last_signal += list_of_data_sizes[round][client_idx]
            print(f"round: {round}, client: {client_idx}, defference: {list(set(list_of_dataLoaders[round][client_idx].dataset.indices) - set(list_of_dataLoaders[last_signal][client_idx].dataset.indices))}")
            Clients[client_idx].label_size_to_cur = [Clients[client_idx].label_size_to_cur[i] + list_of_client_indices_num[round][client_idx][i] for i in range(num_class)]
            # 更新每個client新增的各label數量
            Clients[client_idx].label_size_from_last_signal = [Clients[client_idx].label_size_from_last_signal[i] + list_of_client_indices_num[round][client_idx][i] for i in range(num_class)]
            print(f"label_size_from_last_signal: {Clients[client_idx].label_size_from_last_signal}")
            print(f"label_size_to_cur: {Clients[client_idx].label_size_to_cur}")
            # 更新nse
            Clients[client_idx].compute_nse()
            clients_nse[client_idx] = Clients[client_idx].nse
        # 更新每個client的資料量佔比
        for client_idx in range(num_clients):
            Clients[client_idx].send_label_size_from_last_signal(Server, client_idx)
        Server.send_all_label_size_from_last_signal(Clients)
        for client_idx in range(num_clients):
            Clients[client_idx].compute_label_size_from_last_signal_rank()
        print(f"label_size_from_last_signal_proportions: {Clients[0].label_size_from_last_signal_proportions}")
        print(f"clients_nse: {clients_nse}")
        print(f"data_size_from_last_signal_rank: {Clients[0].data_size_from_last_signal_rank}")
        print(f"label_size_from_last_signal:{Clients[0].all_label_size_from_last_signal}")
        print([Clients[i].data_size_from_last_signal for i in range(num_clients)])
        print([np.sum(Clients[0].all_label_size_from_last_signal[i]) for i in range(num_clients)])
        print([Clients[i].data_size_from_last_signal / np.sum([Clients[j].data_size_from_last_signal for j in selected_clients]) if i in selected_clients and np.sum([data_size_from_last_signal[j] for j in selected_clients]) != 0 else 1.0 / len(selected_clients) for i in range(num_clients)])

        global_model = server_aggregate(
            global_model, 
            client_models_state_dict, 
            selected_clients, 
            [data_size_from_last_signal[i] / np.sum([data_size_from_last_signal[j] for j in selected_clients]) if i in selected_clients and np.sum([data_size_from_last_signal[j] for j in selected_clients]) != 0 else 1.0 / len(selected_clients) for i in range(num_clients)]
        )


        accuracy, loss = test_model(global_model, testloader)
        accuracies.append(accuracy)
        print(f"Round {round + 1}/{num_rounds}, Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        logging.info(f"Round {round + 1}/{num_rounds}, Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        with open(accuracy_log_path, "a") as f:
            f.write(f"{round + 1},{accuracy:.2f},{loss:.4f}\n")
        if round % 10 == 0:
            cur_model_path = f"checkpoints/global_model_{round+1}.pth"
            save_model(global_model, cur_model_path)

        for client_idx in selected_clients:
            if(client_cs_score[client_idx] < cs_threshold):
                signal = True

        if round > last_signal + 1 and not np.array_equal(prev_data_size_from_last_signal_rank, data_size_from_last_signal_rank):
            signal = True

        for client_idx in range(num_clients):
            epsilon = 1e-10
            # 若還沒signal過，就先用local iid程度來替代
            if np.sum(label_size_to_cur[client_idx]) == 0 or np.sum(label_size_to_last_signal[client_idx]) == 0:
                kl = normalized_shannon_entropy(label_size_to_cur[client_idx])
            else:
                P = torch.tensor([
                    label_size_to_cur[client_idx][i] / np.sum(label_size_to_cur[client_idx]) if np.sum(label_size_to_cur[client_idx]) != 0 else 0 
                    for i in range(num_class)
                ]) + epsilon
                Q = torch.tensor([
                    label_size_to_last_signal[client_idx][i] / np.sum(label_size_to_last_signal[client_idx]) if np.sum(label_size_to_last_signal[client_idx]) != 0 else 0 
                    for i in range(num_class)
                ]) + epsilon
                kl = F.kl_div(P.log(), Q, reduction='batchmean')
                print(f"P: {P}")
                print(f"Q: {Q}")
            print(f"Client {client_idx} KL Divergence: {kl.item():.4f}")
            if kl > kl_threshold:
                signal = True

        if signal:
            print(client_cs_score)
            print(label_size_from_last_signal_proportions)
            print(clients_nse)
            score = [alpha * client_cs_score[i] + beta * label_size_from_last_signal_proportions[i] + gamma * clients_nse[i] for i in range(num_clients)]
            scaled_score = torch.tensor(score) / temperature
            probabilities = F.softmax(scaled_score, dim=0).numpy()
            print(f"Updated probabilities: {probabilities}")
            last_signal = round
            label_size_from_last_signal = [[0 for _ in range(num_class)] for _ in range(num_clients)]
            label_size_to_last_signal = copy.deepcopy(label_size_to_cur)
            data_size_from_last_signal = [0 for _ in range(num_clients)]

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