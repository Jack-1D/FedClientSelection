import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.nn import functional as F

class FLServer:
    def __init__(self, model_type, total_class, num_clients, num_rounds, lr, momentum=0.9, weight_decay=5e-4):
        self.total_class = total_class
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_type
        self.all_data_size_from_last_signal = [[0 for _ in range(total_class)] for _ in range(num_clients)]
        self.signal_list = [False for _ in range(num_clients)]
        self.clients_cs_score = [0.0 for _ in range(num_clients)]
        self.clients_label_size_from_last_signal_proportions = [0.0 for _ in range(num_clients)]
        self.clients_nse = [0.0 for _ in range(num_clients)]
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_rounds)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def server_aggregate(self, Clients, selected_clients, client_weights):
        # 1. 聚合模型參數
        self.global_dict = self.model.state_dict()
        for key in self.global_dict.keys():
            self.global_dict[key] = torch.zeros_like(self.global_dict[key], dtype=torch.float32)
            for client_idx in selected_clients:
                self.global_dict[key] += client_weights[client_idx] * Clients[client_idx].model.state_dict()[key].float()
        self.model.load_state_dict(self.global_dict)
        
        # 2. 聚合 momentum buffer
        # 取得所有 client 的 optimizer state_dict
        client_optim_states = [Clients[client_idx].optimizer.state_dict() for client_idx in selected_clients]
        server_optim_state = self.optimizer.state_dict()

        # momentum buffer 通常在 state[param_id]["momentum_buffer"]
        for param_id in server_optim_state["state"]:
            # 檢查這個 param_id 是否有 momentum_buffer
            if "momentum_buffer" in server_optim_state["state"][param_id]:
                avg_buffer = torch.zeros_like(server_optim_state["state"][param_id]["momentum_buffer"])
                for i, client_idx in enumerate(selected_clients):
                    client_state = client_optim_states[i]
                    # 有些 client 可能沒有這個 param_id（例如剛開始），要判斷一下
                    if param_id in client_state["state"] and "momentum_buffer" in client_state["state"][param_id]:
                        avg_buffer += client_weights[client_idx] * client_state["state"][param_id]["momentum_buffer"]
                server_optim_state["state"][param_id]["momentum_buffer"] = avg_buffer

        # 更新 server optimizer 的 momentum buffer
        self.optimizer.load_state_dict(server_optim_state)
        self.scheduler.step()

    def send_model(self, Clients):
        for client_idx in range(len(Clients)):
            Clients[client_idx].receive_model(self.model.state_dict(), self.optimizer.state_dict())

    def test_model(self, testloader):
        self.model = self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        loss = 0.0

        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_loss = loss / len(testloader)
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        return accuracy, avg_loss
    
    def receive_data_size_from_last_signal(self, data_size_from_last_signal, client_idx):
        self.all_data_size_from_last_signal[client_idx] = data_size_from_last_signal

    def send_all_data_size_from_last_signal(self, Clients):
        for client_idx in range(len(Clients)):
            Clients[client_idx].receive_all_data_size_from_last_signal(self.all_data_size_from_last_signal)

    def request_to_recompute_probabilities(self, Clients):
        for client_idx in range(len(Clients)):
            self.clients_cs_score[client_idx], self.clients_label_size_from_last_signal_proportions[client_idx], self.clients_nse[client_idx] = Clients[client_idx].response_server_request()
        
    def recompute_probabilities(self, alpha, beta, gamma, temperature):
        print(self.clients_cs_score)
        print(self.clients_label_size_from_last_signal_proportions)
        print(self.clients_nse)
        score = [(alpha * self.clients_cs_score[i] + beta * self.clients_label_size_from_last_signal_proportions[i] + gamma * self.clients_nse[i]) / (alpha + beta + gamma) for i in range(self.num_clients)]
        scaled_score = torch.tensor(score) / temperature
        probabilities = F.softmax(scaled_score, dim=0).numpy()
        return probabilities
    
    def do_snapshot(self, Clients):
        for client_idx in range(len(Clients)):
            Clients[client_idx].do_snapshot()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")