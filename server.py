import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.nn import functional as F
import numpy as np

class FLServer:
    def __init__(self, model_type, list_of_testloaders, total_class, num_clients, num_rounds, lr, momentum=0.9, weight_decay=5e-4):
        self.list_of_testloaders = list_of_testloaders
        self.total_class = total_class
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_type
        self.all_data_size_from_last_signal = [[0 for _ in range(total_class)] for _ in range(num_clients)]
        self.signal_list = [False for _ in range(num_clients)]
        self.clients_cs_score = [0.0 for _ in range(num_clients)]
        self.clients_label_size_from_last_signal_proportions = [0.0 for _ in range(num_clients)]
        self.clients_nse = [0.0 for _ in range(num_clients)]
        self.score = [0.0 for _ in range(num_clients)]
        
        # 保存初始參數用於恢復訓練
        self.num_rounds = num_rounds
        self.initial_lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
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
        
    def recompute_probabilities(self, zeta, temperature, list_of_dataLoaders, round):
        print(self.clients_cs_score)
        print([len(list_of_dataLoaders[i][round].dataset) / np.sum([len(list_of_dataLoaders[j][round].dataset) for j in range(self.num_clients)]) for i in range(self.num_clients)])
        self.score = [(zeta * (self.clients_cs_score[i] + 1) / 2 + (1-zeta) * len(list_of_dataLoaders[i][round].dataset) / np.sum([len(list_of_dataLoaders[j][round].dataset) for j in range(self.num_clients)])) for i in range(self.num_clients)]
        scaled_score = torch.tensor(self.score) / temperature
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
    
    def resume_training_from_round(self, resume_round):
        """
        從指定的輪數恢復訓練，調整學習率調度器狀態
        """
        if resume_round > 0:
            print(f"Adjusting learning rate scheduler for resume from round {resume_round}")
            
            # 重新創建調度器並快進到指定輪數
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr, 
                                     momentum=self.momentum, weight_decay=self.weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_rounds)
            
            # 快進調度器到指定輪數
            for _ in range(resume_round):
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate adjusted to: {current_lr:.6f} for round {resume_round}")