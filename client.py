import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from loss import client_count_cs_score, normalized_shannon_entropy

class FLClient:
    def __init__(self, model_type, data_loader, num_class, num_clients):
        self.num_class = num_class
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_type.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None).to(self.device)
        self.global_model_replica = self.model
        # dataLoader of each round
        self.data_loader = data_loader
        self.data_size_from_last_signal = 0
        self.label_size_to_cur = [0 for _ in range(num_class)]
        self.label_size_from_last_signal = [0 for _ in range(num_class)]
        self.all_label_size_from_last_signal = [[0 for _ in range(num_class)] for _ in range(num_clients)]
        self.data_size_from_last_signal_rank = [0 for _ in range(num_clients)]
        self.prev_data_size_from_last_signal_rank = [0 for _ in range(num_clients)]
        self._cs = 0.0

    @property
    def cs(self):
        return self._cs

    def client_update(self, loader, epochs=1, lr=0.01):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
        return self.model.state_dict()

    def receive_model(self, global_model_state_dict):
        self.model.load_state_dict(global_model_state_dict)
        self.global_model_replica.load_state_dict(global_model_state_dict)

    def compute_cs_score(self, round):
        self.cs = client_count_cs_score(self.model, self.global_model_replica, self.model.state_dict(), self.data_loader[round])

    def compute_nse(self):
        self.nse = normalized_shannon_entropy(self.label_size_to_cur)

    def send_label_size_from_last_signal(self, Server, client_idx):
        Server.receive_label_size_from_last_signal(self.label_size_from_last_signal, client_idx)

    def receive_all_label_size_from_last_signal(self, all_label_size_from_last_signal):
        self.all_label_size_from_last_signal = all_label_size_from_last_signal

    def compute_label_size_from_last_signal_rank(self):
        total_label_size_from_last_signal_proportions = np.sum([np.sum(self.all_label_size_from_last_signal[client_idx]) for client_idx in range(self.num_clients)])
        self.label_size_from_last_signal_proportions = [np.sum(self.all_label_size_from_last_signal[client_idx]) / total_label_size_from_last_signal_proportions if total_label_size_from_last_signal_proportions != 0 else 0 for client_idx in range(self.num_clients)]
        self.prev_data_size_from_last_signal_rank = copy.deepcopy(self.data_size_from_last_signal_rank)
        # 更新每個client的資料量排名
        self.data_size_from_last_signal_rank = np.array([np.sum(self.all_label_size_from_last_signal[i]) for i in range(self.num_clients)]).argsort().argsort()