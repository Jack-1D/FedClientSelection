import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.nn import functional as F
from loss import client_count_cs_score, normalized_shannon_entropy

class FLClient:
    def __init__(self, model_type, data_loader, num_class, num_clients):
        self.model_type = model_type
        self.num_class = num_class
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_type.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None).to(self.device)
        self.global_model_replica = self.model
        # dataLoader of each round
        self.data_loader = data_loader
        # 紀錄從前一次signal到目前的資料累積
        self.data_size_from_last_signal = 0
        # 紀錄從頭到目前每個class的資料累積
        self.label_size_to_cur = [0 for _ in range(num_class)]
        # 紀錄從前一次signal到目前的各class的資料累積
        self.label_size_from_last_signal = [0 for _ in range(num_class)]
        self.all_data_size_from_last_signal = [0 for _ in range(num_clients)]
        self.data_size_from_last_signal_rank = [0 for _ in range(num_clients)]
        self.prev_data_size_from_last_signal_rank = [0 for _ in range(num_clients)]
        # 紀錄從頭到前一次signal每個class的資料累積
        self.label_size_to_last_signal = [0 for _ in range(num_class)]
        self._cs = 0.0

    @property
    def cs(self):
        return self._cs

    def client_update(self, round, epochs=1, lr=0.01):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for data, target in self.data_loader[round]:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def receive_model(self, global_model_state_dict):
        self.global_model_replica.load_state_dict(global_model_state_dict)
        self.model = copy.deepcopy(self.global_model_replica)
        # self.model.load_state_dict(global_model_state_dict)

    def compute_cs_score(self, round):
        self._cs = client_count_cs_score(self.model_type, self.global_model_replica, self.model.state_dict(), self.data_loader[round])

    def compute_nse(self):
        self.nse = normalized_shannon_entropy(self.label_size_to_cur)

    def send_data_size_from_last_signal(self, Server, client_idx):
        Server.receive_data_size_from_last_signal(self.data_size_from_last_signal, client_idx)

    def receive_all_data_size_from_last_signal(self, all_data_size_from_last_signal):
        self.all_data_size_from_last_signal = all_data_size_from_last_signal

    def compute_label_size_from_last_signal_rank(self):
        self.data_size_from_last_signal_proportions = self.data_size_from_last_signal / np.sum(self.all_data_size_from_last_signal) if np.sum(self.all_data_size_from_last_signal) != 0 else 0
        # 紀錄前一輪每個client data size的排名
        self.prev_data_size_from_last_signal_rank = copy.deepcopy(self.data_size_from_last_signal_rank)
        # 更新每個client的資料量排名
        self.data_size_from_last_signal_rank = np.array([np.sum(self.all_data_size_from_last_signal[i]) for i in range(self.num_clients)]).argsort().argsort()
    
    def check_cs_signal(self, cs_threshold):
        if self._cs < cs_threshold:
            return True
        return False
    
    def check_data_size_rank_change_siganl(self, round, last_signal):
        if round > last_signal + 1 and not np.array_equal(self.prev_data_size_from_last_signal_rank, self.data_size_from_last_signal_rank):
            return True
        return False

    def check_local_iid_signal(self, client_idx, kl_threshold, kl_epsilon):
        # 若還沒signal過，就先用local iid程度來替代
        if np.sum(self.label_size_to_cur) == 0 or np.sum(self.label_size_to_last_signal) == 0:
            self.kl = normalized_shannon_entropy(self.label_size_to_cur)
        else:
            P = torch.tensor([
                self.label_size_to_cur[i] / np.sum(self.label_size_to_cur) if np.sum(self.label_size_to_cur) != 0 else 0 
                for i in range(self.num_class)
            ]) + kl_epsilon
            Q = torch.tensor([
                self.label_size_to_last_signal[i] / np.sum(self.label_size_to_last_signal) if np.sum(self.label_size_to_last_signal) != 0 else 0 
                for i in range(self.num_class)
            ]) + kl_epsilon
            self.kl = F.kl_div(P.log(), Q, reduction='batchmean')
            print(f"P: {P}")
            print(f"Q: {Q}")
        print(f"Client {client_idx} KL Divergence: {self.kl.item():.4f}")
        if self.kl.item() > kl_threshold:
            return True
        return False
        

    def check_signal(self, client_idx, selected_clients, cs_threshold, round, last_signal, kl_threshold, kl_epsilon=1e-10):
        if (client_idx in selected_clients and self.check_cs_signal(cs_threshold)) or self.check_data_size_rank_change_siganl(round, last_signal) or self.check_local_iid_signal(client_idx, kl_threshold, kl_epsilon):
            print(f"Client {client_idx}", f"signal: cs={self._cs:.4f}" if (client_idx in selected_clients and self.check_cs_signal(cs_threshold)) else "", 
                  f"prev_data_size_rank={self.prev_data_size_from_last_signal_rank}, data_size_rank={self.data_size_from_last_signal_rank}" if self.check_data_size_rank_change_siganl(round, last_signal) else "", 
                  f"local_kl={self.kl.item():.4f}" if self.check_local_iid_signal(client_idx, kl_threshold, kl_epsilon) else "")
            return True
        return False

    def response_server_request(self):
        return self._cs, self.data_size_from_last_signal_proportions, self.nse
    
    def do_snapshot(self):
        self.label_size_from_last_signal = [0 for _ in range(self.num_class)]
        self.label_size_to_last_signal = copy.deepcopy(self.label_size_to_cur)
        self.data_size_from_last_signal = 0