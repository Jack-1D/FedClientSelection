import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.nn import functional as F
from loss import client_count_cs_score, normalized_shannon_entropy

class FLClient:
    def __init__(self, model_type, data_loader, total_class, num_clients, lr=0.01, momentum=0.9, weight_decay=5e-4):
        self.model_type = model_type
        self.total_class = total_class
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_type
        self.global_model_replica = self.model
        # dataLoader of each round
        self.data_loader = data_loader
        # 紀錄從前一次signal到目前的資料累積
        self.data_size_from_last_signal = 0
        # 紀錄從頭到目前每個class的資料累積
        self.label_size_to_cur = [0 for _ in range(total_class)]
        # 紀錄從前一次signal到目前的各class的資料累積
        self.label_size_from_last_signal = [0 for _ in range(total_class)]
        self.all_data_size_from_last_signal = [0 for _ in range(num_clients)]
        self.data_size_from_last_signal_rank = [0 for _ in range(num_clients)]
        self.prev_data_size_from_last_signal_rank = [0 for _ in range(num_clients)]
        # 紀錄從頭到前一次signal每個class的資料累積
        self.label_size_to_last_signal = [0 for _ in range(total_class)]
        self._cs = 0.0
        self.avg_train_loss = 0.0
        self.norm = 0.0
        self.gradient = []  # Initialize gradient storage
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    @property
    def cs(self):
        return self._cs

    def get_update_norm(self):
        return self.norm

    def get_gradient_summary(self):
        """
        Returns gradient summary as a flattened 1D numpy array
        使用模型的當前參數結構確保所有客戶端返回相同維度的梯度向量
        """
        # 使用模型參數結構作為基準，確保所有客戶端返回相同長度的向量
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if not self.gradient:
            print(f"No gradient data available, returning zero vector of size {total_params}")
            return np.zeros(total_params)
        
        # Average all gradients across batches
        num_batches = len(self.gradient)
        if num_batches == 0:
            print(f"Empty gradient list, returning zero vector of size {total_params}")
            return np.zeros(total_params)
        
        print(f"Processing {num_batches} batches of gradients")
        
        # 根據模型參數建立梯度向量，而不是依賴存儲的梯度
        gradient_vector = np.zeros(total_params)
        
        if num_batches > 0 and len(self.gradient[0]) > 0:
            # 取最後一個 batch 的梯度
            last_gradients = self.gradient[-1]
            
            param_idx = 0
            start_idx = 0
            
            for param, grad in zip(self.model.parameters(), last_gradients):
                param_size = param.numel()
                end_idx = start_idx + param_size
                
                try:
                    # 確保梯度與參數形狀匹配
                    if grad is not None and grad.shape == param.shape:
                        flat_grad = grad.flatten().cpu().numpy()
                        # 檢查並清理 NaN/Inf 值
                        flat_grad = np.nan_to_num(flat_grad, nan=0.0, posinf=0.0, neginf=0.0)
                        gradient_vector[start_idx:end_idx] = flat_grad
                    else:
                        print(f"Gradient mismatch for parameter {param_idx}: expected {param.shape}, got {grad.shape if grad is not None else None}")
                        # 填充為零
                        gradient_vector[start_idx:end_idx] = 0.0
                        
                except Exception as e:
                    print(f"Error processing parameter {param_idx}: {e}")
                    gradient_vector[start_idx:end_idx] = 0.0
                
                start_idx = end_idx
                param_idx += 1
        
        print(f"Generated gradient vector with {len(gradient_vector)} parameters (expected: {total_params})")
        return gradient_vector

    def client_update(self, round, epochs=5, lr=0.01, beta=0.5, temperature=4.0):
        self.model = self.model.to(self.device)
        self.global_model_replica = self.global_model_replica.to(self.device)
        self.model.train()
        self.global_model_replica.eval()
        # 更新 learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        
        # Clear previous gradients
        self.gradient = []
        
        total_loss = 0.0
        total_samples = len(self.data_loader[round].dataset)
        for epoch in range(epochs):
            for data, target in self.data_loader[round]:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                if scaler:
                    with torch.amp.autocast("cuda"):
                        # Local model output
                        local_output = self.model(data)
                        # Global model output (teacher)
                        with torch.no_grad():
                            global_output = self.global_model_replica(data)
                        
                        # Cross-entropy loss
                        loss_ce = criterion_ce(local_output, target)
                        
                        # Knowledge distillation loss
                        local_logits_soft = F.log_softmax(local_output / temperature, dim=1)
                        global_logits_soft = F.softmax(global_output / temperature, dim=1)
                        loss_kd = criterion_kd(local_logits_soft, global_logits_soft) * (temperature ** 2)
                        
                        # Total loss
                        loss = 1.5 * (beta * loss_ce + (1 - beta) * loss_kd)
                    scaler.scale(loss).backward()
                    # Store gradients only in the last epoch
                    if epoch == epochs - 1:
                        self.gradient.append([
                            param.grad.clone().detach().cpu() 
                            for param in self.model.parameters() if param.grad is not None
                        ])
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Local model output
                    local_output = self.model(data)
                    # Global model output (teacher)
                    with torch.no_grad():
                        global_output = self.global_model_replica(data)
                    
                    # Cross-entropy loss
                    loss_ce = criterion_ce(local_output, target)
                    
                    # Knowledge distillation loss
                    local_logits_soft = F.log_softmax(local_output / temperature, dim=1)
                    global_logits_soft = F.softmax(global_output / temperature, dim=1)
                    loss_kd = criterion_kd(local_logits_soft, global_logits_soft) * (temperature ** 2)
                    
                    # Total loss
                    loss = 1.5 * (beta * loss_ce + (1 - beta) * loss_kd)
                    loss.backward()
                    # Store gradients only in the last epoch
                    if epoch == epochs - 1:
                        self.gradient.append([
                            param.grad.clone().detach().cpu() 
                            for param in self.model.parameters() if param.grad is not None
                        ])
                    self.optimizer.step()
                total_loss += loss.item() * data.size(0)

            del data, target, local_output, global_output, loss
            torch.cuda.empty_cache()
        self.model = self.model.to("cpu")
        self.global_model_replica = self.global_model_replica.to("cpu")
        torch.cuda.empty_cache()
        self.avg_train_loss = total_loss / total_samples if total_samples > 0 and not np.isnan(total_loss) else 0.0
        self.norm = np.linalg.norm(
            torch.cat([
            (param1 - param2).view(-1) 
            for param1, param2 in zip(self.global_model_replica.parameters(), self.model.parameters())
            ]).detach().cpu().numpy()
        )
        

    def receive_model(self, global_model_state_dict, global_optimizer_state_dict):
        self.global_model_replica.load_state_dict(global_model_state_dict)
        self.model = copy.deepcopy(self.global_model_replica)
        # 同步 optimizer 狀態，避免 momentum buffer 失效
        if global_optimizer_state_dict is not None:
            # 重新建立optimizer，避免GradScaler追蹤不到
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'], momentum=self.optimizer.param_groups[0]['momentum'], weight_decay=self.optimizer.param_groups[0]['weight_decay'])
            self.optimizer.load_state_dict(global_optimizer_state_dict)

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
                for i in range(self.total_class)
            ]) + kl_epsilon
            Q = torch.tensor([
                self.label_size_to_last_signal[i] / np.sum(self.label_size_to_last_signal) if np.sum(self.label_size_to_last_signal) != 0 else 0 
                for i in range(self.total_class)
            ]) + kl_epsilon
            self.kl = F.kl_div(P.log(), Q, reduction='batchmean')
            print(f"P: {P}")
            print(f"Q: {Q}")
        print(f"Client {client_idx} KL Divergence: {self.kl.item():.4f}")
        if self.kl.item() > kl_threshold:
            return True
        return False
        

    def check_signal(self, client_idx, round_idx_increment, cs_threshold, round, last_signal, kl_threshold, kl_epsilon=1e-10):
        # if round in round_idx_increment or self.check_cs_signal(cs_threshold) or self.check_data_size_rank_change_siganl(round, last_signal) or self.check_local_iid_signal(client_idx, kl_threshold, kl_epsilon):
        if round in round_idx_increment or self.check_cs_signal(cs_threshold) or self.check_data_size_rank_change_siganl(round, last_signal):
            print(f"Client {client_idx}, signal:", 
                  f"new_class_incoming" if (round+1) in round_idx_increment else "",
                  f"cs={self._cs:.4f}" if (self.check_cs_signal(cs_threshold)) else "", 
                  f"prev_data_size_rank={self.prev_data_size_from_last_signal_rank}, data_size_rank={self.data_size_from_last_signal_rank}" if self.check_data_size_rank_change_siganl(round, last_signal) else "", 
                #   f"local_kl={self.kl.item():.4f}" if self.check_local_iid_signal(client_idx, kl_threshold, kl_epsilon) else ""
                )
            return True
        return False

    def response_server_request(self):
        return self._cs, self.data_size_from_last_signal_proportions, self.nse
    
    def do_snapshot(self):
        self.label_size_from_last_signal = [0 for _ in range(self.total_class)]
        self.label_size_to_last_signal = copy.deepcopy(self.label_size_to_cur)
        self.data_size_from_last_signal = 0