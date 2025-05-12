import torch
import torch.nn as nn
import os
from torch.nn import functional as F

class FLServer:
    def __init__(self, model_type, total_class, num_clients):
        self.total_class = total_class
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_type.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None).to(self.device)
        self.all_data_size_from_last_signal = [[0 for _ in range(total_class)] for _ in range(num_clients)]
        self.signal_list = [False for _ in range(num_clients)]
        self.clients_cs_score = [0.0 for _ in range(num_clients)]
        self.clients_label_size_from_last_signal_proportions = [0.0 for _ in range(num_clients)]
        self.clients_nse = [0.0 for _ in range(num_clients)]

    def server_aggregate(self, Clients, selected_clients, client_weights):
        self.global_dict = self.model.state_dict()
        for key in self.global_dict.keys():
            self.global_dict[key] = torch.zeros_like(self.global_dict[key])
            for client_idx in selected_clients:
                self.global_dict[key] += client_weights[client_idx] * Clients[client_idx].model.state_dict()[key]
        self.model.load_state_dict(self.global_dict)

    def send_model(self, Clients, selected_clients):
        for client_idx in selected_clients:
            Clients[client_idx].receive_model(self.model.state_dict())

    def test_model(self, testloader):
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
        score = [alpha * self.clients_cs_score[i] + beta * self.clients_label_size_from_last_signal_proportions[i] + gamma * self.clients_nse[i] for i in range(self.num_clients)]
        scaled_score = torch.tensor(score) / temperature
        probabilities = F.softmax(scaled_score, dim=0).numpy()
        return probabilities
    
    def do_snapshot(self, Clients):
        for client_idx in range(len(Clients)):
            Clients[client_idx].do_snapshot()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, model_class, path, device):
        self.model = model_class().to(device)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=device))
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")