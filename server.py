import torch
import copy
import torch.nn as nn


class FLServer:
    def __init__(self, model_type, num_class, num_clients):
        self.num_class = num_class
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_type.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None).to(self.device)
        self.all_label_size_from_last_signal = [[0 for _ in range(num_class)] for _ in range(num_clients)]

    def server_aggregate(self, client_models_state_dict, selected_clients, client_weights):
        self.global_dict = self.model.state_dict()
        for key in self.global_dict.keys():
            self.global_dict[key] = torch.zeros_like(self.global_dict[key])
            for client_idx in selected_clients:
                self.global_dict[key] += client_weights[client_idx] * client_models_state_dict[client_idx][key]
        self.model.load_state_dict(self.global_dict)
        return self.model

    def send_model(self, clients, selected_clients):
        for client_idx in selected_clients:
            clients[client_idx].receive_model(self.model.state_dict())

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
    
    def receive_label_size_from_last_signal(self, label_size_from_last_signal, client_idx):
        self.all_label_size_from_last_signal[client_idx] = label_size_from_last_signal

    def send_all_label_size_from_last_signal(self, Clients):
        for client_idx in range(len(Clients)):
            Clients[client_idx].receive_all_label_size_from_last_signal(self.all_label_size_from_last_signal)