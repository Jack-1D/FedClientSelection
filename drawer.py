import matplotlib.pyplot as plt
import numpy as np
import os

def draw_client_label_each_round(num_clients, num_rounds, trainset, list_of_client_indices_num):
    """
    Draw each client's data size per class for each round, combined into a single figure.
    :param num_clients: Number of clients
    :param num_rounds: Number of rounds
    :param trainset: Training dataset
    :param list_of_client_indices_num: List of client indices per round
    """
    plt.figure(figsize=(20, 15))
    for client_idx in range(num_clients):
        plt.subplot((num_clients + 2) // 3, 3, client_idx + 1)  # Arrange subplots in a grid
        for class_idx in range(len(trainset.classes)):
            plt.plot(
                range(num_rounds),
                [list_of_client_indices_num[round_idx][client_idx][class_idx] for round_idx in range(num_rounds)],
                label=f'Class {class_idx}'
            )
        plt.title(f'Client {client_idx}')
        plt.xlabel('Round')
        plt.ylabel('Data Size')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.grid()
    plt.tight_layout()
    os.makedirs('experiments_plot', exist_ok=True)
    plt.savefig('experiments_plot/all_clients_class_data_size_distribution_across_rounds.png')

def draw_data_consumption(num_rounds, list_of_data_sizes):
    """
    Draw total data consumption for each round.
    :param num_rounds: Number of rounds
    :param list_of_data_sizes: List of data sizes for each round
    """
    data_consumption = [sum(list_of_data_sizes[round_idx]) for round_idx in range(num_rounds)]
    plt.figure()
    plt.plot(range(num_rounds), data_consumption, marker='o')
    plt.title('Data Consumption Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Data Consumed')
    plt.grid()
    os.makedirs('experiments_plot', exist_ok=True)
    plt.savefig('experiments_plot/data_consumption.png')

def draw_label_consumption(num_clients, num_rounds, list_of_client_indices_num, trainset):
    """
    Draw label consumption for each class over rounds.
    :param num_clients: Number of clients
    :param num_rounds: Number of rounds
    :param list_of_client_indices_num: List of client indices per round
    :param trainset: Training dataset
    """
    label_consumption = np.zeros((num_rounds, len(trainset.classes)))
    for round_idx in range(num_rounds):
        for client_idx in range(num_clients):
            for class_idx in range(len(trainset.classes)):
                label_consumption[round_idx][class_idx] += list_of_client_indices_num[round_idx][client_idx][class_idx]
    plt.figure()
    for class_idx in range(len(trainset.classes)):
        plt.plot(range(num_rounds), label_consumption[:, class_idx], label=f'Class {class_idx}')
    plt.title('Label Consumption Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Data Consumed')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.grid()
    os.makedirs('experiments_plot', exist_ok=True)
    plt.savefig('experiments_plot/label_consumption.png')