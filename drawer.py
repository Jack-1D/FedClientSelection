import matplotlib.pyplot as plt
import numpy as np
import os

def draw_client_label_each_round(num_clients, num_rounds, trainset, list_of_client_indices_num):
    """
    Draw each client's data size per class for each round, separated by client.
    :param num_clients: Number of clients
    :param num_rounds: Number of rounds
    :param trainset: Training dataset
    :param list_of_client_indices_num: List of client indices per round
    """
    for client_idx in range(num_clients):
        plt.figure(figsize=(12, 8))
        for class_idx in range(len(trainset.classes)):
            plt.plot(
                range(num_rounds),
                [list_of_client_indices_num[round_idx][client_idx][class_idx] for round_idx in range(num_rounds)],
                label=f'Class {class_idx}'
            )
        plt.title(f'Client {client_idx} Data Size Per Class Across Rounds')
        plt.xlabel('Round')
        plt.ylabel('Data Size')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.grid()
        os.makedirs('experiments_plot/distribution', exist_ok=True)
        plt.savefig(f'experiments_plot/distribution/client_{client_idx}_class_data_size_distribution_across_rounds.png')
        plt.show()

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
    plt.show()

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
    plt.legend()
    plt.grid()
    os.makedirs('experiments_plot', exist_ok=True)
    plt.savefig('experiments_plot/label_consumption.png')
    plt.show()