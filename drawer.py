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
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
    plt.grid()
    plt.tight_layout()
    os.makedirs('experiments_plot', exist_ok=True)
    plt.savefig('experiments_plot/label_consumption.png')

def draw_client_data_size(num_clients, num_rounds, list_of_data_sizes):
    """
    Draw each client's data size per round.
    :param num_clients: Number of clients
    :param num_rounds: Number of rounds
    :param list_of_data_sizes: List of data sizes for each round
    """
    plt.figure(figsize=(12, 6))
    for client_idx in range(num_clients):
        client_sizes = [list_of_data_sizes[round_idx][client_idx] for round_idx in range(num_rounds)]
        plt.plot(range(num_rounds), client_sizes, label=f'Client {client_idx}')
    plt.xlabel('Round')
    plt.ylabel('Data Size')
    plt.title('Client Data Size per Round')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5)
    plt.tight_layout()
    os.makedirs('experiments_plot', exist_ok=True)
    plt.savefig('experiments_plot/client_data_size_per_round.png')

def draw_cumulative_data_size(num_clients, num_rounds, list_of_data_sizes):
    """
    Draw cumulative data size for each client over rounds, marking the highest point for each line.
    :param num_clients: Number of clients
    :param num_rounds: Number of rounds
    :param list_of_data_sizes: List of data sizes for each round
    """
    cumulative_data_sizes = [[sum(list_of_data_sizes[r][client_idx] for r in range(round_idx + 1)) for client_idx in range(num_clients)] for round_idx in range(num_rounds)]
    plt.figure(figsize=(12, 6))
    for client_idx in range(num_clients):
        client_cumulative_sizes = [cumulative_data_sizes[round_idx][client_idx] for round_idx in range(num_rounds)]
        plt.plot(range(num_rounds), client_cumulative_sizes, label=f'Client {client_idx}')
        # Mark the highest point
        max_value = max(client_cumulative_sizes)
        max_index = client_cumulative_sizes.index(max_value)
        plt.text(max_index, max_value, f'{max_value:.2f}', fontsize=8, ha='center', va='bottom')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Data Size')
    plt.title('Cumulative Data Size per Client per Round')
    plt.legend()
    plt.tight_layout()
    os.makedirs('experiments_plot', exist_ok=True)
    plt.savefig('experiments_plot/cumulative_data_size_per_client.png')

def draw_client_selected_times(num_clients, client_selection_counts):
    """
    Draw the number of times each client was selected.
    :param num_clients: Number of clients
    :param client_selection_counts: List of counts for each client
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clients), client_selection_counts, color='skyblue')
    plt.title('Client Selection Counts')
    plt.xlabel('Client Index')
    plt.ylabel('Selection Count')
    plt.grid(axis='y')
    os.makedirs('result_plot', exist_ok=True)
    plt.savefig('result_plot/client_selection_counts.png')

def draw_accuracy(num_rounds, accuracies, dirichlet_alpha):
    """
    Draw the test accuracy over communication rounds.
    :param num_rounds: Number of rounds
    :param accuracies: List of accuracies for each round
    :param dirichlet_alpha: Dirichlet alpha value used in the experiment
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_rounds + 1), accuracies, marker='o', linestyle='-', color='b')
    plt.title(f'Test Accuracy vs. Communication Round test (Dirichlet α={dirichlet_alpha})')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    os.makedirs('result_plot', exist_ok=True)
    plt.savefig('result_plot/accuracy_vs_round_dirichlet_10.png')

def draw_zeta(num_rounds, all_zeta_per_round):
    """
    Draw the average zeta value over rounds.
    :param num_rounds: Number of rounds
    :param all_zeta_per_round: List of average zeta values per round
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_rounds), all_zeta_per_round, marker='o', color='purple')
    plt.xlabel('Round')
    plt.ylabel('Average Zeta Value')
    plt.title('Average Zeta Value over Rounds')
    plt.grid(True)
    os.makedirs('result_plot', exist_ok=True)
    plt.savefig('result_plot/zeta_per_round.png')

def draw_cs_mean(num_rounds, cs_mean_record):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_rounds + 1), cs_mean_record, marker='o')
    plt.xlabel('Round')
    plt.ylabel('Average CS')
    plt.title('Average CS per Round')
    plt.grid(True)
    os.makedirs('result_plot', exist_ok=True)
    plt.savefig('result_plot/average_cs_per_round.png')