from server import FLServer
from client import FLClient
from data_preprocess import *
import copy
import logging
from set_seed import set_seed

def OCS(args, model_type):
    set_seed(args)
    # 準備數據
    list_of_dataLoaders, list_of_data_sizes, total_class, list_of_client_indices_num, \
    end_training_exclusive, round_idx_increment = get_training_data(
        trainset=args.dataset,
        distribution_shifting=args.distribution_shifting,
        class_increment=args.class_increment,
        data_distribution_alpha=args.data_distribution_alpha,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        new_distribution_weight=args.new_distribution_weight,
        data_size_gain_ratio=args.data_size_gain_ratio,
        new_data_size_distribution_weight=args.new_data_size_distribution_weight,
        rounds_to_get_new_data=args.rounds_to_get_new_data,
        data_size_alphas=args.data_size_alphas,
        start_class_num=args.start_class_num,
        increment_period=args.increment_period,
        random_seed=args.random_seed
    )
    print(list_of_client_indices_num)
    list_of_testloaders = get_test_data(
        testset=args.dataset,
        batch_size=args.test_batch_size,
        random_seed=args.random_seed,
        class_increment=args.class_increment,
        num_rounds=args.num_rounds,
        start_class_num=args.start_class_num,
        round_idx_increment=round_idx_increment)

    Server = FLServer(copy.deepcopy(model_type), list_of_testloaders, total_class, args.num_clients, args.num_rounds, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    list_of_dataLoaders = list(map(list, zip(*list_of_dataLoaders)))    # [client][round]
    Clients = [FLClient(copy.deepcopy(model_type), list_of_dataLoaders[i], total_class, args.num_clients, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay) for i in range(args.num_clients)]

    # 用於記錄每輪的準確率
    accuracies = []

    # 創建保存模型的目錄
    os.makedirs("checkpoints", exist_ok=True)

    participate_client_num = int(args.num_clients * args.participate_ratio)
    client_list = [i for i in range(args.num_clients)]
    probabilities = [1.0 / args.num_clients for _ in range(args.num_clients)]

    # 紀錄每個client被選到的次數
    client_selection_counts = [0 for _ in range(args.num_clients)]

    # 紀錄最後一次signal是第幾輪
    last_signal = 0

    # 聯邦學習訓練
    for round in range(args.num_rounds):
        # Print cumulative data size and cumulative data distribution for each client
        cumulative_data_sizes = [len(list_of_dataLoaders[i][round].dataset) for i in range(args.num_clients)]
        cumulative_data_distributions = [Clients[i].label_size_to_cur for i in range(args.num_clients)]
        iid_distribution = [1.0 / total_class for _ in range(total_class)]
        kl_divergences = [
            sum(
            p * np.log(p / q) if p > 0 else 0
            for p, q in zip(client_distribution, iid_distribution)
            )
            for client_distribution in cumulative_data_distributions
        ]
        print(f"Cumulative data sizes: {cumulative_data_sizes}")
        print(f"KL Divergences from IID: {kl_divergences}")

        for client_idx in range(args.num_clients):
            # 更新data size
            Clients[client_idx].data_size_from_last_signal += list_of_data_sizes[round][client_idx]
            # print(f"round: {round}, client: {client_idx}, defference: {list(set(list_of_dataLoaders[client_idx][round].dataset.indices) - set(list_of_dataLoaders[client_idx][last_signal].dataset.indices))}")
            Clients[client_idx].label_size_to_cur = [Clients[client_idx].label_size_to_cur[i] + list_of_client_indices_num[round][client_idx][i] for i in range(total_class)]
            # 更新每個client新增的各label數量
            Clients[client_idx].label_size_from_last_signal = [Clients[client_idx].label_size_from_last_signal[i] + list_of_client_indices_num[round][client_idx][i] for i in range(total_class)]
            print(f"label_size_from_last_signal: {Clients[client_idx].label_size_from_last_signal}")
            print(f"label_size_to_cur: {Clients[client_idx].label_size_to_cur}")
            # 更新nse
            Clients[client_idx].compute_nse()
        # 傳model給被選到的client
        Server.send_model(Clients)
        # 每個client進行local training
        for client_idx in range(args.num_clients):
            Clients[client_idx].client_update(round, epochs=args.epochs_per_client, lr=Server.get_lr())
            # 每個被選到的client計算CS Score
            Clients[client_idx].compute_cs_score(round)
        print(f"CS Scores: {[f'{Clients[i].cs:.4f}' for i in range(args.num_clients)]}")
        # 更新每個client的資料量佔比
        for client_idx in range(args.num_clients):
            Clients[client_idx].send_data_size_from_last_signal(Server, client_idx)
        Server.send_all_data_size_from_last_signal(Clients)
        for client_idx in range(args.num_clients):
            Clients[client_idx].compute_label_size_from_last_signal_rank()
        print(f"label_size_from_last_signal_proportions: {[Clients[i].data_size_from_last_signal_proportions for i in range(args.num_clients)]}")
        print(f"clients_nse: {[Clients[i].nse for i in range(args.num_clients)]}")
        print(f"data_size_from_last_signal_rank: {Clients[0].data_size_from_last_signal_rank}")
        print(f"label_size_from_last_signal:{Clients[0].all_data_size_from_last_signal}")
        print([Clients[i].data_size_from_last_signal for i in range(args.num_clients)])
        print([np.sum(Clients[0].all_data_size_from_last_signal[i]) for i in range(args.num_clients)])
        # print([Clients[i].data_size_from_last_signal / np.sum([Clients[j].data_size_from_last_signal for j in selected_clients]) if i in selected_clients and np.sum([Clients[j].data_size_from_last_signal for j in selected_clients]) != 0 else 1.0 / len(selected_clients) for i in range(args.num_clients)])

        # === 預備：計算 update vector & norm ===
        updates = []
        weights = [
            len(Clients[i].data_loader[round].dataset) /
            sum([len(Clients[j].data_loader[round].dataset) for j in range(args.num_clients)])
            for i in range(args.num_clients)
        ]

        for client in Clients:
            update = client.get_update_norm()  # 只算更新向量，不做實際 training
            updates.append(update)

        norms = [np.linalg.norm(u) for u in updates]
        weighted_norms = np.array([w * n for w, n in zip(weights, norms)])

        # === Algorithm 1 (OCS) 核心部分 ===
        n = len(weighted_norms)
        sorted_indices = np.argsort(weighted_norms)
        sorted_norms = weighted_norms[sorted_indices]

        m = min(participate_client_num, 8)  # 限制最多選擇 8 個 client
        l = 0
        for l in range(n):
            if sorted_norms[l] == 0:
                continue
            threshold = np.sum(sorted_norms[:l + 1]) / sorted_norms[l]
            if m + l - n <= threshold:
                break

        # 用排序後的 index 找出 threshold 對應的 weighted norm 值
        threshold_idx = sorted_indices[l]
        threshold_value = weighted_norms[threshold_idx]

        # 計算 sampling 機率 p_i
        p = np.zeros(n)
        denom = np.sum(sorted_norms[:l + 1])
        for i in range(n):
            if weighted_norms[i] >= threshold_value:
                p[i] = 1.0
            else:
                p[i] = (m + l - n) * weighted_norms[i] / denom if denom > 0 else 0.0
                p[i] = min(p[i], 1.0)  # 安全剪裁，避免浮點誤差導致 p_i > 1

        # Bernoulli sampling：決定哪些 client 被選中
        selected_clients = np.array([i for i in range(n) if np.random.rand() < p[i]])
        if len(selected_clients) > 8:  # 如果選中的 client 超過 8 個，隨機選擇 8 個
            selected_clients = np.random.choice(selected_clients, 8, replace=False)
        print(f"Selected clients (OCS) for round {round + 1}: {selected_clients}")

        # 記錄每一輪選到的client
        for client_idx in selected_clients:
            client_selection_counts[client_idx] += 1

        weights = [1.0 / len(selected_clients) for _ in range(args.num_clients)]
        Server.server_aggregate(Clients, selected_clients, weights)
        # Server.server_aggregate(Clients, selected_clients, [Clients[i].data_size_from_last_signal / np.sum([Clients[j].data_size_from_last_signal for j in selected_clients]) if i in selected_clients and np.sum([Clients[j].data_size_from_last_signal for j in selected_clients]) != 0 else 1.0 / len(selected_clients) for i in range(args.num_clients)])

        accuracy, loss = Server.test_model(list_of_testloaders[round])
        accuracies.append(accuracy)
        print(f"Round {round + 1}/{end_training_exclusive}, Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        if round < end_training_exclusive:
            logging.info(f"Round {round + 1}/{end_training_exclusive}, Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        else:
            logging.info(f"Round {round + 1}/{args.num_rounds}, Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        if round % 10 == 0:
            cur_model_path = f"checkpoints/global_model_{round+1}.pth"
            Server.save_model(cur_model_path)

        

    # 保存最終模型
    final_model_path = "checkpoints/global_model_final.pth"
    Server.save_model(final_model_path)

    draw_client_selected_times(args.num_clients, client_selection_counts)
    draw_accuracy(args.num_rounds, accuracies, args.data_distribution_alpha)

    # 示例：加載最終模型並測試
    Server.load_model(final_model_path)
    accuracy, loss = Server.test_model(list_of_testloaders[-1])
    print(f"Loaded Model - Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
