#!/bin/bash

run_experiment() {
    local log_prefix=$1
    shift # 剩餘的參數都會傳給 python
    
    echo "開始實驗: $log_prefix"
    python3.10 -u main.py "$@" > "${log_prefix}_err.log" 2>&1
    echo "完成實驗: $log_prefix"
}

# 實驗 1
run_experiment "random_42" --dataset CIFAR10 \
    --method fedDynamic \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file random_42.log \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --random 42

run_experiment "random_43" --dataset CIFAR10 \
    --method fedDynamic \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file random_43.log \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --random 43

run_experiment "random_44" --dataset CIFAR10 \
    --method fedDynamic \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file random_44.log \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --random 44

run_experiment "random_45" --dataset CIFAR10 \
    --method fedDynamic \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file random_45.log \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --random 45

run_experiment "random_46" --dataset CIFAR10 \
    --method fedDynamic \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file random_46.log \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --random 46

run_experiment "random_47" --dataset CIFAR10 \
    --method fedDynamic \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file random_47.log \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --random 47

run_experiment "PoC" --dataset CIFAR10 \
    --method PoC \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file PoC.log \
    --momentum 0.9 \
    --weight_decay 5e-4

run_experiment "OCS" --dataset CIFAR10 \
    --method OCS \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file OCS.log \
    --momentum 0.9 \
    --weight_decay 5e-4

run_experiment "PNCS" --dataset CIFAR10 \
    --method PNCS \
    --model_type CNN \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 0.3 \
    --num_clients 30 \
    --num_rounds 108 \
    --batch_size 64 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 50 \
    --data_size_alphas "[3.7, 13.2, 18.0, 20.0, 3.3, 1.8, 11.5, 16.4, 4.2, 3.1, 
                 4.8, 15.1, 1.9, 2.8, 9.9, 17.5, 3.9, 2.2, 4.4, 14.8, 
                 1.2, 17.3, 3.0, 7.2, 0.9, 12.3, 1.5, 11.0, 3.7, 2.3]" \
    --start_class_num 5 \
    --increment_period 15 \
    --increment_class_num 1 \
    --epochs_per_client 3 \
    --learning_rate 0.1 \
    --participate_ratio 0.2 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file PNCS.log \
    --momentum 0.9 \
    --weight_decay 5e-4