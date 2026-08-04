#!/bin/bash
# filepath: /home/hscc/jack/FedClientSelection/fedPro.sh

nohup python3.10 -u main.py \
    --dataset CIFAR10 \
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
    --participate_ratio 0.3 \
    --random_seed 42 \
    --zeta 1.0 \
    --beta 0.6666666 \
    --mu 0.5 \
    --softmax_temperature 0.02 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file Log10.log \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --random 51 \
    > fedPro.log 2>&1 &