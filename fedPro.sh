#!/bin/bash
# filepath: /home/hscc/jack/FedClientSelection/fedPro.sh

nohup python3 -u main.py \
    --dataset CIFAR100 \
    --method proposed \
    --model_type ResNet18 \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 2.0 \
    --num_clients 10 \
    --num_rounds 600 \
    --batch_size 128 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 100 \
    --data_size_alphas "[3.7, 8.2, 10.0, 11.0, 3.3, 6.6, 5.5, 7.4, 4.2, 3.1]" \
    --start_class_num 10 \
    --increment_period 5 \
    --epochs_per_client 2 \
    --learning_rate 1 \
    --participate_ratio 0.8 \
    --random_seed 42 \
    --alpha 0.6666666 \
    --beta 0.5 \
    --softmax_temperature 0.08 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file Log10.log \
    --momentum 0 \
    --weight_decay 0 \
    > fedPro.log 2>&1 &