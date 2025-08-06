#!/bin/bash
# filepath: /home/hscc/jack/FedClientSelection/fedPro.sh

nohup python3 -u main.py \
    --dataset TinyImageNet \
    --method OCS \
    --model_type ResNet34 \
    --distribution_shifting \
    --class_increment \
    --data_distribution_alpha 2.0 \
    --num_clients 30 \
    --num_rounds 600 \
    --batch_size 128 \
    --test_batch_size 100 \
    --new_distribution_weight 0.1 \
    --data_size_gain_ratio 0.1 \
    --new_data_size_distribution_weight 0.5 \
    --rounds_to_get_new_data 30 \
    --data_size_alphas "[3.7, 8.2, 10.0, 11.0, 3.3, 6.6, 5.5, 7.4, 4.2, 3.1, 
                 4.8, 9.1, 6.3, 7.7, 5.9, 8.5, 3.9, 6.2, 4.4, 7.0, 
                 5.1, 8.8, 6.0, 7.2, 4.6, 9.3, 5.7, 8.0, 6.8, 7.5]" \
    --start_class_num 20 \
    --increment_period 3 \
    --epochs_per_client 2 \
    --learning_rate 1 \
    --participate_ratio 0.1 \
    --random_seed 42 \
    --alpha 0.6666666 \
    --beta 0.6666666 \
    --softmax_temperature 0.0533 \
    --kl_temperature 4.0 \
    --cs_threshold 0.5 \
    --kl_threshold 0.01 \
    --kl_epsilon 1e-10 \
    --log_file Log10.log \
    --momentum 0 \
    --weight_decay 0 \
    > fedPro.log 2>&1 &