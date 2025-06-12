#!/bin/bash
# filepath: /home/hscc/jack/FedClientSelection/fedPro.sh

nohup python3.10 -u fedPro.py \
    --dataset CIFAR10 \
    --model_type CNN \
    --num_clients 10 \
    --num_rounds 400 \
    --batch_size 64 \
    --test_batch_size 100 \
    --epochs_per_client 2 \
    --learning_rate 0.01 \
    --participate_ratio 0.8 \
    --random_seed 42 \
    --log_file Log10.log \
    > fedPro.log 2>&1 &