# !/bin/bash 

set -e 

# CUDA_VISIBLE_DEVICES=0 python Train_cifar.py --data_path ./cifar-10-batches-py --project_name cifar10_dividemix --r 0.5 --dataset cifar10 --gpuid 0 --batch_size 256 --lr 0.02

CUDA_VISIBLE_DEVICES=0 python Train_cifar.py --data_path ./cifar-10-batches-py --project_name cifar10_dividemix_baseline --r 0.5 --dataset cifar10 --gpuid 0 --batch_size 128 --lr 0.02

# Test the leanring rate with big batch size.
# CUDA_VISIBLE_DEVICES=1 python Train_cifar.py --data_path ./cifar-10-batches-py --project_name cifar10_dividemix --r 0.5 --dataset cifar10 --gpuid 0 --batch_size 1024 --lr 0.005  