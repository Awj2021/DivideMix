# !/bin/bash 

set -e 

CUDA_VISIBLE_DEVICES=1 python Train_cifar.py --data_path ./cifar-10-batches-py --project_name cifar10_dividemix --r 0.5 --dataset cifar10 --gpuid 0