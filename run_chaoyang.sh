# !/bin/bash 

set -e
# cd /mnt/fast/nobackup/scratch4weeks/wa00433/projects/DivideMix
# python=/mnt/fast/nobackup/users/wa00433/miniconda3/envs/divide1/bin/python

# T_max=$1
# $python Train_cifar_lr_scheduler_setting.py --data_path ./cifar-10-batches-py --project_name cifar10n_dividemix_single_annotator_rethink_setting --dataset cifar10 --gpuid 0 --batch_size 128 --lr 0.02 --T_max $T_max --wandb
# FIXME: Cifar100n; Please Change the code of dataloader_cifar.py to load the cifar100n dataset;
# $python Train_cifar.py --data_path ./cifar-100-python --project_name cifar100n_dividemix_baseline --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine --lambda_u 150 --noise_file CIFAR-100_human.pt --num_class 100 --wandb

# Cifar100 IDN Setting.
# annotator=$1
lambda_u=$1
annotator=$2
# $python Train_cifar.py --data_path ./cifar-100-python --project_name cifar100_IDN50_dividemix_cleanlab --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine --noise_file Simulated_Human.pt --num_class 100 --lambda_u $lambda_u --annotator $annotator --wandb 
CUDA_VISIBLE_DEVICES=0 python Train_chaoyang.py --data_path ./chaoyang --project_name chaoyang_baseline_single_label --dataset chaoyang --num_epochs 100 --batch_size 32 --lr 0.002 --warm_up_epochs 1 --cosine --num_class 4 --lambda_u 0 --annotator label_mv --wandb
