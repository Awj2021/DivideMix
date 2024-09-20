# !/bin/bash 
# Test the trained dividemix model on CIFAR-10/100.
set -e
cd /mnt/fast/nobackup/scratch4weeks/wa00433/projects/DivideMix
python=/mnt/fast/nobackup/users/wa00433/miniconda3/envs/divide1/bin/python
# Cifar100 IDN Setting.
$python Test_cifar.py --dataset cifar100 --data_path ./cifar-100-python --gpuid 0 --batch_size 64 --num_class 100 \
--pretrained_model ./cifar100_IDN50_dividemix_baseline/cifar100_resnet18_64_random_label1_lambda_u_150.0_300_last.pth ./cifar100_IDN50_dividemix_baseline/cifar100_resnet18_64_random_label2_lambda_u_150.0_300_last.pth