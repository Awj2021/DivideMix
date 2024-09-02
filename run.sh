# !/bin/bash 

set -e
cd /mnt/fast/nobackup/scratch4weeks/wa00433/projects/DivideMix
python=/mnt/fast/nobackup/users/wa00433/miniconda3/envs/divide1/bin/python
num_epochs=$1
$python Train_cifar.py --data_path ./cifar-10-batches-py --project_name Regenerated_labels_cifar10_dividemix_two_annotators --dataset cifar10 --gpuid 0 --num_epochs $num_epochs --batch_size 64 --lr 0.02 --warm_up_epochs 10 --cosine --lambda_u 0 --wandb
# $python Train_cifar.py --data_path ./cifar-10-batches-py --project_name cifar10n_dividemix_test_lambda_u --dataset cifar10 --gpuid 0 --num_epochs 300 --batch_size $batch_size --lr 0.02 --warm_up_epochs 10 --wandb --lambda_u 1