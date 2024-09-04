# !/bin/bash 

set -e
cd /mnt/fast/nobackup/scratch4weeks/wa00433/projects/DivideMix
python=/mnt/fast/nobackup/users/wa00433/miniconda3/envs/divide1/bin/python

# T_max=$1
# $python Train_cifar_lr_scheduler_setting.py --data_path ./cifar-10-batches-py --project_name cifar10n_dividemix_single_annotator_rethink_setting --dataset cifar10 --gpuid 0 --batch_size 128 --lr 0.02 --T_max $T_max --wandb
annotator=$1
$python Train_cifar.py --data_path ./cifar-10-batches-py --project_name cifar10n_dividemix_baseline --dataset cifar10 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 10 --wandb --cosine --lambda_u 0 --annotator $annotator
# $python Train_cifar.py --data_path ./cifar-10-batches-py --project_name cifar10n_dividemix_test_lambda_u --dataset cifar10 --gpuid 0 --num_epochs 300 --batch_size $batch_size --lr 0.02 --warm_up_epochs 10 --wandb --lambda_u 1