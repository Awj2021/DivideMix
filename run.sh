# !/bin/bash 

set -e
cd /mnt/fast/nobackup/scratch4weeks/wa00433/projects/DivideMix
annotator=$1
python=/mnt/fast/nobackup/users/wa00433/miniconda3/envs/divide1/bin/python
$python Train_cifar.py --data_path ./cifar-10-batches-py --project_name cifar10_dividemix  --dataset cifar10 --gpuid 0 --batch_size 64 --lr 0.02 --annotator $annotator --wandb