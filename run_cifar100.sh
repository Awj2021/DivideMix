# !/bin/bash 
set -e 

python Train_cifar.py --data_path ./cifar-100-python --project_name Baseline_Cifar100_IDN50 \
  --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
  --noise_file Simulated_Human.pt --num_class 100 --lambda_u 50 --annotator mv_label --wandb 