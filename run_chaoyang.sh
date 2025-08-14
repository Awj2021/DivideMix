#!/bin/bash 

set -e 

# python Train_chaoyang.py --data_path ./chaoyang --project_name DivideMix_Random_Chaoyang --dataset chaoyang --gpuid 1 --num_epochs 100 --batch_size 64 --lr 0.01 \
#     --warm_up_epochs 10 --cosine --lambda_u 25 --num_class 4 --annotator three_annotators --dropout_rate 0.5 --wandb

python Train_chaoyang.py --data_path ./chaoyang --project_name DivideMix_Random_Chaoyang --dataset chaoyang --gpuid 0 --num_epochs 100 --batch_size 64 --lr 0.01 \
    --warm_up_epochs 10 --cosine --lambda_u 150 --num_class 4 --annotator two_annotators --dropout_rate 0.5 --resume --wandb

# python Train_chaoyang.py --data_path ./chaoyang --project_name DivideMix_Random_Chaoyang --dataset chaoyang --gpuid 1 --num_epochs 100 --batch_size 64 --lr 0.002 \
#     --warm_up_epochs 5 --cosine --lambda_u 25 --num_class 4 --annotator three_annotators --wandb
