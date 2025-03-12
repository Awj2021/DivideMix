#!/bin/bash 

set -e 

python Train_chaoyang.py --data_path ./chaoyang --project_name Chaoyang_Dividemix_Multi_Annotators --dataset chaoyang --gpuid 1 --num_epochs 100 --batch_size 64 --lr 0.002 \
    --warm_up_epochs 1 --cosine --lambda_u 0 --num_class 4 --annotator two_annotators --wandb
