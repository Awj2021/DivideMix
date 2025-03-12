#!/bin/bash 

set -e 

python Train_dopanim.py --data_path ./dopanim --project_name Dopanim_Dividemix_Multi_Annotators --dataset dopanim --gpuid 0 --num_epochs 100 --batch_size 64 --lr 0.002 \
    --warm_up_epochs 1 --cosine --lambda_u 0 --num_class 15 --annotator two_annotators --wandb