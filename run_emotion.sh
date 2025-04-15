#!/bin/bash 

set -e 

# # Twitter.
# python Train_emotion.py --data_path ./LDL/Twitter_LDL --project_name Dividemix_Random_Twitter --dataset Twitter --model dino --gpuid 1 --num_epochs 50 --batch_size 64 --lr 0.001 \
#     --warm_up_epochs 1 --cosine --lambda_u 0 --num_class 8 --annotator three_annotators --wandb

# Flickr.
python Train_emotion.py --data_path ./LDL/Flickr_LDL --project_name Dividemix_Random_Flickr --dataset Flickr --model dino --gpuid 0 --num_epochs 50 --batch_size 64 --lr 0.001 \
    --warm_up_epochs 1 --cosine --lambda_u 0 --num_class 8 --annotator three_annotators #--wandb