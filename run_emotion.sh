# !/bin/bash 

set -e
# Flickr
python Train_emotion.py --data_path ./LDL/Flickr_LDL --project_name dopanim_baseline_flickr --dataset flickr \
    --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 50 \
    --gpuid 1 --annotator majority_vote_1_4 --wandb

# Twitter
# python Train_dopanim.py --data_path ./LDL/Twitter_LDL --project_name dopanim_baseline_twitter --dataset twitter \
#     --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 5 --cosine --num_class 8 --lambda_u 0 \
#     --gpuid 1 --annotator majority_vote_1_3 # --wandb
