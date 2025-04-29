# !/bin/bash 

set -e
# Flickr
# python Train_emotion.py --data_path ./LDL/Flickr_LDL --project_name dopanim_baseline_flickr --dataset flickr \
#     --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 150 \
#     --gpuid 1 --annotator majority_vote_1_2 --wandb

# python Train_emotion.py --data_path ./LDL/Flickr_LDL --project_name dopanim_baseline_flickr --dataset flickr \
#     --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 150 \
#     --gpuid 1 --annotator majority_vote_1_3 --wandb

# python Train_emotion.py --data_path ./LDL/Flickr_LDL --project_name dopanim_baseline_flickr --dataset flickr \
#     --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 150 \
#     --gpuid 1 --annotator majority_vote_1_4 --wandb

# python Train_emotion.py --data_path ./LDL/Flickr_LDL --project_name dopanim_baseline_flickr --dataset flickr \
#     --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 150 \
#     --gpuid 1 --annotator majority_vote_1_5 --wandb

# Twitter
python Train_emotion.py --data_path ./LDL/Twitter_LDL --project_name divide_baseline_twitter --dataset twitter \
    --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 50 \
    --gpuid 0 --annotator majority_vote_1_3 --wandb


python Train_emotion.py --data_path ./LDL/Twitter_LDL --project_name divide_baseline_twitter --dataset twitter \
    --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 50 \
    --gpuid 0 --annotator majority_vote_1_2 --wandb

python Train_emotion.py --data_path ./LDL/Twitter_LDL --project_name divide_baseline_twitter --dataset twitter \
    --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 50 \
    --gpuid 0 --annotator majority_vote_1_4 --wandb

python Train_emotion.py --data_path ./LDL/Twitter_LDL --project_name divide_baseline_twitter --dataset twitter \
    --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 --cosine --num_class 8 --lambda_u 50 \
    --gpuid 0 --annotator majority_vote_1_5 --wandb

