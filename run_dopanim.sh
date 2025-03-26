#!/bin/bash 

set -e 

# python Train_dopanim.py --data_path ./dopanim --project_name Dividemix_Random_Dopanim --dataset dopanim --model resnet34 --gpuid 0 --num_epochs 100 --batch_size 32 --lr 0.001 \
#     	--warm_up_epochs 1 --cosine --lambda_u 0 --num_class 15 --annotator four_annotators --noise_file dopanim_worst-4.json --wandb

# train the dopanim with dino model.
# set the hyper-parameters as the dopanim paper shows.
python Train_dopanim.py --data_path ./dopanim --project_name Dividemix_Random_Dopanim --dataset dopanim --model dino --gpuid 1 --num_epochs 50 --batch_size 64 --lr 0.001 \
    --warm_up_epochs 5 --cosine --lambda_u 0 --num_class 15 --annotator two_annotators --noise_type worst --wandb
