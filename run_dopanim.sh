# !/bin/bash 

set -e
python Train_dopanim.py --data_path ./dopanim --project_name divide_baseline_dopanim \
  --dataset dopanim --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 \
  --cosine --num_class 15 --lambda_u 50 --gpuid 1 --annotator mv_label --noise_file dopanim_worst-3.json --wandb

# python Train_dopanim.py --data_path ./dopanim --project_name divide_baseline_dopanim \
#   --dataset dopanim --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 \
#   --cosine --num_class 15 --lambda_u 150 --gpuid 1 --annotator mv_label --noise_file dopanim_worst-3.json --wandb

# python Train_dopanim.py --data_path ./dopanim --project_name divide_baseline_dopanim \
#   --dataset dopanim --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 \
#   --cosine --num_class 15 --lambda_u 0 --gpuid 1 --annotator mv_label --noise_file dopanim_worst-3.json --wandb



# python Train_dopanim.py --data_path ./dopanim --project_name divide_baseline_dopanim \
#   --dataset dopanim --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 \
#   --cosine --num_class 15 --lambda_u 50 --gpuid 1 --annotator mv_label --noise_file dopanim_worst-4.json --wandb

# python Train_dopanim.py --data_path ./dopanim --project_name divide_baseline_dopanim \
#   --dataset dopanim --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 \
#   --cosine --num_class 15 --lambda_u 50 --gpuid 1 --annotator mv_label --noise_file dopanim_rand-5.json --wandb

# python Train_dopanim.py --data_path ./dopanim --project_name divide_baseline_dopanim \
#   --dataset dopanim --model dino --num_epochs 50 --batch_size 64 --lr 0.001 --warm_up_epochs 1 \
#   --cosine --num_class 15 --lambda_u 50 --gpuid 1 --annotator mv_label --noise_file dopanim_rand-1.json --wandb












