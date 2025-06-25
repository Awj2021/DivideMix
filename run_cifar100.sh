# !/bin/bash 
set -e 

# python Train_cifar.py --data_path ./cifar-100-python --project_name Baseline_Conformal_Prediction \
#   --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
#   --noise_file cifar100_split_train_noise_50.pt --num_class 100 --lambda_u 50 --annotator random_label1 \
#   --calibration_alpha 0.1  --wandb

python Train_cifar.py --data_path ./cifar-100-python --project_name Baseline_Conformal_Prediction \
  --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
  --noise_file cifar100_split_train_noise_50.pt --num_class 100 --lambda_u 50 --annotator random_label1 \
  --calibration_alpha 0.3  --wandb

# python Train_cifar.py --data_path ./cifar-100-python --project_name Baseline_Conformal_Prediction \
# --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
# --noise_file cifar100_split_train_noise_50.pt --num_class 100 --lambda_u 50 --annotator random_label1 \
# --calibration_alpha 0.5  --wandb 