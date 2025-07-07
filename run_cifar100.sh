# !/bin/bash 
set -e 

# baseline method:
# python Train_cifar.py --data_path ./cifar-100-python --project_name Conformal_Prediction \
#   --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
#   --noise_file cifar100_split_train_noise_50.pt --num_class 100 --lambda_u 50 --annotator random_label1 \
#   --mixmatch --cp_loss mse --wandb

# method 1: 
calibration_alphas=(0.1 0.3)
cp_weights=(0.5 0.7 0.9)
for calibration_alpha in ${calibration_alphas[@]}; do
  for cp_weight in ${cp_weights[@]}; do
    python Train_cifar.py --data_path ./cifar-100-python --project_name Conformal_Prediction_New \
      --dataset cifar100 --gpuid 1 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
      --noise_file cifar100_split_train_noise_50.pt --num_class 100 --lambda_u 50 --annotator random_label1 \
      --calibration_alpha $calibration_alpha --mixmatch --conformal_prediction --cp_weight $cp_weight --cp_loss mse --wandb
  done
done

################ method 2:
# python Train_cifar.py --data_path ./cifar-100-python --project_name Conformal_Prediction \
#   --dataset cifar100 --gpuid 1 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
#   --noise_file cifar100_split_train_noise_50.pt --num_class 100 --lambda_u 0.5 --annotator random_label1 \
#   --calibration_alpha 0.3 --mixmatch --conformal_prediction --cp_weight 0.5 --cp_loss ce --wandb
# after using the mixmatch, the training is not diverge.

# Do not use the mixmatch.
# python Train_cifar.py --data_path ./cifar-100-python --project_name Conformal_Prediction \
#   --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
#   --noise_file cifar100_split_train_noise_50.pt --num_class 100 --lambda_u 0.5 --annotator random_label1 \
#   --calibration_alpha 0.3 --conformal_prediction --cp_weight 0.5 --cp_loss ce --wandb


  # python Train_cifar.py --data_path ./cifar-100-python --project_name Baseline_Conformal_Prediction \
  # --dataset cifar100 --gpuid 0 --num_epochs 300 --batch_size 64 --lr 0.02 --warm_up_epochs 30 --cosine \
  # --noise_file cifar100_split_train_noise_50.pt --num_class 100 --lambda_u 50 --annotator random_label1 \
  # --calibration_alpha 0.1 --mixmatch --wandb