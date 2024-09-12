# !/bin/bash 
# Test the trained dividemix model on CIFAR-10/100.
set -e
cd /mnt/fast/nobackup/scratch4weeks/wa00433/projects/DivideMix
python=/mnt/fast/nobackup/users/wa00433/miniconda3/envs/divide1/bin/python
# Cifar100 IDN Setting.
# annotators=("majority_label1_2" "majority_label1_3" "majority_label2_3")
# models=("cifar100_resnet18_64_majority_label1_2_lambda_u_150.0_300_last.pth" "cifar100_resnet18_64_majority_label1_3_lambda_u_150.0_300_last.pth" "cifar100_resnet18_64_majority_label2_3_lambda_u_150.0_300_last.pth")
#/mnt/fast/nobackup/scratch4weeks/wa00433/projects/DivideMix/cifar100_IDN50_dividemix_baseline/cifar100_resnet18_64_majority_label1_2_lambda_u_150.0_300_last.pth

# annotators=("aggre_4_label" "aggre_5_label" "aggre_6_label")
# models=("cifar100_resnet18_64_aggre_4_label_lambda_u_150_300_last.pth" "cifar100_resnet18_64_aggre_5_label_lambda_u_150_300_last.pth" "cifar100_resnet18_64_aggre_6_label_lambda_u_150_300_last.pth")

annotators=('random_label1' 'random_label2' 'random_label3' 'random_label4' 'random_label5' 'random_label6')
# since the model is missing for random_label2, we will use the model for random_label3.
models=('cifar100_resnet18_64_random_label1_lambda_u_150.0_300_last.pth' 'cifar100_resnet18_64_random_label3_lambda_u_150.0_300_last.pth' 'cifar100_resnet18_64_random_label3_lambda_u_150.0_300_last.pth' 'cifar100_resnet18_64_random_label4_lambda_u_150_300_last.pth' 'cifar100_resnet18_64_random_label5_lambda_u_150_300_last.pth' 'cifar100_resnet18_64_random_label6_lambda_u_150_300_last.pth')
# Check if the number of annotators matches the number of models
if [ ${#annotators[@]} -ne ${#models[@]} ]; then
  echo "The number of annotators and models must be the same."
  exit 1
fi

for i in "${!annotators[@]}"; do
  annotator=${annotators[$i]}
  model=${models[$i]}
  
  echo "Running Test_cifar.py with annotator: $annotator and model: $model"
  
  $python Test_cifar.py --dataset cifar100 --data_path ./cifar-100-python --noise_file Simulated_Human.pt --gpuid 0 --batch_size 256 --num_class 100 --annotator "$annotator" \
        --pretrained_model "./cifar100_IDN50_dividemix_baseline/$model"
done

# $python Test_cifar.py --dataset cifar100 --data_path ./cifar-100-python --noise_file Simulated_Human.pt --gpuid 0 --batch_size 256 --num_class 100 --annotator $annotator \
# --pretrained_model $model