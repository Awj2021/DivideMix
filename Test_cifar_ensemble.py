## Write for testing the CIFAR-10 and CIFAR-100 dataset by reference to the Train_cifar.py.
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
import dataloader_cifar as dataloader
import torch.nn.functional as F
import ipdb

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--project_name', default='DivideMix', type=str, help='name of the wandb project.')
parser.add_argument('--noise_file', default='CIFAR-10_human.pt', type=str, help='name of the noise file.')
parser.add_argument('--wandb', action='store_true', help='use wandb to log the training process.')
parser.add_argument('--annotator', default='aggre_label', type=str, help='name of the annotator.')
parser.add_argument('--model', default='resnet18', type=str, help='name of the model.')
args = parser.parse_args()

checkpoint_dir = './cifar100_IDN50_dividemix_single_label'
# Last Models for the fair comparison.
label_model_pairs = {
    "random_label1": "cifar100_resnet18_64_random_label1_lambda_u_150.0_300_last.pth",
    "random_label2": "cifar100_resnet18_64_random_label2_lambda_u_150.0_300_last.pth",
    "random_label3": "cifar100_resnet18_64_random_label3_lambda_u_150.0_300_last.pth",
    "random_label4": "cifar100_resnet18_64_random_label4_lambda_u_150.0_300_last.pth",
    "random_label5": "cifar100_resnet18_64_random_label5_lambda_u_150.0_300_last.pth",
    "random_label6": "cifar100_resnet18_64_random_label6_lambda_u_150.0_300_last.pth",
}
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def test_ensemble(networks):
    print('*** Ensemble Testing ***')
    print("Number of networks: {}".format(len(networks)))
    
    ensemble_outputs = []
    ensemble_outputs_w_sf = []

    for net1, net2 in networks:
        net1.eval()
        net2.eval()
        outputs = []
        outputs_w_sf = []
        targets = []
        for _, (inputs, target) in enumerate(test_loader):
            inputs, target = inputs.cuda(), target.cuda()
            with torch.no_grad():
                output1 = net1(inputs)
                output2 = net2(inputs)
                output = (output1 + output2) / 2
                outputs.append(output)
                output_w_sf = (F.softmax(output1, dim=1) + F.softmax(output2, dim=1)) / 2
                outputs_w_sf.append(output_w_sf.cpu())
                targets.append(target.cpu().numpy())

        outputs = torch.cat(outputs, dim=0)
        outputs_w_sf = torch.cat(outputs_w_sf, dim=0)
        targets = np.concatenate(targets)
        
        ensemble_outputs.append(outputs)
        ensemble_outputs_w_sf.append(outputs_w_sf)

        outputs_all = torch.stack(ensemble_outputs, dim=0).mean(0)
        outputs_all_w_sf = torch.stack(ensemble_outputs_w_sf, dim=0).mean(0)

        _, predicted = torch.max(outputs_all, 1)
        _, predicted_w_sf = torch.max(outputs_all_w_sf, 1)
        correct = predicted.cpu().eq(torch.tensor(targets)).sum().item()
        correct_w_sf = predicted_w_sf.cpu().eq(torch.tensor(targets)).sum().item()

        accuracy = 100. * correct / len(targets)
        accuracy_w_sf = 100. * correct_w_sf / len(targets)
        print(f'Ensemble Accuracy of the model: {accuracy:.2f}%')
        print(f'Ensemble Accuracy of the model with softmax: {accuracy_w_sf:.2f}%')


def create_model():
    if args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_class)
    elif args.model == 'resnet34':
        model = ResNet34(num_classes=args.num_class)
    elif args.model == 'resnet50':
        model = ResNet50(num_classes=args.num_class)
    else:
        raise ValueError('Model not supported.')
    model = model.cuda()
    return model


loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,noise_file=args.noise_file, annotator=args.annotator)

cudnn.benchmark = True
test_loader = loader.run('test')

networks = []
print('*** Ensemble Testing ***')
print('='*50)
for model in label_model_pairs.values():
    print('Loading model from %s' % model)
    print('****** Building net ******')
    net1 = create_model()
    net2 = create_model()
    net1.load_state_dict(torch.load(os.path.join(checkpoint_dir, model))['net1'])
    net2.load_state_dict(torch.load(os.path.join(checkpoint_dir, model))['net2'])
    networks.append([net1,net2])
test_ensemble(networks)

