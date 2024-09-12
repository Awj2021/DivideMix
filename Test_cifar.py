## Write for testing the CIFAR-10 and CIFAR-100 dataset by reference to the Train_cifar.py.
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import wandb
import ipdb
import math
import torch.nn.functional as F

# TODO: Reload the saved model and test the model on the test dataset.
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--project_name', default='DivideMix', type=str, help='name of the wandb project.')
parser.add_argument('--noise_file', default='CIFAR-10_human.pt', type=str, help='name of the noise file.')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--warm_up_epochs', default=30, type=int, help='number of warm-up epochs.')
parser.add_argument('--wandb', action='store_true', help='use wandb to log the training process.')
parser.add_argument('--annotator', default='aggre_label', type=str, help='name of the annotator.')
parser.add_argument('--model', default='resnet18', type=str, help='name of the model.')
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--pretrained_model', nargs='+', type=str, help='path to the pretrained model.')
# parser.add_argument('--ensemble', action='store_true', help='use ensemble model.')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

def eval_train(networks):
    print('*** Evaluating Training Dataset ***')
    probs = []
    for _, (inputs, targets, idx) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        output_w_sfs = []
        for net in networks:
            net.eval()
            with torch.no_grad():   
                output = net(inputs)
                output_w_sf = F.softmax(output, dim=1)
                output_w_sfs.append(output_w_sf)
        ensemble_output_w_sf = torch.stack(output_w_sfs).mean(0)
        probs.append(ensemble_output_w_sf.cpu().numpy())
    probs = np.concatenate(probs, axis=0)        
    # ipdb.set_trace()
    # probs_name = '_'.join(args.pretrained_model) + '_probs.npy'
    probs_name = args.dataset + '_' + args.annotator + '_probs.npy'
    np.save(probs_name, probs)

def test(networks):
    print('Testing using the test dataset...')
    correct = 0
    correct_w_sf = 0
    total = 0
    # outputs = []
    for _, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = []
        outputs_w_sf = []
        for net in networks:
            net.eval()
            with torch.no_grad():   
                output = net(inputs)
                output_w_sf = F.softmax(output, dim=1)
                outputs.append(output)
                outputs_w_sf.append(output_w_sf)
        # ipdb.set_trace()
        ensemble_output = torch.stack(outputs).mean(0)
        ensemble_output_w_sf = torch.stack(outputs_w_sf).mean(0)
        _, predicted = torch.max(ensemble_output, 1)
        _, predicted_w_sf = torch.max(ensemble_output_w_sf, 1)            
                        
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item() 
        correct_w_sf += predicted_w_sf.eq(targets).cpu().sum().item()    
    acc = 100.*correct/total
    acc_w_sf = 100.*correct_w_sf/total
    print("Test Ensemble Accuracy: {}\n".format(acc))
    print("Test Ensemble Accuracy with Softmax: {}\n".format(acc_w_sf))
    

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
# Eval_loader is used for getting the predictions of the training dataset, which will be used for the cleanlab. 
eval_loader = loader.run('eval_train')

networks = []
print('*** Ensemble Testing ***')
print(args.pretrained_model)
print('='*50)
for divide_model in args.pretrained_model:
    print('Loading model from %s' % divide_model)
    print('****** Building net ******')
    net1 = create_model()
    net2 = create_model()
    net1.load_state_dict(torch.load(divide_model)['net1'])
    net2.load_state_dict(torch.load(divide_model)['net2'])
    networks.extend([net1,net2])
test(networks)
# eval_train(networks) # when do not used for the cleanlab, comment this line.

