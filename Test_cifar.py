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

# Last Models for the fair comparison.
checkoint_dir = './cifar100_IDN50_dividemix_MV'
label_model_pairs = {
    "aggre_label": "cifar100_resnet18_64_aggre_label_lambda_u_80.0_300_last.pth",
    "aggre_4_label": "cifar100_resnet18_64_aggre_4_label_lambda_u_25.0_300_last.pth",
    "aggre_5_label": "cifar100_resnet18_64_aggre_5_label_lambda_u_0.0_300_last.pth",
    "aggre_6_label": "cifar100_resnet18_64_aggre_6_label_lambda_u_0.0_300_last.pth",
}

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def eval_train(label, net1, net2):
    print('*** Evaluating Training Dataset ***')
    print('*** Generating the probs file ***')

    net1.eval()
    net2.eval()
    probs = []
    correct = 0
    total = 0
    for _, (inputs, targets, idx) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():   
            output1 = net1(inputs)
            output2 = net2(inputs)
            # output = (output1 + output2) / 2
            output_w_sf = (F.softmax(output1, dim=1) + F.softmax(output2, dim=1)) / 2
            probs.append(output_w_sf.cpu().numpy())

            _, predicted = torch.max(output_w_sf, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    accuracy = 100. * correct / total
    print(f'Training Accuracy of the model: {accuracy:.2f}%')
    output_probs = np.concatenate(probs, axis=0)
    probs_name = args.dataset + '_' + label + '_probs_2.npy' # regenerate the probs file.
    print(f'Saving the probs file to {probs_name}')
    np.save(os.path.join(checkoint_dir, probs_name), output_probs)


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
eval_loader = loader.run('eval_train')
networks = []

print('*** Ensemble Testing ***')
print('='*50)
for label, model in label_model_pairs.items():
    print('Loading model from %s' % model)
    print('****** Building net ******')
    checkpoint = os.path.join(checkoint_dir, model)
    net1 = create_model()
    net2 = create_model()

    net1.load_state_dict(torch.load(checkpoint)['net1'])
    net2.load_state_dict(torch.load(checkpoint)['net2'])
    eval_train(label, net1, net2) 

