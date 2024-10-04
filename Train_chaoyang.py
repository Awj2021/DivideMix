from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_chaoyang as dataloader
from sklearn.mixture import GaussianMixture
import ipdb
import wandb
import math
from PreResNet import ResNet18, ResNet34, ResNet50 
from tqdm import tqdm


# TODO:
# 1. copy the dataset into the data folder.
parser = argparse.ArgumentParser(description='PyTorch chaoyang Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='decay rate for learning rate')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--warm_up_epochs', default=10, type=int)
parser.add_argument('--data_path', default='./chaoyang', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--project_name', default='chaoyang-training', type=str, help='name of the wandb project.')
parser.add_argument('--dataset', default='choayang', type=str, help='name of the dataset.')
parser.add_argument('--wandb', action='store_true', help='use wandb to log the training process.')
parser.add_argument('--annotator', default='label_A', type=str, help='name of the annotator.')
parser.add_argument('--model', default='resnet34', type=str, help='name of the model.')
parser.add_argument('--cosine', action='store_true', help='use cosine annealing.')
args = parser.parse_args()


# Initialize wandb
running_name = args.dataset + '_' + args.model + str(args.batch_size) + '_' + str(args.annotator) + '_lambda_u_' + str(args.lambda_u) 
wandb.init(project=args.project_name, name=running_name, config=args) if args.wandb else None

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    # num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)        
        
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss.item(), "labeled_loss": Lx.item(), "penalty": penalty.item()}) if args.wandb else None

    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path, index) in tqdm(enumerate(dataloader)):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 
        wandb.log({"warmup_loss": loss.item(), "warmup_penalty": penalty.item()}) if args.wandb else None

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    correct_w_sf = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)       
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            outputs_w_sf = (F.softmax(outputs1, dim=1) + F.softmax(outputs2, dim=1)) / 2
            _, predicted = torch.max(outputs, 1)  
            _, predicted_w_sf = torch.max(outputs_w_sf, 1)          
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()       
            correct_w_sf += predicted_w_sf.eq(targets).cpu().sum().item()
    acc = 100.*correct/total
    acc_w_sf = 100.*correct_w_sf/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    print("\n| Test Acc with Softmax: %.2f%%\n" %(acc_w_sf))
    wandb.log({"test_acc_wo_sf": acc, "test_acc_w_sf": acc_w_sf, "epch": epoch}) if args.wandb else None

    
def eval_train(epoch,model):
    print('\n==== evaluate next epoch training data loss ====')
    model.eval()
    # num_samples = args.num_batches*args.batch_size  # TODO: the num_samples should be the length of the training dataset.
    num_samples = len(eval_loader.dataset)
    losses = torch.zeros(num_samples)
    paths = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, path, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() # inputs: (batch_size, 3, 224, 224), targets: (batch_size,)
            # ipdb.set_trace() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]  
                paths.append(path[b])
    # ipdb.set_trace()
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob,paths  
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.num_epochs)) / 2
    else:
        if epoch%40==0 and epoch>0:  # put the original learning rate here. Just for 300 epochs.
            lr *= args.lr_decay_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_model():
    # if args.model == 'resnet18': # Please make sure that the model has the pre activate layer.
    #     model = ResNet18(num_classes=args.num_class)
    if args.model == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features,args.num_class)

    elif args.model == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features,args.num_class)

    elif args.model == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, args.num_class)
    else:
        raise ValueError('Model not supported.')
    
    model = model.cuda()
    return model


loader = dataloader.chaoyang_dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=5, annotator=args.annotator)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()


for epoch in range(args.num_epochs+1):   
    adjust_learning_rate(args, optimizer1, epoch)
    adjust_learning_rate(args, optimizer2, epoch)
        
    if epoch<1:     # warm up  
        train_loader = loader.run('warmup')
        print('| Warmup Net1\n')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('| Warmup Net2')
        warmup(net2,optimizer2,train_loader)                  
    else:       
        pred1 = (prob1 > args.p_threshold)  # divide dataset  
        pred2 = (prob2 > args.p_threshold)      
        
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)              # train net1
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)              # train net2
    
    eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
    prob1,paths1 = eval_train(epoch,net1) 

    print('\n==== net 2 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  
    prob2,paths2 = eval_train(epoch,net2) 

    print('\n==== Testing ====')
    test_loader = loader.run('test')
    test(net1,net2,test_loader)

    if epoch == args.num_epochs:
        last_checkpoint = os.path.join(args.project_name, running_name+'_' + str(epoch) + '_last.pth')
        torch.save({'net1': net1.state_dict(), 'net2': net2.state_dict()}, last_checkpoint)
        print('\nSaving Last Model to %s \n' % last_checkpoint)