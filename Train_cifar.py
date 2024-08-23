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
# from pycave.bayes import GaussianMixture

# TODO: requirements for environment.
# TODO: setting the GMM to GPU.
# TODO: image size of dataset.
 
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
parser.add_argument('--annotator', default='random_label1', type=str, help='name of the annotator.')
parser.add_argument('--model', default='resnet18', type=str, help='name of the model.')
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

# running name should include the dataset and the noise mode.
running_name = args.dataset + '_' + args.model + '_' + str(args.batch_size) + '_' + str(args.annotator) + '_lambda_u_' + str(args.lambda_u) 
wandb.init(project=args.project_name, name=running_name, config=args) if args.wandb else None

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
      
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):  # labels_x is the target label.    
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()          # Get two different unlabeled samples.But from the dataloader, the images are the same.       
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        # FIXME: check the dimension of the labels_x.
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        # ipdb.set_trace()
        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       # shape: (batch_size, num_class)
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # ipdb.set_trace()
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0) # shape: (4*batch_size, 3, 32, 32)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0) # shape: (4*batch_size, num_class)

        idx = torch.randperm(all_inputs.size(0)) # generate the random index.

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b       
        mixed_target = l * target_a + (1 - l) * target_b # The ground truth labels of mixed data for calculating the loss.
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2] # refer to the all_inputs. [inputs_x, inputs_x2, xxx]
        logits_u = logits[batch_size*2:] # refer to the all_inputs. [xxx, xxx, inputs_u, inputs_u2]
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'epoch': epoch, 'num_iter': num_iter, 'Labeled_loss': Lx.item(), 'Unlabeled_loss': Lu.item(), 'loss': loss.item(), 'penalty': penalty.item(), 'lamb': lamb}) if args.wandb else None
        sys.stdout.write('\r')
        sys.stdout.write('%s:  Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        # if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
        #     penalty = conf_penalty(outputs)
        #     L = loss + penalty      
        # elif args.noise_mode=='sym':   
        L = loss
        L.backward()  
        optimizer.step() 

        wandb.log({'epoch': epoch, 'num_iter': num_iter, 'CE_loss': loss.item()}) if args.wandb else None
        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    wandb.log({'epoch': epoch, 'Accuracy': acc}) if args.wandb else None
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]  # save the loss for each sample.        
    losses = (losses-losses.min())/(losses.max()-losses.min())    # normalize the loss
    all_loss.append(losses)
    # ema
    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else: 
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    # gmm = GaussianMixture(num_components=2, covariance_type='full').cuda()
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)  # cluster the loss into two classes: noisy and clean. Shape: (50000,2)
    # TODO: check the dimension of the prob.
    # ipdb.set_trace()
    # cluster_means = torch.mean(prob, dim=0)  # calculate the mean of the two clusters. Shape: (2,)
    prob = prob[:,gmm.means_.argmin()]    # choose the cluster with lower mean as the clean sample. Shape: (50000,) 
    # prob = prob[:,cluster_means.argmin()]    # choose the cluster with lower mean as the clean sample. Shape: (50000,)
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

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

# Below function is copied from the official implementation of ProMix.
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.num_epochs)) / 2
    # else:
    #     steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    #     if steps > 0:
    #         lr = lr * (args.lr_decay_rate ** steps)
    else:
        if epoch%150==0 and epoch>0:  # put the original learning rate here. Just for 300 epochs.
            lr *= args.lr_decay_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

stats_log=open('./checkpoint/'+running_name+'_stats.txt','w') 
test_log=open('./checkpoint/'+running_name+'_acc.txt','w')     

warm_up = args.warm_up_epochs

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file=args.noise_file, annotator=args.annotator)

print('****** Building net ******')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
# if args.noise_mode=='asym':
#     conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

for epoch in range(args.num_epochs+1):   
    # lr=args.lr
    # if epoch % 150 == 0 and epoch>0:
    #     lr /= 10
    adjust_learning_rate(args, optimizer1, epoch)
    adjust_learning_rate(args, optimizer2, epoch)      
    # for param_group in optimizer1.param_groups: # adjust learning rate when epoch >= 150.
    #     param_group['lr'] = lr       
    # for param_group in optimizer2.param_groups:
    #     param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   # The probability is calculated when evaluating. 
        prob2,all_loss[1]=eval_train(net2,all_loss[1])   # Use the all train_data and the noisy labels.        
               
        pred1 = (prob1 > args.p_threshold)      # The threshold is set to 0.5 except for the CIFAR-10 dataset r = 0.9.
        pred2 = (prob2 > args.p_threshold)      # The list of pred only contains the True or False.
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2         
    # Save the model as the last one model. 
    if epoch == args.num_epochs:
        torch.save(net1.state_dict(),'./checkpoint/'+running_name+'_last_net1.pth')
        torch.save(net2.state_dict(),'./checkpoint/'+running_name+'_last_net2.pth')

    test(epoch,net1,net2)  


