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
# from pycave.bayes import GaussianMixture

# TODO: setting the GMM to GPU.

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--project_name', default='DivideMix', type=str, help='name of the wandb project.')
parser.add_argument('--noise_file', default='CIFAR-10_human.pt', type=str, help='name of the noise file.')
parser.add_argument('--warm_up_epochs', default=10, type=int, help='number of warm-up epochs.')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

# running name should include the dataset and the noise mode.
running_name = args.dataset + '_' + str(args.batch_size) + '_' + str(args.lr)
wandb.init(project=args.project_name, name=running_name, config=args)

# Training
def train(epoch,net,net2,net3,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    net3.eval()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):  # labels_x is the target label.    
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()          # Get two different unlabeled samples.But from the dataloader, the images are the same.       
        batch_size = inputs_x.size(0)
        
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        # convert the labels to one-hot encoding.
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)
            outputs_u31 = net3(inputs_u)
            outputs_u32 = net3(inputs_u2)

            # FIXME: Check the outputs of the network.          
            # TODO: 1. Seperately calculate the pu for net2 and net3. And then average the pu.
            # TODO: 2. Seperately calculate the px for net2 and net3. And then find the overlap of the pus to get a new pu.
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) \
                  + torch.softmax(outputs_u22, dim=1) + torch.softmax(outputs_u31, dim=1) + torch.softmax(outputs_u32, dim=1)) / 6       
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

        wandb.log({'epoch': epoch, 'num_iter': num_iter, 'Labeled_loss': Lx.item(), 'Unlabeled_loss': Lu.item(), 'loss': loss.item(), 'penalty': penalty.item()})
        sys.stdout.write('\r')
        sys.stdout.write('%s: Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader): # annotator=0, 1, 2 for cifair10.
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):     
        # how to convert the labels into one-hot encoding?
        # FIXME: check the dimension of the labels.
        labels = torch.zeros(inputs.size(0), args.num_class).scatter_(1, labels.view(-1,1), 1) # convert the labels to one-hot encoding.
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

        wandb.log({'epoch': epoch, 'num_iter': num_iter, 'CE_loss': loss.item()})
        sys.stdout.write('\r')
        sys.stdout.write('%s: Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2, net3):
    net1.eval()
    net2.eval()
    net3.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs3 = net3(inputs)           
            outputs = outputs1+outputs2+outputs3 # shape: (batch_size, num_class)
            # TODO: please make sure the outputs are the sum of the three networks.
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    wandb.log({'epoch': epoch, 'Accuracy': acc})
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

# TODO: use the trained model to evaluate the other two labels. And get the overlap of the clean samples.
def eval_train(model, annotator=0):    
    model.eval()
    eval_loader_list = []
    probs = []
    if annotator==0:
        eval_loader_list = [eval_loader_2, eval_loader_3]
    elif annotator==1:
        eval_loader_list = [eval_loader_1, eval_loader_3]
    elif annotator==2:
        eval_loader_list = [eval_loader_1, eval_loader_2]
    else:
        raise ValueError('The annotator is not valid.')
    
    for eval_loader in eval_loader_list:
        losses = torch.zeros(50000)    
        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
                targets = torch.zeros(inputs.size(0), args.num_class).scatter_(1, targets.view(-1,1), 1) 
                inputs, targets = inputs.cuda(), targets.cuda() 
                outputs = model(inputs) 
                loss = CE(outputs, targets)  
                for b in range(inputs.size(0)):
                    losses[index[b]]=loss[b]  # save the loss for each sample.        
        losses = (losses-losses.min())/(losses.max()-losses.min())    # normalize the loss
        # all_loss[annotator].append(losses)
    
        input_loss = losses.reshape(-1,1)
    
        # fit a two-component GMM to the loss
        print('GMM fitting on the model %d using other two dataset..'%(annotator + 1))
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        # gmm = GaussianMixture(num_components=2, covariance_type='full').cuda()
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)  # cluster the loss into two classes: noisy and clean. Shape: (50000,2)
        prob = prob[:,gmm.means_.argmin()]    # choose the cluster with lower mean as the clean sample. Shape: (50000,) 
        probs.append(prob)
    return probs

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
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

stats_log=open('./checkpoint/'+running_name+'_stats.txt','w') 
test_log=open('./checkpoint/'+running_name+'_acc.txt','w')     

# if args.dataset=='cifar10':
#     warm_up = a
# elif args.dataset=='cifar100':
#     warm_up = 30
warm_up = args.warm_up_epochs

loader = dataloader.cifar_dataloader(args.dataset,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log, noise_file=args.noise_file) # need to modify the noise_file.

print('*************** Building net ****************')
net1 = create_model()
net2 = create_model()
net3 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer3 = optim.SGD(net3.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

all_loss = [[],[],[]] # save the history of losses from two networks

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups: # adjust learning rate when epoch >= 150.
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer3.param_groups:
        param_group['lr'] = lr        
    test_loader = loader.run('test')
    eval_loader_1 = loader.run('eval_train', annotator=0)
    eval_loader_2 = loader.run('eval_train', annotator=1)
    eval_loader_3 = loader.run('eval_train', annotator=2)   
    
    if epoch<warm_up:
        print(' =========== Warming up the networks...')
        # ipdb.set_trace()       
        print('Warmup Net1')
        warmup_trainloader_1 = loader.run('warmup', annotator=0) # warmup net1
        warmup(epoch,net1,optimizer1,warmup_trainloader_1)   

        print('\nWarmup Net2')
        warmup_trainloader_2 = loader.run('warmup', annotator=1) # warmup net2
        warmup(epoch,net2,optimizer2,warmup_trainloader_2) 

        print('\nWarmup Net3')
        warmup_trainloader_3 = loader.run('warmup', annotator=2) # warmup net3
        warmup(epoch,net3,optimizer3,warmup_trainloader_3)
   
    else:         
        probs_2_3 = eval_train(net1, annotator=0)   # The probability is calculated when evaluating. 
        probs_1_3 = eval_train(net2, annotator=1)   # Use the all train_data and the noisy labels.     
        probs_1_2 = eval_train(net3, annotator=2)   # Use the all train_data and the noisy labels.   
               
        pred_2_3 = [(prob > args.p_threshold) for prob in probs_2_3]      # The threshold is set to 0.5 except for the CIFAR-10 dataset r = 0.9.
        pred_1_3 = [(prob > args.p_threshold) for prob in probs_1_3]      # The list of pred only contains the True or False.
        pred_1_2 = [(prob > args.p_threshold) for prob in probs_1_2]     # pred1.shape: (50000,)  prob1.shape: (50000,)
        
        # ipdb.set_trace()
        print('Train Net1')
        # pred_overlap_23 = pred2 & pred3
        pred_overlap_23 = pred_2_3[0] & pred_2_3[1]
        prob_overlap_23 = (probs_2_3[0] + probs_2_3[1]) / 2
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred_overlap_23,prob_overlap_23,annotator=0) # co-divide
        train(epoch,net1,net2,net3,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        pred_overlap_13 = pred_1_3[0] & pred_1_3[1]
        prob_overlap_13 = (probs_1_3[0] + probs_1_3[1]) / 2
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred_overlap_13,prob_overlap_13,annotator=1) # co-divide
        train(epoch,net2,net1,net3,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2 
        
        print('\nTrain Net3')  
        pred_overlap_12 = pred_1_2[0] & pred_1_2[1]
        prob_overlap_12 = (probs_1_2[0] + probs_1_2[1]) / 2
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred_overlap_12,prob_overlap_12,annotator=2) # co-divide
        train(epoch,net3,net1,net2, optimizer3,labeled_trainloader, unlabeled_trainloader) # train net3.
    # Save the model as the last one model. 
    if epoch == args.num_epochs:
        torch.save(net1.state_dict(),'./checkpoint/'+running_name+'_last_net1.pth')
        torch.save(net2.state_dict(),'./checkpoint/'+running_name+'_last_net2.pth')
        torch.save(net3.state_dict(),'./checkpoint/'+running_name+'_last_net3.pth')

    test(epoch,net1,net2,net3)  


