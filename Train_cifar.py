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
from tqdm import tqdm 

 
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
parser.add_argument('--noise_file', default='Regenerated_Simulated_Human.pt', type=str, help='name of the noise file.')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--warm_up_epochs', default=30, type=int, help='number of warm-up epochs.')
parser.add_argument('--wandb', action='store_true', help='use wandb to log the training process.')
# parser.add_argument('--annotator', default='random_label1', type=str, help='name of the annotator.')
parser.add_argument('--annotator', default='', type=str, help='name of the annotator.')
parser.add_argument('--model', default='resnet18', type=str, help='name of the model.')
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')

parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

if args.annotator == 'two_annotators':
    annotators = ['random_label1', 'random_label2']
    mv_annotator = 'aggre_2_label'
elif args.annotator == 'three_annotators':
    annotators = ['random_label1', 'random_label2', 'random_label3']
    mv_annotator = 'aggre_3_label'
elif args.annotator == 'four_annotators':
    annotators = ['random_label1', 'random_label2', 'random_label3', 'random_label4']
    mv_annotator = 'aggre_4_label'
elif args.annotator == 'five_annotators':
    annotators = ['random_label1', 'random_label2', 'random_label3', 'random_label4', 'random_label5']
    mv_annotator = 'aggre_5_label'
elif args.annotator == 'six_annotators':
    annotators = ['random_label1', 'random_label2', 'random_label3', 'random_label4', 'random_label5', 'random_label6']
    mv_annotator = 'aggre_6_label'
else:
    raise ValueError('The annotator should be specified {}.'.format(args.annotator))

running_name = args.dataset + '_' + args.model + '_' + str(args.batch_size) + '_' + str(args.lambda_u) + '_' + str(len(annotators))+ 'new_algorithm_student_teacher'
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
        
        # For debugging, I found that the Lu is not 0. Just it is very tiny, like 0.0001.
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        # if epoch == 6:
        #     breakpoint() 
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0) # shape: (num_class,)
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
    logits = []
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, index) in tqdm(enumerate(dataloader)):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)
        # logits_mean = torch.softmax(outputs, dim=1).mean(0)
        logits.append(torch.softmax(outputs, dim=1))          
        loss = CEloss(outputs, labels)        
        L = loss
        L.backward()  
        optimizer.step() 

        wandb.log({'  epoch': epoch, 'num_iter': batch_idx, 'CE_loss': loss.item()}) if args.wandb else None
        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f \n'
                %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
    # dimension of the logits: (batch_size * num_iteration, num_class)
    # ipdb.set_trace()
    return torch.cat(logits, dim=0) # if I mean the logits along the dim=1, the results will be all 0.01.

def calculating_logits(net, dataloader):
    net.eval()
    logits = []
    with torch.no_grad():
        for batch_idx, (inputs, labels, _) in tqdm(enumerate(dataloader)):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            logits.append(torch.softmax(outputs, dim=1))
        return torch.cat(logits, dim=0)

def test(epoch, nets):
    nets = [net.eval() for net in nets]
    correct = 0
    correct_after_sf = 0
    total = 0
    global best_acc, best_acc_after_sf
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_all = [net(inputs) for net in nets]
            outputs = sum(outputs_all)
            outputs_after_sf = sum([torch.softmax(output, dim=1) for output in outputs_all])/len(outputs_all)
            _, predicted = torch.max(outputs, 1)            
            _, predicted_after_sf = torch.max(outputs_after_sf, 1)           
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item() 
            correct_after_sf += predicted_after_sf.eq(targets).cpu().sum().item()
    acc = 100.*correct/total
    acc_after_sf = 100.*correct_after_sf/total

    if acc > best_acc and epoch>warm_up:
        best_acc = acc
        best_checkpoint = os.path.join(args.project_name, running_name + '_best.pth') 
        torch.save({f'net{i+1}': net.state_dict() for i, net in enumerate(nets)}, best_checkpoint)
        print('\nSaving Best Model to %s \n' % best_checkpoint)

    if acc_after_sf > best_acc_after_sf and epoch>warm_up:
        best_acc_after_sf = acc_after_sf
        best_checkpoint = os.path.join(args.project_name, 'after_sf_' + running_name + '_best.pth') 
        torch.save({f'net{i+1}': net.state_dict() for i, net in enumerate(nets)}, best_checkpoint)
        print('\nSaving Best Model to %s \n' % best_checkpoint)
    wandb.log({'epoch': epoch, 'Accuracy_wo_sf': acc, 'Accuracy_w_sf': acc_after_sf}) if args.wandb else None
    print("\n| Test Epoch #%d\t w/o. Softmax Accuracy: %.2f%%, w. Softmax Accuracy: %.2f%%,\n" %(epoch,acc,acc_after_sf))  
    sys.stdout.write('%s: | Test Epoch #%d\t w/o. Softmax Accuracy: %.2f%%, w. Softmax Accuracy: %.2f%%,\n' %(args.dataset, epoch, acc, acc_after_sf))
    sys.stdout.flush()

    return acc, acc_after_sf

def eval_train_mv(nets, mv_loader): # mv_loader: the dataloader for majority vote.
    """
    here, I want to write a function to evaluate the training data using the majority vote labels.
    """
    nets = [net.eval() for net in nets]
    correct = [0 for _ in range(num_networks)]
    correct_after_sf = [0 for _ in range(num_networks)]
    total = 0
    # In this function, we don't need to sum all the outputs. We just need to get the outputs of each network.
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(mv_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_all = [net(inputs) for net in nets]
            outputs_after_sf = [torch.softmax(output, dim=1) for output in outputs_all]
            predicted = [torch.max(output, dim=1)[1] for output in outputs_all]
            predicted_after_sf = [torch.max(output, dim=1)[1] for output in outputs_after_sf]
            total += targets.size(0)
            for i in range(num_networks):
                correct[i] += predicted[i].eq(targets).cpu().sum().item() 
                correct_after_sf[i] += predicted_after_sf[i].eq(targets).cpu().sum().item()
    acc = [correct[i]/total for i in range(num_networks)]
    acc_after_sf = [correct_after_sf[i]/total for i in range(num_networks)]

    for i in range(len(acc)):
        wandb.log({f'\n Evaluation MV Accuracy_wo_sf_net{i+1}': 100.*acc[i], f'Evaluation MV Accuracy_w_sf_net{i+1}': 100. *acc_after_sf[i]}) if args.wandb else None
        sys.stdout.write('\n %s: | Evaluation MV Accuracy_wo_sf_net%d: %.2f%%, Evaluation MV Accuracy_w_sf_net%d: %.2f%%,\n' %(args.dataset, i+1, 100.*acc[i], i+1, 100.*acc_after_sf[i]))
        sys.stdout.flush()

    return acc, acc_after_sf


def eval_train(model, eval_loader):  
    """
    model & annotator: net2 for annotator1 | net1 for annotator2.
    """  
    model.eval()
    losses = torch.zeros(50000) # actually, the size of the dataset should be changed.    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]  # save the loss for each sample.        
    losses = (losses-losses.min())/(losses.max()-losses.min())    # normalize the loss
    input_loss = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    # gmm = GaussianMixture(num_components=2, covariance_type='full')
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)  # cluster the loss into two classes: noisy and clean. Shape: (50000,2)
    prob = prob[:,gmm.means_.argmin()]    # choose the cluster with lower mean as the clean sample. Shape: (50000,) 
    # prob = prob[:,cluster_means.argmin()]    # choose the cluster with lower mean as the clean sample. Shape: (50000,)
    return prob

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

def calculate_cosine_similarity(logits1, logits2):
    # Calculate the cosine similarity between two logits
    """
    logits1: torch.Tensor
        The logits of the first model.
    logits2: torch.Tensor
        The logits of the second model.
    """
    if logits1.shape != logits2.shape:
        raise ValueError("The shapes of the logits should be the same.")
    
    similarity = torch.mean(F.cosine_similarity(logits1, logits2, dim=1))
    return 1.0 - similarity


def choose_high_probability_comparied_model(similarity_matrix, num_nets, accuracy):
    """
    similarity_matrix: torch.Tensor
        The similarity matrix between the models.
    num_nets: int
        The number of the models.
    accuracy: torch.Tensor
        The accuracy of the models.
        """
    # Normalize the similarity matrix
    sample_index = []
    similarity_matrix = torch.clamp(similarity_matrix, min=0)
    # The below codes for calculating the similarity matrix are same.
    # similarity_matrix *= torch.tensor(accuracy).unsqueeze(1).expand(-1, 3) 
    similarity_matrix *= torch.tensor(accuracy)[None, :]
    # similarity_matrix = similarity_matrix / similarity_matrix.sum()

    for i in range(num_nets):
        flattened_similarity = torch.flatten(similarity_matrix)
        flattened_similarity /= flattened_similarity.sum()
        sample = torch.multinomial(flattened_similarity, num_samples=1, replacement=False)
        num_cols = similarity_matrix.shape[1]
        col_index = sample % num_cols 
        row_index = torch.div(sample, num_cols, rounding_mode='floor')
        # sample_2d_index = torch.stack((row_index, col_index), dim=1)
        sample_2d_index = (row_index, col_index)
        sample_index.append(sample_2d_index)
        similarity_matrix[row_index, :] = 0

    return sample_index # return the index of the models.

warm_up = args.warm_up_epochs
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path, noise_file=args.noise_file)

print('****** Building net ******')
nets = [create_model() for _ in range(len(annotators))]
print('****** Building copy net ******')
nets_copy = [create_model() for _ in range(len(annotators))]
cudnn.benchmark = True

criterion = SemiLoss()
optimizers = [optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) for net in nets]
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

best_acc = 0
best_acc_after_sf = 0
# Check the directory of saving models.
if not os.path.exists(args.project_name):
    os.makedirs(args.project_name)


latest_checkpoint = os.path.join(args.project_name, running_name + '_' + 'latest.pth')

if args.resume:
    if os.path.isfile(latest_checkpoint):
        print("=> loading checkpoint '{}'".format(latest_checkpoint))
        checkpoint = torch.load(latest_checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        for i, net in enumerate(nets):
            net.load_state_dict(checkpoint['nets'][f'net{i+1}'])
        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint['optimizers'][f'optimizer{i+1}'])
        best_acc = checkpoint['best_acc']
        best_acc_after_sf = checkpoint['best_acc_after_sf']
        print("=> loaded checkpoint '{}' (epoch {})".format(latest_checkpoint, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(latest_checkpoint))
        start_epoch = 0
else:
    start_epoch = 0

num_networks = len(annotators)
similarity_matrix = torch.zeros((num_networks, num_networks))
test_loader = loader.run('test')
eval_loaders = [loader.run('eval_train', annotator=annotators[i]) for i in range(num_networks)]
mv_eval_loader = loader.run('eval_train', annotator=mv_annotator)

last_5_acc = []
last_5_acc_after_sf = []

for epoch in range(start_epoch, args.num_epochs+1):    
    logits_list = []
    for optimizer in optimizers:
        adjust_learning_rate(args, optimizer, epoch) 

    if epoch<warm_up:
        warmup_trainloaders = [loader.run('warmup', annotator=annotators[i]) for i in range(len(annotators))]
        for i, (net, optimizer, warmup_trainloader) in enumerate(zip(nets, optimizers, warmup_trainloaders)):
            print(f'Warmup Net{i+1}: ')
            logits = warmup(epoch, net, optimizer, warmup_trainloader) # logits: (batch_size*num_iteration, num_class)
            logits_list.append(logits)
        # ipdb.set_trace()
        for i in range(num_networks):
            for j in range(num_networks):
                similarity_matrix[i, j] = calculate_cosine_similarity(logits_list[i], logits_list[j]) # I checked the similarity matrix, it is correct. it is a Symmetric Matrices.
    else:
        for i, net in enumerate(nets):
            nets_copy[i].load_state_dict(net.state_dict())

        for index in samples_index:
            stu_index, tea_index = index[0], index[1] # student index and teacher index.
            prob = eval_train(nets[stu_index], eval_loader=eval_loaders[stu_index])
            pred = (prob > args.p_threshold)
            print('\n Student Network: ', stu_index, ' Teacher Network: ', tea_index)
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred,prob,annotator=annotators[stu_index]) # co-divide
            train(epoch, nets[stu_index], nets_copy[tea_index], optimizers[stu_index], labeled_trainloader, unlabeled_trainloader)

        # Save the model as the latest one. 
        if epoch % 10 == 0:
            torch.save({
                        'epoch': epoch,
                        'nets': {f'net{i+1}': net.state_dict() for i, net in enumerate(nets)},
                        'optimizers': {f'optimizer{i+1}': optimizer.state_dict() for i, optimizer in enumerate(optimizers)},
                        'best_acc': best_acc,
                        'best_acc_after_sf': best_acc_after_sf
                    }, latest_checkpoint)
            
            print('\n Saving Checkpoint to %s \n' % latest_checkpoint)

        # here, calculate the similarity matrix between the models.
        print('\n Calculating the similarity matrix between the models.')
        for i in range(num_networks):
            logits = calculating_logits(nets[i], warmup_trainloaders[i])
            logits_list.append(logits) 
    
        # Update the sample index.
        for i in range(num_networks):
            for j in range(num_networks):
                similarity_matrix[i, j] = calculate_cosine_similarity(logits_list[i], logits_list[j])

    print('\n Evaluating the models using the majority vote labels.')    
    _, eval_acc_sf = eval_train_mv(nets, mv_eval_loader)
    samples_index = choose_high_probability_comparied_model(similarity_matrix, num_networks, eval_acc_sf)

    print('\n Testing the models.') 
    acc, acc_after_sf = test(epoch, nets)

    # Append current epoch accuracies to the lists
    if epoch > args.num_epochs - 6:
        last_5_acc.append(acc)
        last_5_acc_after_sf.append(acc_after_sf)

        # Keep only the last 5 epochs
        if len(last_5_acc) > 5:
            last_5_acc.pop(0)
        if len(last_5_acc_after_sf) > 5:
            last_5_acc_after_sf.pop(0)

        # Calculate average accuracy for the last 5 epochs
        avg_acc_last_5 = sum(last_5_acc) / len(last_5_acc)
        avg_acc_after_sf_last_5 = sum(last_5_acc_after_sf) / len(last_5_acc_after_sf)

        # Log the average accuracies
        wandb.log({'Average_Accuracy_Last_5_Epochs': avg_acc_last_5, 'Average_Accuracy_After_SF_Last_5_Epochs': avg_acc_after_sf_last_5}) if args.wandb else None
        print("\n| Average Accuracy for Last 5 Epochs: %.2f%%, Average Accuracy After SF for Last 5 Epochs: %.2f%%\n" % (avg_acc_last_5, avg_acc_after_sf_last_5))
        