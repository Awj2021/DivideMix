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
from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics import MeanMetric


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=150, type=float, help='weight for unsupervised loss') # set lambda_u = 150 for all the experiments.
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--project_name', default='DivideMix', type=str, help='name of the wandb project.')
parser.add_argument('--noise_file', default='cifar100_noisy_labels_noise_50.pt', type=str, help='name of the noise file.')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--warm_up_epochs', default=30, type=int, help='number of warm-up epochs.')
parser.add_argument('--wandb', action='store_true', help='use wandb to log the training process.')
parser.add_argument('--annotator', default='random_label1', type=str, help='name of the annotator.')
parser.add_argument('--model', default='resnet18', type=str, help='name of the model.')
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
# Settings of the conformal prediction.
parser.add_argument('--calibration_file', default='cifar100_split_calibration_noise_50.pt', type=str, help='name of the calibration file.')
parser.add_argument('--calibration_alpha', default=0.1, type=float, help='user-chosen error rate for the conformal prediction.')
parser.add_argument('--mixmatch', action='store_true', default=False, help='use mixmatch.')
parser.add_argument('--conformal_prediction', action='store_true', default=False, help='use conformal prediction.')
parser.add_argument('--cp_weight', default=0.5, type=float, help='hyperparameter for the conformal prediction.')
parser.add_argument('--cp_loss', default='kl', type=str, choices=['ce', 'mse', 'kl'], help='loss function for the conformal prediction.')
parser.add_argument('--clean_or_noisy', default='clean', type=str, choices=['clean', 'noisy'], help='clean or noisy calibration sets.')
parser.add_argument('--resume_checkpoint', default=None, type=str, help='name of the checkpoint to resume training.')
parser.add_argument('--annealing_start_epoch', default=120, type=int, help='epoch to start annealing the weight.')
parser.add_argument('--annealing_end_epoch', default=300, type=int, help='epoch to end annealing the weight.')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)
# you should have the goal of life.
# running name should include the dataset and the noise mode.
wandb.init(project=args.project_name, config=args) if args.wandb else None

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, q_hat):
    net.train()
    net2.eval() #fix one network and train the other
    
    # Torchmetrics mean metrics for loss tracking
    mean_Lx = MeanMetric().cuda()
    mean_Lu = MeanMetric().cuda()
    mean_loss = MeanMetric().cuda()
    mean_penalty = MeanMetric().cuda()
    mean_lamb = MeanMetric().cuda()
    
    unlabeled_train_iter = iter(unlabeled_trainloader)   
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    # the unlabeled_trainloader maybe has the different size with the labeled_trainloader.
    # So, it use the re-iter to get the matched number of unlabeled samples with the labeled samples.
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):  # labels_x is the target label.    
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()          # Get two different unlabeled samples.But from the dataloader, the images are the same.       
        batch_size = inputs_x.size(0)
        
        # labels_x is already in soft-label format (probability distribution), no need to convert to one-hot
        # Ensure labels_x has the correct shape for soft labels
        if labels_x.dim() == 1:
            # If labels_x is 1D (hard labels), convert to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        # If labels_x is already 2D (soft labels), keep as is
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
            
            # when we use the Conformal Prediction, temparature sharpening is operated before the conformal prediction.
            # baseline method:
            if not args.conformal_prediction:
                ptu = pu**(1/args.T) # temparature sharpening 
                ptu = torch.clamp(ptu, min=1e-8, max=1.0)
            
            # method 1: q_b = (1 - cp_weight) * pred + cp_weight * average(pred). Temperature sharpening is operated after the conformal prediction.
            if args.conformal_prediction and args.cp_loss == 'mse':
                pu_pred_set = pu >= (1 - q_hat) # here, we use the average of the two networks.
                mask = pu_pred_set.float()
                mask_sum = mask.sum(dim=1, keepdim=True)
                mask_sum = torch.clamp(mask_sum, min=1e-8)  # Prevent division by zero
                aver_pu = mask / mask_sum # average the soft softmax outputs.
                ptu = ptu**(1/args.T)
                ptu = torch.clamp(ptu, min=1e-8, max=1.0)

            # method 2: use the CE loss function for the unlabeled data. Temperature sharpening is not used.
            if args.conformal_prediction and args.cp_loss == 'ce':
                pu_pred_set = pu >= (1 - q_hat) # here, we use the average of the two networks.
                mask = pu_pred_set.float()
                mask_sum = mask.sum(dim=1, keepdim=True)
                mask_sum = torch.clamp(mask_sum, min=1e-8)  # Prevent division by zero
                ptu = mask / mask_sum # average the soft softmax outputs. 
                
            ptu_sum = ptu.sum(dim=1, keepdim=True)
            ptu_sum = torch.clamp(ptu_sum, min=1e-8)  # Prevent division by zero
            targets_u = ptu / ptu_sum # normalize
            targets_u = targets_u.detach()       # shape: (batch_size, num_class)
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px     

            # no temperature sharpening.
            # ptx = px
            ptx = px**(1/args.T) # temparature sharpening 
            ptx = torch.clamp(ptx, min=1e-8, max=1.0)
                                                                                                                                                                                                                                                                                                      
            ptx_sum = ptx.sum(dim=1, keepdim=True)
            ptx_sum = torch.clamp(ptx_sum, min=1e-8)  # Prevent division by zero
            targets_x = ptx / ptx_sum # normalize           
            targets_x = targets_x.detach()       
        
        if args.mixmatch:
            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)        
            l = max(l, 1-l)
                    
            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0) # shape: (4*batch_size, 3, 32, 32)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0) # shape: (4*batch_size, num_class)

            idx = torch.randperm(all_inputs.size(0)) # generate the random index.

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx] # target_a and target_b are the same as all_targets, but with different order.
            
            mixed_input = l * input_a + (1 - l) * input_b     # actually, do the same operation for the labeled and unlabeled data.
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
        else:
            # We don't use the mixmatch here.
            # Without mixmatch - direct training on labeled and unlabeled data
            inputs_all_x = torch.cat([inputs_x, inputs_x2], dim=0)
            inputs_all_u = torch.cat([inputs_u, inputs_u2], dim=0)
            outputs_x = net(inputs_all_x)
            outputs_u = net(inputs_all_u)
            targets_x = torch.cat([targets_x, targets_x], dim=0)
            targets_u = torch.cat([targets_u, targets_u], dim=0)
            Lx, Lu, lamb = criterion(outputs_x, targets_x, outputs_u, targets_u, epoch+batch_idx/num_iter, warm_up)
            
            # TODO: fix the code below before this weekend.
            # labeled_all_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
            # labeled_all_targets = torch.cat([targets_x, targets_x], dim=0)
            # unlabeled_all_inputs = torch.cat([inputs_u, inputs_u2], dim=0)
            # unlabeled_all_targets = torch.cat([targets_u, targets_u], dim=0)
            # regularization
            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda()        
            pred_mean = torch.softmax(torch.cat([outputs_x, outputs_u], dim=0), dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            loss = Lx + lamb * Lu + penalty

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update torchmetrics mean metrics
        mean_Lx.update(Lx.detach())
        mean_Lu.update(Lu.detach())
        mean_loss.update(loss.detach())
        mean_penalty.update(penalty.detach())
        mean_lamb.update(torch.tensor(lamb, device=loss.device))

        sys.stdout.write('\r')
        sys.stdout.write('%s:  Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.5f  Unlabeled loss: %.5f  lambda: %.5f'
                %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item(), lamb)) 
        sys.stdout.flush()

    # Log to wandb only at the end of the epoch
    wandb.log({
        'epoch': epoch,
        'train/num_iter': num_iter,
        'train/Labeled_loss': mean_Lx.compute().item(),
        'train/Unlabeled_loss': mean_Lu.compute().item(),
        'train/loss': mean_loss.compute().item(),
        'train/penalty': mean_penalty.compute().item(),
        'train/lamb': mean_lamb.compute().item(),
        'train/mixmatch': args.mixmatch
    }, step=epoch) if args.wandb else None

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        L = loss
        L.backward()  
        optimizer.step() 

        wandb.log({'epoch': epoch, 'num_iter': num_iter, 'CE_loss': loss.item()}, step=epoch) if args.wandb else None
        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    correct_after_sf = 0
    total = 0
    global best_acc, best_acc_after_sf

    # Calculate calibration scores
    q_hat1, q_hat2, q_hat_aver = calibration(net1,net2)
    pred_net1 = []
    pred_net2 = []
    pred_aver = []
    targets_all = [target for _, target in test_loader.dataset]

    # Test dataset predictions.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            pred_net1.append(torch.softmax(outputs1, dim=1))
            pred_net2.append(torch.softmax(outputs2, dim=1))
            pred_aver.append((torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2)
            outputs_after_sf = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2
            _, predicted = torch.max(outputs1, 1)            
            _, predicted_after_sf = torch.max(outputs_after_sf, 1)           
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item() 
            correct_after_sf += predicted_after_sf.eq(targets).cpu().sum().item()
    pred_net1 = torch.cat(pred_net1, dim=0)
    pred_net2 = torch.cat(pred_net2, dim=0)
    pred_aver = torch.cat(pred_aver, dim=0)
    targets_all = torch.tensor(targets_all).cuda()  # Move to CUDA
    # calculate the conformal prediction for each network
    pred_set1 = pred_net1 >= (1 - q_hat1)
    pred_set2 = pred_net2 >= (1 - q_hat2)
    pred_set_aver = pred_aver >= (1 - q_hat_aver)
    
    # Calculate prediction set sizes
    pred_set_size1 = pred_set1.sum(dim=1).float().mean()
    pred_set_size2 = pred_set2.sum(dim=1).float().mean()
    pred_set_size_aver = pred_set_aver.sum(dim=1).float().mean()
    
    print(f"\nTest Set Prediction Set Sizes:")
    print(f"Network 1 - Avg Prediction Set Size: {pred_set_size1:.3f}")
    print(f"Network 2 - Avg Prediction Set Size: {pred_set_size2:.3f}")
    print(f"Ensemble  - Avg Prediction Set Size: {pred_set_size_aver:.3f}")
    
    # calculate the empirical coverage.
    coverage_net1 = pred_set1[np.arange(pred_set1.shape[0]), targets_all].float().mean()
    coverage_net2 = pred_set2[np.arange(pred_set2.shape[0]), targets_all].float().mean()
    coverage_aver = pred_set_aver[np.arange(pred_set_aver.shape[0]), targets_all].float().mean()

    print(f"\nTest Set Conformal Prediction Results:")
    print(f"Network 1 - Coverage: {coverage_net1:.3f}")
    print(f"Network 2 - Coverage: {coverage_net2:.3f}")
    print(f"Ensemble  - Coverage: {coverage_aver:.3f}")

    acc = 100.*correct/total
    acc_after_sf = 100.*correct_after_sf/total

    # calculating the calibration error here.
    calibration_error_net1_l2 = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='l2')
    calibration_error_net2_l2 = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='l2')

    calibration_error_net1_l1 = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='l1')
    calibration_error_net2_l1 = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='l1')

    calibration_error_net1_max = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='max')
    calibration_error_net2_max = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='max')

    calibration_net1_value = calibration_error_net1_l2(pred_net1, targets_all)
    calibration_net2_value = calibration_error_net2_l2(pred_net2, targets_all)
    calibration_net1_value_l1 = calibration_error_net1_l1(pred_net1, targets_all)
    calibration_net2_value_l1 = calibration_error_net2_l1(pred_net2, targets_all)
    calibration_net1_value_max = calibration_error_net1_max(pred_net1, targets_all)
    calibration_net2_value_max = calibration_error_net2_max(pred_net2, targets_all)

    print(f"Network 1 - Calibration Error (L2): {calibration_net1_value:.3f}")
    print(f"Network 2 - Calibration Error (L2): {calibration_net2_value:.3f}")
    print(f"Network 1 - Calibration Error (L1): {calibration_net1_value_l1:.3f}")
    print(f"Network 2 - Calibration Error (L1): {calibration_net2_value_l1:.3f}")
    print(f"Network 1 - Calibration Error (Max): {calibration_net1_value_max:.3f}")
    print(f"Network 2 - Calibration Error (Max): {calibration_net2_value_max:.3f}")

    wandb.log({
        'epoch': epoch, 
        'accuracy/Accuracy_w_sf': acc_after_sf, 
        # 'Best_Acc_w_sf': best_acc_after_sf,
        'cp/test_coverage_net1': coverage_net1, 
        'cp/test_coverage_net2': coverage_net2, 
        'cp/test_coverage_aver': coverage_aver,
        'cp/test_pred_set_size_net1': pred_set_size1,
        'cp/test_pred_set_size_net2': pred_set_size2,
        'cp/test_pred_set_size_aver': pred_set_size_aver,
        'calibration/test_calibration_error_net1_l2': calibration_net1_value,
        'calibration/test_calibration_error_net2_l2': calibration_net2_value,
        'calibration/test_calibration_error_net1_l1': calibration_net1_value_l1,
        'calibration/test_calibration_error_net2_l1': calibration_net2_value_l1,
        'calibration/test_calibration_error_net1_max': calibration_net1_value_max,
        'calibration/test_calibration_error_net2_max': calibration_net2_value_max,
    }, step=epoch) if args.wandb else None
    
    print("\n| Test Epoch #%d\t w/o. Softmax Accuracy: %.2f%%, w. Softmax Accuracy: %.2f%%,\n" %(epoch,acc,acc_after_sf))

def calibration(net1,net2):
    # actually, the net1 and net2 are not trained.
    # net1.eval()
    # net2.eval()
    calibration_pred1 = []
    calibration_pred2 = []
    calibration_pred_aver = []
    calib_n = len(calibration_loader.dataset)
    targets_all = torch.tensor(calibration_loader.dataset.cali_label).cuda()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(calibration_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = torch.softmax(net1(inputs), dim=1)
            outputs2 = torch.softmax(net2(inputs), dim=1)
            calibration_pred_aver.append((outputs1 + outputs2) / 2)
            # calculate the calibration prediction seperately for each network
            calibration_pred1.append(outputs1)
            calibration_pred2.append(outputs2)
    
    calibration_pred1 = torch.cat(calibration_pred1, dim=0)
    calibration_pred2 = torch.cat(calibration_pred2, dim=0)
    calibration_pred_aver = torch.cat(calibration_pred_aver, dim=0)
    
    # calculate the calibration prediction seperately for each network
    # get conformal scores for each network
    cal_score1 = 1 - calibration_pred1[np.arange(calib_n), targets_all]
    cal_score2 = 1 - calibration_pred2[np.arange(calib_n), targets_all]
    cal_score_aver = 1 - calibration_pred_aver[np.arange(calib_n), targets_all]
    
    # Move tensors to CPU before converting to numpy
    cal_score1 = cal_score1.cpu().numpy()
    cal_score2 = cal_score2.cpu().numpy()
    cal_score_aver = cal_score_aver.cpu().numpy()
    
    # get adjusted quantile. network1 and network2 have same number of samples.
    q_level = np.ceil((calib_n + 1) * (1 - args.calibration_alpha)) / calib_n
    q_hat1 = np.quantile(cal_score1, q_level, method='higher')
    q_hat2 = np.quantile(cal_score2, q_level, method='higher')
    q_hat_aver = np.quantile(cal_score_aver, q_level, method='higher')
    return q_hat1, q_hat2, q_hat_aver

def annealing_weight(epoch, start_epoch = 120, end_epoch = 300):
    if epoch < start_epoch:
        return 1
    elif epoch >= start_epoch and epoch < end_epoch:
        return 1 - (epoch - start_epoch) / (end_epoch - start_epoch)
    else:
        return 0

def annealing_weight_version1(epoch, start_epoch = 120, end_epoch = 300):
    if epoch < start_epoch:
        return 1
    elif epoch >= start_epoch and epoch < end_epoch:
        return 1 - (epoch - start_epoch) / (end_epoch - start_epoch)
    else:
        return 0

def annealing_weight_version2(epoch, start_epoch = 120, end_epoch = 300):
    if epoch < start_epoch:
        return 0
    else:
        return min(1, (epoch - start_epoch) / (end_epoch - start_epoch))

def cosine_annealing(epoch, start_epoch = 120, end_epoch = 300):
    if epoch < start_epoch:
        return 1
    else:
        return 0.5 * (1 + np.cos(4 * np.pi * (epoch - start_epoch) / (end_epoch - start_epoch)))
   
def eval_train(epoch, model,soft_labels, all_loss):    
    model.eval()
    w = annealing_weight_version1(epoch, args.annealing_start_epoch, args.annealing_end_epoch)
    # w = annealing_weight_version3(epoch, args.annealing_start_epoch, args.annealing_end_epoch)
    # w = cosine_annealing(epoch, args.annealing_start_epoch, args.annealing_end_epoch)
    # w = annealing_weight_version2(epoch, args.annealing_start_epoch, args.annealing_end_epoch)
    losses = torch.zeros(len(eval_loader.dataset))
    targets_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs = inputs.cuda()
            soft_labels_batch = soft_labels[index]
            # Convert targets to one-hot encoding for CIFAR100
            targets_one_hot = torch.zeros(targets.size(0), args.num_class).scatter_(1, targets.view(-1, 1), 1)
            targets_batch = targets_one_hot * (1 - w) + soft_labels_batch * w
            targets_all.append(targets_batch)
            targets_batch = targets_batch.cuda()
            outputs = model(inputs) 
            # ipdb.set_trace()
            loss = nn.KLDivLoss(reduction='none')(F.log_softmax(outputs, dim=1), targets_batch)
            loss_per_sample = loss.sum(dim=1)
            for b in range(inputs.size(0)):
                losses[index[b]]=loss_per_sample[b]  # save the loss for each sample.    
            # loss = CE(outputs, targets_batch)
            # for b in range(inputs.size(0)):
            #     losses[index[b]]=loss[b]  # save the loss for each sample.     
    losses = (losses-losses.min())/(losses.max()-losses.min())    # normalize the loss
    
    wandb.log({
        'epoch': epoch,
        'train/w': w
    }, step=epoch) if args.wandb else None
    
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
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)  # cluster the loss into two classes: noisy and clean. Shape: (50000,2)
    prob = prob[:,gmm.means_.argmin()]    # choose the cluster with lower mean as the clean sample. Shape: (50000,) 
    targets_all = torch.cat(targets_all, dim=0)
    return prob, all_loss, targets_all

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # ipdb.set_trace()
        Lu = torch.mean((probs_u - targets_u)**2) 

        return Lx, Lu, linear_rampup(epoch,warm_up)

class SemiLoss_CE(object):
    """
    Loss function for the conformal prediction with the Cross Entropy loss for the unlabeled data.
    """
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))
        # Lu = torch.mean((probs_u - targets_u)**2) # TODO: here, replace the mean square loss with with the KL divergence loss.
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
    else:
        if epoch%150==0 and epoch>0:  # put the original learning rate here. Just for 300 epochs.
            lr *= args.lr_decay_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def conformal_prediction_analysis(net, data_loader, q_hat, labeled_pred_idx, unlabeled_pred_idx, epoch, net_name):
    """
    Perform conformal prediction analysis on labeled and unlabeled data
    
    Args:
        net: The neural network to evaluate
        data_loader: DataLoader for all data, including the labeled and unlabeled data.
        q_hat: Quantile threshold for conformal prediction
        labeled_pred_idx: Indices of labeled predictions
        unlabeled_pred_idx: Indices of unlabeled predictions
        epoch: Current epoch number
        net_name: Name of the network (e.g., 'net1' or 'net2')
    """
    net.eval() 

    pred_all = []
    targets_all = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            pred_all.append(torch.softmax(outputs, dim=1))
            targets_all.extend(targets.cpu().numpy())
    
    pred_all = torch.cat(pred_all, dim=0)
    targets_all = torch.tensor(targets_all) # targets is the ground truth labels.
    
    # Calculate conformal prediction sets for all data
    pred_set = pred_all >= (1 - q_hat)

    # split the pred_set into labeled and unlabeled
    labeled_pred_set = pred_set[labeled_pred_idx]
    unlabeled_pred_set = pred_set[unlabeled_pred_idx]

    # calculate the prediction set sizes for labeled and unlabeled data
    labeled_pred_set_size = labeled_pred_set.sum(dim=1).float().mean()
    unlabeled_pred_set_size = unlabeled_pred_set.sum(dim=1).float().mean()
    # calculate the coverage for labeled and unlabeled data
    labeled_coverage = labeled_pred_set[np.arange(labeled_pred_set.shape[0]), targets_all[labeled_pred_idx]].float().mean()
    unlabeled_coverage = unlabeled_pred_set[np.arange(unlabeled_pred_set.shape[0]), targets_all[unlabeled_pred_idx]].float().mean()

    # Calculate calibration error for labeled and unlabeled data
    labeled_pred = pred_all[labeled_pred_idx] 
    unlabeled_pred = pred_all[unlabeled_pred_idx]
    labeled_targets = targets_all[labeled_pred_idx].cuda()  # Move to CUDA
    unlabeled_targets = targets_all[unlabeled_pred_idx].cuda()  # Move to CUDA
    
    # Initialize calibration error metrics
    labeled_calibration_error_l2 = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='l2')
    unlabeled_calibration_error_l2 = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='l2')

    labeled_calibration_error_l1 = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='l1')
    unlabeled_calibration_error_l1 = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='l1')

    labeled_calibration_error_max = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='max')
    unlabeled_calibration_error_max = MulticlassCalibrationError(num_classes=args.num_class, num_bins=20, norm='max')
    
    # Add predictions to calibration error metrics
    labeled_calibration_error_l2_value = labeled_calibration_error_l2(labeled_pred, labeled_targets)
    unlabeled_calibration_error_l2_value = unlabeled_calibration_error_l2(unlabeled_pred, unlabeled_targets)

    labeled_calibration_error_l1_value = labeled_calibration_error_l1(labeled_pred, labeled_targets)
    unlabeled_calibration_error_l1_value = unlabeled_calibration_error_l1(unlabeled_pred, unlabeled_targets)

    labeled_calibration_error_max_value = labeled_calibration_error_max(labeled_pred, labeled_targets)
    unlabeled_calibration_error_max_value = unlabeled_calibration_error_max(unlabeled_pred, unlabeled_targets)
 
    print(f"\n{net_name} Prediction Set Sizes:")
    print(f"Labeled Prediction Set Size: {labeled_pred_set_size:.3f}")
    print(f"Unlabeled Prediction Set Size: {unlabeled_pred_set_size:.3f}")
    print(f"Labeled Coverage: {labeled_coverage:.3f}")
    print(f"Unlabeled Coverage: {unlabeled_coverage:.3f}")
    print(f"Labeled Calibration Error (L2): {labeled_calibration_error_l2_value:.3f}")
    print(f"Unlabeled Calibration Error (L2): {unlabeled_calibration_error_l2_value:.3f}")
    print(f"Labeled Calibration Error (L1): {labeled_calibration_error_l1_value:.3f}")
    print(f"Unlabeled Calibration Error (L1): {unlabeled_calibration_error_l1_value:.3f}")
    print(f"Labeled Calibration Error (Max): {labeled_calibration_error_max_value:.3f}")
    print(f"Unlabeled Calibration Error (Max): {unlabeled_calibration_error_max_value:.3f}")

    wandb.log({
        'epoch': epoch,
        f'cp/{net_name}_labeled_pred_set_size': labeled_pred_set_size,
        f'cp/{net_name}_unlabeled_pred_set_size': unlabeled_pred_set_size,
        f'cp/{net_name}_labeled_coverage': labeled_coverage,
        f'cp/{net_name}_unlabeled_coverage': unlabeled_coverage,
        f'calibration/{net_name}_labeled_calibration_error_l2': labeled_calibration_error_l2_value,
        f'calibration/{net_name}_unlabeled_calibration_error_l2': unlabeled_calibration_error_l2_value,
        f'calibration/{net_name}_labeled_calibration_error_l1': labeled_calibration_error_l1_value,
        f'calibration/{net_name}_unlabeled_calibration_error_l1': unlabeled_calibration_error_l1_value,
        f'calibration/{net_name}_labeled_calibration_error_max': labeled_calibration_error_max_value,
        f'calibration/{net_name}_unlabeled_calibration_error_max': unlabeled_calibration_error_max_value,
    }, step=epoch) if args.wandb else None


def generate_soft_label(model, dataloader, q_hat):
    model.eval()
    soft_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, _, _) in enumerate(dataloader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            # Conformal prediction: mask and renormalize
            mask = (probs >= (1 - q_hat)).float()
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-8)  # TODO: here, we set the soft labels as the average of the two networks.
            soft = mask / mask_sum
            soft_labels.append(soft.cpu())
    soft_labels = torch.cat(soft_labels, dim=0)
    return soft_labels

warm_up = args.warm_up_epochs

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,noise_file=args.noise_file, annotator=args.annotator)

calibration_loader = dataloader.cifar_calibration_dataloader(args.dataset, root_dir=args.data_path, 
                                                             mode='calibration', batch_size=args.batch_size, 
                                                             num_workers=5, noise_file=args.calibration_file, 
                                                             annotator=args.annotator, clean_or_noisy=args.clean_or_noisy).run()

print('****** Building net ******')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

if args.cp_loss == 'ce': # using the Cross Entropy loss for the unlabeled data.
    criterion = SemiLoss_CE()
elif args.cp_loss == 'mse':
    criterion = SemiLoss()
else:
    raise ValueError('Loss function not supported.')

warmup_checkpoint = os.path.join(args.project_name, f'{args.dataset}_{args.annotator}_{warm_up - 1}_warmup.pth')
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = 0
if os.path.exists(warmup_checkpoint):
    print(f'Loading warmup model from {warmup_checkpoint}')
    checkpoint = torch.load(warmup_checkpoint)
    net1.load_state_dict(checkpoint['net1'])
    net2.load_state_dict(checkpoint['net2'])
    optimizer1.load_state_dict(checkpoint['optimizer1'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])
    start_epoch = checkpoint['epoch'] + 1

# checkpoint = os.path.join(args.project_name, args.resume_checkpoint)
if args.resume_checkpoint is not None:
    checkpoint = os.path.join(args.project_name, args.resume_checkpoint)
    print(f'Loading checkpoint from {checkpoint}')
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    net1.load_state_dict(checkpoint['net1'])
    net2.load_state_dict(checkpoint['net2'])
    optimizer1.load_state_dict(checkpoint['optimizer1'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])
    start_epoch = checkpoint['epoch'] + 1

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

best_acc = 0
best_acc_after_sf = 0
# Check the directory of saving models.
if not os.path.exists(args.project_name):
    os.makedirs(args.project_name)

all_loss = [[],[]] # save the history of losses from two networks#

test_loader = loader.run('test')
eval_loader = loader.run('eval_train')
train_conformal_loader = loader.run('train_conformal')

for epoch in range(start_epoch, args.num_epochs+1):   
    adjust_learning_rate(args, optimizer1, epoch)
    adjust_learning_rate(args, optimizer2, epoch)        

    q_hat1, q_hat2, q_hat_aver = calibration(net1, net2)
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader)
        if epoch == warm_up - 1:
            torch.save({
                'epoch': epoch,
                'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict()
            }, warmup_checkpoint)
            print('\nSaving Warmup Model to %s \n' % warmup_checkpoint)

    else:
        # Before the training, replace the labels of all datasets. 
        # For the training datasets, we replace all the labels with the average predictions of conformal prediction sets.
        soft_labels_net1 = generate_soft_label(net1, eval_loader, q_hat1) # net1 
        soft_labels_net2 = generate_soft_label(net2, eval_loader, q_hat2)
        # I think we should seperate the soft labels into two parts: the net1 and net2.
        
        # ipdb.set_trace()
        loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, 
                                             batch_size=args.batch_size,num_workers=5,
                                             root_dir=args.data_path,noise_file=args.noise_file, 
                                             annotator=args.annotator)

        prob1,all_loss[0], targets_all1=eval_train(epoch, net1, soft_labels_net1, all_loss[0])   # The probability is calculated when evaluating. 
        prob2,all_loss[1], targets_all2=eval_train(epoch, net2, soft_labels_net2, all_loss[1])   # Use the all train_data and the noisy labels.        
               
        pred1 = (prob1 > args.p_threshold)      # The threshold is set to 0.5 except for the CIFAR-10 dataset r = 0.9.
        pred2 = (prob2 > args.p_threshold)      # The list of pred only contains the True or False.
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader, labeled_pred_idx, unlabeled_pred_idx = loader.run('train',pred2,prob2, targets_all2) # co-divide
        conformal_prediction_analysis(net2, train_conformal_loader, q_hat2, labeled_pred_idx, unlabeled_pred_idx, epoch, 'net2')
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, q_hat2) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader, labeled_pred_idx, unlabeled_pred_idx = loader.run('train',pred1,prob1, targets_all1)
        conformal_prediction_analysis(net1, train_conformal_loader, q_hat1, labeled_pred_idx, unlabeled_pred_idx, epoch, 'net1')    
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, q_hat1)
    
    # Calculate calibration predictions every 10 epochs
    if epoch % 5 == 0:
        print(f"\nCalculating calibration predictions at epoch {epoch}")
        test(epoch, net1, net2)
    
    if epoch > warm_up and epoch % 10 == 0:
        # Use wandb run name if wandb is enabled, else fallback to args.project_name
        if args.wandb and wandb.run is not None:
            checkpoint_dir = os.path.join(args.project_name, wandb.run.name)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        # Try to save optimizer states if available
        torch.save({'net1': net1.state_dict(), 
                    'net2': net2.state_dict(),
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'epoch': epoch}, checkpoint)
        
        print(f'\nSaving Checkpoint to {checkpoint}\n')
    


