from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import ipdb
import wandb

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], annotator='', soft_labels=None): 
        self.r = r
        self.mode = mode
        self.transform = transform  
        self.soft_labels = soft_labels
        # add the code for downloading the dataset.
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if dataset=='cifar10' and not os.path.exists('%s/cifar-10-batches-py'%root_dir):
            os.system('wget -O %s/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'%root_dir)
            os.system('tar -xzvf %s/cifar-10-python.tar.gz -C %s'%(root_dir, root_dir))
        if dataset=='cifar100' and not os.path.exists('%s/cifar-100-python'%root_dir):
            os.system('wget -O %s/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'%root_dir)
            os.system('tar -xzvf %s/cifar-100-python.tar.gz -C %s'%(root_dir, root_dir))
             
            # there is another way to download the dataset using torchvision.datasets.
            # datasets.CIFAR10(root_dir, train=True, download=True)

        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_clean_label=[] # clean label is the list.
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_clean_label = train_clean_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))


            if dataset == 'cifar10': 
                multi_rater = torch.load(os.path.join(root_dir, noise_file))
                noise_label = multi_rater[annotator]
            elif dataset == 'cifar100': 
                multi_rater = torch.load(os.path.join(root_dir, noise_file))
                noise_label = multi_rater[annotator]
                self.train_clean_label = multi_rater['clean_label']
                train_data = train_data[multi_rater['indices']]
                
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'train_conformal':
                self.train_data = train_data
                # self.train_clean_label = train_clean_label
                self.train_clean_label = multi_rater['clean_label']
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]  # why it looks like this?   
                    pred_idx = (1-pred).nonzero()[0]  # why it looks like this?   
                    # pred_idx = np.where(~pred)[0]                                    
                    pred_idx = (1-pred).nonzero()[0]  # why it looks like this?                                      
                    # pred_idx = np.where(~pred)[0]                                    
                
                self.train_data = train_data[pred_idx]
                if self.soft_labels is not None:
                    self.noise_label = [self.soft_labels[i] for i in pred_idx]
                else:
                    self.noise_label = [noise_label[i] for i in pred_idx]

                if self.mode == 'labeled':
                    if self.soft_labels is not None:
                        # For soft labels, we can't directly compare with clean labels
                        # Instead, we can check if the predicted class (argmax) matches the clean label
                        noise_label_array = np.array(self.noise_label)
                        predicted_classes = np.argmax(noise_label_array, axis=1)
                        clean_labels_array = np.array([self.train_clean_label[i] for i in pred_idx])
                        clean_in_labeled = np.sum(predicted_classes == clean_labels_array)
                    else:
                        # For hard labels, direct comparison
                        clean_in_labeled = np.sum(np.array(self.noise_label) == np.array([self.train_clean_label[i] for i in pred_idx]))
                    ratio_clean_in_labeled = clean_in_labeled / len(pred_idx)
                    print(f'clean_in_labeled: {clean_in_labeled} : {len(self.train_clean_label)}, ratio: {ratio_clean_in_labeled:.3f}')
                    if wandb.run is not None:
                        wandb.log({
                            "labeled/clean_in_labeled": clean_in_labeled,
                            "labeled/total_in_labeled": len(pred_idx),
                            "labeled/ratio_clean_in_labeled": ratio_clean_in_labeled
                        })
                elif self.mode == 'unlabeled':
                    if self.soft_labels is not None:
                        # For soft labels, we can't directly compare with clean labels
                        # Instead, we can check if the predicted class (argmax) matches the clean label
                        noise_label_array = np.array(self.noise_label)
                        predicted_classes = np.argmax(noise_label_array, axis=1)
                        clean_labels_array = np.array([self.train_clean_label[i] for i in pred_idx])
                        clean_in_unlabeled = np.sum(predicted_classes == clean_labels_array)
                    else:
                        # For hard labels, direct comparison
                        clean_in_unlabeled = np.sum(np.array(self.noise_label) == np.array([self.train_clean_label[i] for i in pred_idx]))
                    ratio_clean_in_unlabeled = clean_in_unlabeled / len(pred_idx)
                    print(f'clean_in_unlabeled: {clean_in_unlabeled} : {len(self.train_clean_label)}, ratio: {ratio_clean_in_unlabeled:.3f}')
                    if wandb.run is not None:
                        wandb.log({
                            "unlabeled/clean_in_unlabeled": clean_in_unlabeled,
                            "unlabeled/total_in_unlabeled": len(pred_idx),
                            "unlabeled/ratio_clean_in_unlabeled": ratio_clean_in_unlabeled
                        })
                self.pred_idx = pred_idx                           
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)  # why return two images?
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            if self.soft_labels is not None:
                target = self.soft_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index    
        elif self.mode == 'train_conformal':
            img, target = self.train_data[index], self.train_clean_label[index]
            img = Image.fromarray(img)
            img = self.transform(img) # use the test transform.
            return img, target, index
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, noise_file='', annotator=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.annotator = annotator
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    


        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,pred=[],prob=[], soft_labels=None):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, annotator=self.annotator)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob, annotator=self.annotator, soft_labels=soft_labels)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred, annotator=self.annotator, soft_labels=soft_labels)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader, labeled_dataset.pred_idx, unlabeled_dataset.pred_idx
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file, annotator=self.annotator)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader
        
        elif mode=='train_conformal':
            train_conformal_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='train_conformal', noise_file=self.noise_file, annotator=self.annotator)      
            train_conformal_loader = DataLoader(
                dataset=train_conformal_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return train_conformal_loader

class cifar_calibration_dataset(Dataset):
    def __init__(self, dataset, root_dir, transform, mode, noise_file='', annotator='', clean_or_noisy='clean'):
        self.mode = mode
        self.transform = transform
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.annotator = annotator
        
        if self.mode == 'calibration':
            if dataset == 'cifar100':
                train_dic = unpickle('%s/train'%root_dir)
                self.cali_data = train_dic['data']
                self.cali_data = self.cali_data.reshape((50000, 3, 32, 32))
                self.cali_data = self.cali_data.transpose((0, 2, 3, 1))
                # self.cali_label = train_dic['fine_labels']# TODO: change the label to noisy labels.
                multi_rater = torch.load(os.path.join(root_dir, noise_file))
                self.indices = torch.load((os.path.join(root_dir, noise_file)))['indices']  
                # Convert lists to numpy arrays before indexing
                self.cali_data = np.array(self.cali_data)[self.indices]
                if clean_or_noisy == 'clean':
                    self.cali_label = np.array(multi_rater['clean_label'])
                elif clean_or_noisy == 'noisy':
                    self.cali_label = np.array(multi_rater[self.annotator])
                else:
                    raise ValueError('Clean or noisy not supported')
                
            else:
                raise ValueError('Dataset not supported')
        else:
            raise ValueError('Mode not supported')

    def __getitem__(self, index):
        img, target = self.cali_data[index], self.cali_label[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.cali_data)
    
class cifar_calibration_dataloader():
    def __init__(self, dataset, root_dir, mode, batch_size, num_workers, noise_file='', annotator='', clean_or_noisy=''):
        self.dataset = dataset
        self.root_dir = root_dir
        self.mode = mode
        self.noise_file = noise_file
        self.annotator = annotator
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
        self.clean_or_noisy = clean_or_noisy

    def run(self):
        if self.mode == 'calibration':
            if self.dataset == 'cifar100':
                cali_dataset = cifar_calibration_dataset(dataset=self.dataset, root_dir=self.root_dir, 
                                                         transform=self.transform_test, mode='calibration', 
                                                         noise_file=self.noise_file, annotator=self.annotator, clean_or_noisy=self.clean_or_noisy)
                cali_loader = DataLoader(
                    dataset=cali_dataset, 
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers)
                return cali_loader
        else:
            raise ValueError('Mode not supported')