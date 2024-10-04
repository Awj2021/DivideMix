from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import ipdb

class chaoyang_dataset(Dataset): 
    def __init__(self, root, transform, mode, pred=None, probability=None, paths=None, annotator=''): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}

        train_json_file = os.path.join(self.root, 'json', 'train_label.json')
        test_json_file = os.path.join(self.root, 'json', 'test_ori.json')

        if mode == 'test':
            with open(test_json_file, 'r') as f:
                test_json = json.load(f)
                for entry in test_json:
                    img_path = os.path.join(self.root, entry['name'])
                    self.test_labels[img_path] = entry['label']
        else:
            with open(train_json_file, 'r') as f:
                train_json = json.load(f)
                # ipdb.set_trace()
                for entry in train_json:
                    img_path = os.path.join(self.root, entry['name'])
                    self.train_labels[img_path] = entry[annotator]

        if mode == 'all':
            self.train_imgs=[]
            with open(train_json_file,'r') as f:
                train_json = json.load(f)
                for entry in train_json:
                    img_path = os.path.join(self.root, entry['name'])
                    self.train_imgs.append(img_path)
            # ipdb.set_trace()
            random.shuffle(self.train_imgs)
        
        elif self.mode == "labeled":
            train_imgs = paths 
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == "unlabeled":  
            train_imgs = paths 
            pred_idx = (1-pred).nonzero()[0]  
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                    
                         
        elif mode=='test':
            self.test_imgs = []
            with open(test_json_file,'r') as f:
                # lines = f.read().splitlines()
                test_json = json.load(f)
                # ipdb.set_trace()
                for entry in test_json:
                    img_path = os.path.join(self.root, entry['name'])
                    self.test_imgs.append(img_path)      
        else:
            raise ValueError("Invalid mode")
                    
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path, index       
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        else:
            return len(self.train_imgs)            
        
class chaoyang_dataloader():  
    def __init__(self, root, batch_size, num_workers, annotator):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.annotator = annotator
                   
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])        

    def run(self,mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = chaoyang_dataset(self.root,transform=self.transform_train, mode='all', annotator=self.annotator)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = chaoyang_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths, annotator=self.annotator)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = chaoyang_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths, annotator=self.annotator)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = chaoyang_dataset(self.root,transform=self.transform_test, mode='all', annotator=self.annotator)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
        elif mode=='test':
            test_dataset = chaoyang_dataset(self.root,transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader             