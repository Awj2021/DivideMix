from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import ipdb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class twitter_dataset(Dataset): 
    def __init__(self, root, train_noise_file, transform, mode, pred=None, probability=None, paths=None, annotator=''): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}

        train_json_file = os.path.join(self.root, train_noise_file)  # Firstly let us use the rand-3. (rand-3 / rand-4)
        test_json_file = os.path.join(self.root, 'test_LDL.json')
        if self.mode == 'test':
            with open(test_json_file, 'r') as f:
                test_json = json.load(f)
                for key, entry in test_json.items():
                    image_name = entry['image_name']
                    img_path = os.path.join(self.root, image_name)
                    self.test_labels[img_path] = entry['majority_vote']
        else:
            with open(train_json_file, 'r') as f:
                train_json = json.load(f)
                for key, entry in train_json.items():
                    image_name = entry['image_name']
                    img_path = os.path.join(self.root, image_name)
                    self.train_labels[img_path] = entry[annotator]

        if mode == 'all':
            self.train_imgs=[]
            with open(train_json_file,'r') as f:
                train_json = json.load(f)
                for key, entry in train_json.items():
                    image_name = entry['image_name']
                    img_path = os.path.join(self.root, image_name)
                    self.train_imgs.append(img_path)
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
                test_json = json.load(f)
                for key, entry in test_json.items():
                    img_path = os.path.join(self.root, entry['image_name'])
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
        
class twitter_dataloader():  
    def __init__(self, root, noise_file, batch_size, num_workers):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.noise_file = noise_file
                   
        self.transform_train = transforms.Compose([
                transforms.Resize(232),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.RandomErasing(),               
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
                
    def run(self,mode,annotator,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = twitter_dataset(self.root, self.noise_file,transform=self.transform_train, mode='all', annotator=annotator)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = twitter_dataset(self.root, self.noise_file, transform=self.transform_train, mode='labeled',pred=pred, probability=prob, paths=paths, annotator=annotator)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)           
            unlabeled_dataset = twitter_dataset(self.root,self.noise_file, transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths, annotator=annotator)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = twitter_dataset(self.root, self.noise_file, transform=self.transform_test, mode='all', annotator=annotator)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
        elif mode=='test':
            test_dataset = twitter_dataset(self.root,self.noise_file,transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader             