import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math
from PIL import Image
from night_aug import NightAug
from collections import OrderedDict


class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, valid, subset_size):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''

        self.Tensor = transforms.ToTensor()
        self.valid=valid
        self.night_aug = NightAug()
        
        if self.valid:
            self.root='../datasets/uda_dataset_clear/val/night/images'
            self.names=os.listdir(self.root)
        else:
            self.root='../datasets/uda_dataset_clear/train'
            self.names = os.listdir(os.path.join(self.root,'day/images'))
            self.names_night = os.listdir(os.path.join(self.root,'night/images'))
            self.names = random.sample(self.names, subset_size)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        :train_day: 1 
        :label_day: 1 (1 for segment, 2 for lane)
        :train_night: 2
        :val_night: 3
        :
        '''
        W_=640
        H_=360

        if not self.valid:
            img_name=os.path.join(self.root+'/day/images',self.names[idx])
            img = cv2.imread(img_name)
            img = self.night_aug.aug(img)
            target_img = cv2.imread(os.path.join(self.root+'/night/images',self.names_night[idx]))
        else:
            img_name = os.path.join(self.root,self.names[idx])
            img = cv2.imread(img_name)


        label1 = cv2.imread(img_name.replace("images","colormap_seg").replace("jpg","png"))
        label1 = cv2.resize(label1, (W_, H_))      #Directly&Alternative
        img = cv2.resize(img, (W_, H_))

        if not self.valid:
            target_img = cv2.resize(target_img, (W_, H_))

        main_lane=np.where(label1==(86,94,219),255,0).astype(np.uint8)
        sub_lane=np.where(label1==(219,211,86),255,0).astype(np.uint8)
        br_lane=np.where(label1==(0,0,0),255,0).astype(np.uint8)
        main_lane = self.Tensor(main_lane)
        sub_lane = self.Tensor(sub_lane)
        br_lane = self.Tensor(br_lane)

        seg_da = torch.stack((br_lane[0],sub_lane[0],main_lane[0]),0)

        img = np.array(img)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        if not self.valid:
            target_img = np.array(target_img)
            target_img = target_img[:, :, ::-1].transpose(2, 0, 1)
            target_img = np.ascontiguousarray(target_img)

        return img_name, torch.from_numpy(target_img) if not self.valid else torch.from_numpy(img), torch.from_numpy(img), (seg_da)

if __name__ == '__main__':
    from tqdm import tqdm

    valLoader = torch.utils.data.DataLoader(
        MyDataset(valid=True, subset_size=3),
        batch_size=8, shuffle=False, num_workers=12, pin_memory=True)
    
    total_batches = len(valLoader)
    pbar = enumerate(valLoader)
    pbar = tqdm(pbar, total=total_batches)

    for i, (_1, _2, input, target) in pbar:
        print(_1)
        break

