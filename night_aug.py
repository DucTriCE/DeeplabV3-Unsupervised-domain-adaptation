import torch
import torchvision.transforms as T
import numpy as np
from numpy import random as R
import cv2

class NightAug:
    def __init__(self):
        self.gaussian = T.GaussianBlur(11,(0.1,2.0))
    def mask_img(self,img,cln_img):
        while R.random()>0.4:
            x1 = R.randint(img.shape[1])
            x2 = R.randint(img.shape[1])
            y1 = R.randint(img.shape[2])
            y2 = R.randint(img.shape[2])
            img[:,x1:x2,y1:y2]=cln_img[:,x1:x2,y1:y2]
        return img

    def gaussian_heatmap(self,x):
        """
        It produces single gaussian at a random point
        """
        sig = torch.randint(low=1,high=150,size=(1,))[0]
        image_size = x.shape[1:]
        center = (torch.randint(image_size[0],(1,))[0], torch.randint(image_size[1],(1,))[0])
        x_axis = torch.linspace(0, image_size[0]-1, image_size[0]) - center[0]
        y_axis = torch.linspace(0, image_size[1]-1, image_size[1]) - center[1]
        xx, yy = torch.meshgrid(x_axis, y_axis)
        kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
        new_img = (x*(1-kernel) + 255*kernel).type(torch.uint8)
        return new_img

    def aug(self,x):
        img = x[:,:,::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        # img = img.cuda()
        g_b_flag = True
        # Guassian Blur
        if R.random()>0.5:
            img = self.gaussian(img)
        
        cln_img_zero = img.detach().clone()

        # Gamma
        if R.random()>0.5:
            cln_img = img.detach().clone()
            val = 1/(R.random()*0.8+0.2)
            img = T.functional.adjust_gamma(img,val)
            img= self.mask_img(img,cln_img)
            g_b_flag = False
        
        # Brightness
        if R.random()>0.5 or g_b_flag:
            cln_img = img.detach().clone()
            val = R.random()*0.8+0.2
            img = T.functional.adjust_brightness(img,val)
            img= self.mask_img(img,cln_img)

        # Contrast
        if R.random()>0.5:
            cln_img = img.detach().clone()
            val = R.random()*0.8+0.2
            img = T.functional.adjust_contrast(img,val)
            img= self.mask_img(img,cln_img)
        img= self.mask_img(img,cln_img_zero)

        prob = 0.5
        while R.random()>prob:
            img=self.gaussian_heatmap(img)
            prob+=0.1

        # Noise
        if R.random()>0.5:
            n = torch.clamp(torch.normal(0,R.randint(50)+0.1,img.shape),min=0)
            img = n + img
            img = torch.clamp(img,max = 255).type(torch.uint8)
        img = img.cpu().numpy()[::-1,:,:].transpose(1,2,0)
        return img
        
if __name__ == '__main__':
    path = '/home/ceec/tri/TwinVast/TwinLiteNet_done2/images/bdc7ed93-19e4b08c.jpg'
    img = cv2.imread(path)
    transformer = NightAug()
    img = transformer.aug(img)
    success = cv2.imwrite('./ok.jpg', img)
