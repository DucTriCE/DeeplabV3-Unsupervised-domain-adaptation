import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import os
import torch
from argparse import ArgumentParser
import cv2

def Run(model,root,imgName):
    image_name = os.path.join(root,imgName)
    img = cv2.imread(image_name)
    img = cv2.resize(img, (500, 500))
    img_rs=img.copy()
    img_ori=img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
        
    x0=img_out
    # x1=img_out[1]

    # x0,x1=x0[:,:,12:-12],x1[:,:,12:-12]

    _,da_predict=torch.max(x0, 1)
    # _,ll_predict=torch.max(x1, 1)

    # print(da_predict.size(),ll_predict.size())

    DA = da_predict.byte().cpu().data.numpy()[0]*100
    # LL = ll_predict.byte().cpu().data.numpy()[0]*255
    img_rs[DA==200]=[86,94,219]
    img_rs[DA==100]=[219,211,86]
    # img_rs[LL==255]=[6,247,109]

    img_rs = img_rs[24:-24,:,:]

    label1 = cv2.imread(image_name.replace("images","colormaps").replace("jpg","png"))
    # label2 = cv2.imread(image_name.replace("images","lane").replace("jpg","png"))
    label1 = cv2.resize(label1, (500, 500))
    # label2 = cv2.resize(label2, (1280, 720))

    # _,seg2 = cv2.threshold(label2,1,255,cv2.THRESH_BINARY)
    main_lane=np.where(label1==(86,94,219),255,0).astype(np.uint8)
    sub_lane=np.where(label1==(219,211,86),255,0).astype(np.uint8)

    # img_ori = img_ori[24:-24,:,:]
    # print(main_lane.shape,img_ori.shape)
    img_ori[main_lane[:,:,0]==255]=[86,94,219]
    img_ori[sub_lane[:,:,0]==255]=[219,211,86]
    # img_ori[seg2[:,:,0]==255]=[6,247,109]

    
    return img_rs,img_ori

parser = ArgumentParser()


parser.add_argument('--seda', action='store_true', help='sigle encoder for Drivable Segmentation')
parser.add_argument('--sell', action='store_true', help='sigle encoder for Lane Segmentation')
parser.add_argument('--type', default="large", help='')
args = parser.parse_args()
import network

model = network.modeling.__dict__['deeplabv3_resnet50'](num_classes=3, output_stride=8)
model = model.cuda()
model.load_state_dict(torch.load('/home/ceec/tri/ssh_data/student_29.pth'))
model.eval()

root='/home/ceec/huycq/TwinVast_1/bdd100k/images/val'
image_list=os.listdir(root)
shutil.rmtree('results')
os.mkdir('results')

import shutil
for i, imgName in enumerate(image_list):
    img_pd,img_ori=Run(model,root,imgName)
    # shutil.copy2(os.path.join(root,imgName), os.path.join('images',imgName))
    cv2.imwrite(os.path.join('results',imgName),img_pd)
    cv2.imwrite(os.path.join('results',imgName.replace(".","_gt.")),img_ori)