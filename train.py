import os
import torch
import pickle
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from copy import deepcopy
from loss import TotalLoss
import network
import numpy as np
import time
import random
import yaml
from pathlib import Path
import math
import torch.multiprocessing as mp
from network.ema import ModelEMA
from network._deeplab import DeepLabHead
from torchvision import models
from collections import OrderedDict

def train_net(args, hyp):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", num_gpus)
    student = network.modeling.__dict__['deeplabv3_resnet50'](num_classes=3, output_stride=8)
    if num_gpus > 1:
        student = torch.nn.DataParallel(student)

    args.savedir = args.savedir + '/'

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=False, subset_size=args.subset_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True, subset_size=args.subset_size),#12454
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        print("CUDA AVAILABLE")
        args.onGPU = True
        student = student.cuda()
        cudnn.benchmark = True

    total_paramters = netParams(student)
    print('Total network parameters of teacher/student: ' + str(total_paramters))

    criteria = TotalLoss(hyp['alpha1'], hyp['gamma1'], hyp['alpha2'], hyp['gamma2'], hyp['alpha3'], hyp['gamma3'])

    start_iter = 0
    lr = hyp['lr']

    optimizer = torch.optim.AdamW(student.parameters(), lr=hyp['lr'], betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         dict_path = args.resume
    #         dict_path 
    #         checkpoint = torch.load(args.resume)
    #         start_iter = checkpoint['iter']
    #         teacher.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (iter {})"
    #             .format(args.resume, checkpoint['iter']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    scaler = torch.cuda.amp.GradScaler()
    teacher = ModelEMA(student, num_gpus)
    teacher_update=False

    if args.only_sup:
        print("ONLY SUPERVISED")
        
    uncertainty_rate = 0.5

    for iter in range(start_iter, args.max_iters):
        teacher_file_name = args.savedir + os.sep + 'teacher_{}.pth'.format(iter)
        student_file_name = args.savedir + os.sep + 'student_{}.pth'.format(iter)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))
        print("Uncertainty Rate: ", uncertainty_rate)

        poly_lr_scheduler(args,hyp,optimizer, iter)

        # train for 50k iter
        student.train()
        if not args.only_sup:
            if iter+1<args.max_iters//2:
                train(args, trainLoader, student, criteria, optimizer, iter, scaler, teacher, uncertainty_rate , args.verbose, teacher_update=False)
            elif iter+1==args.max_iters//2:
                teacher.update(student, keep_rate=0.00)
                train(args, trainLoader, student, criteria, optimizer, iter, scaler, teacher , uncertainty_rate, args.verbose, teacher_update=True)
            else:
                train(args, trainLoader, student, criteria, optimizer, iter, scaler, teacher , uncertainty_rate, args.verbose, teacher_update=True)
                uncertainty_rate+=0.02
        else:
            train(args, trainLoader, student, criteria, optimizer, iter, scaler, teacher , uncertainty_rate, args.verbose, teacher_update=False)

        
        student.eval()
        # validation
        # da_segment_results,ll_segment_results=val(valLoader, model)
        da_segment_results = val(valLoader, student) #da_mIoU_seg
        msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})'.format(
                    da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2])
        print(msg)

        torch.save(teacher.ema.state_dict(), teacher_file_name)
        torch.save(student.state_dict(), student_file_name)

        save_checkpoint({
             'iter': iter + 1,
             'teacher_state_dict': teacher.ema.state_dict(),
             'student_state_dict': student.state_dict(),
             'optimizer': optimizer.state_dict(),
             'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_iters', type=int, default=30, help='Max. number of iters')
    
    parser.add_argument('--subset_size', type=int, default=12454, help='Size of each training size')
    parser.add_argument('--only_sup', action='store_true', help='Supervised training only')
    parser.add_argument('--num_workers', type=int, default=8, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--savedir', default='./pretrained', help='directory to save the results')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='hyperparameters path')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--verbose', action='store_false', help='')

    args = parser.parse_args()
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    train_net(args, hyp.copy())
