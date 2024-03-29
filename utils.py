
import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
import yaml
import matplotlib
import matplotlib.pyplot as plt

LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, hyp, optimizer, iter, power=1.5):
    lr = round(hyp['lr'] * (1 - iter / args.max_iters) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def train(args, train_loader, student,  criterion, optimizer, iter, scaler, teacher, uncertainty_rate, verbose=True, teacher_update=False):
    student.train()
    print("iter: ", iter)
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    if verbose and not teacher_update:
        LOGGER.info(('\n' + '%18s' * 4) % ('Iter','TverskyLoss','FocalLoss' ,'SupervisedLoss'))
        pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    elif verbose:
        LOGGER.info(('\n' + '%18s' * 4) % ('Iter','SupervisedLoss','UnsupervisedLoss' ,'TotalLoss'))
        pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')

    for i, (_, night_img, input, target) in pbar:
        optimizer.zero_grad()
        if args.onGPU == True:
            input = input.cuda().float() / 255.0   
            night_img = night_img.cuda().float() / 255.0  
        output = student(input)

        if not teacher_update:
            with torch.cuda.amp.autocast():
                focal_loss,tversky_loss,sup_loss = criterion(output,target, uncertainty_rate, teacher_update=False)
            scaler.scale(sup_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if verbose:
                pbar.set_description(('%18s' * 1 + '%18.4g' * 3) %
                                     (f'{iter}/{args.max_iters - 1}', tversky_loss, focal_loss, sup_loss.item()))
        else:
            teacher.update(student)
            output_night = student(night_img)

            with torch.no_grad():
                target_night = teacher.ema(night_img)
                
            with torch.cuda.amp.autocast():
                _,_,sup_loss = criterion(output,target, uncertainty_rate, teacher_update=False)
                _,_,unsup_loss = criterion(output_night, target_night, uncertainty_rate, teacher_update=True)
                loss = sup_loss + 0.3*unsup_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if verbose:
                pbar.set_description(('%18s' * 1 + '%18.4g' * 3) %
                                        (f'{iter}/{args.max_iters - 1}', sup_loss.item(), unsup_loss.item(),loss.item()))
        
@torch.no_grad()
def val(val_loader = None, model = None):

    model.eval()

    DA=SegmentationMetric(3)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    
    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)

    for i, (_, _, input, target) in pbar:        
        input = input.cuda().float() / 255.0   
        input_var = input
        target_var = target

        #produce pseudo labels
        with torch.no_grad():
            output = model(input_var)
        out_da = output
        target_da = target_var
        _,da_predict=torch.max(out_da, 1)
        _,da_gt=torch.max(target_da, 1)

        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())

        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc,input.size(0))
        da_IoU_seg.update(da_IoU,input.size(0))
        da_mIoU_seg.update(da_mIoU,input.size(0))

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    return da_segment_result

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])