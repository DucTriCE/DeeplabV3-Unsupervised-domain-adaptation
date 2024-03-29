import torch
import torch
from network import TwinLite2 as net
import torch.backends.cudnn as cudnn
import DataSetval as myDataLoader
from argparse import ArgumentParser
from utils import val, netParams
import torch.optim.lr_scheduler
from const import *
from loss import TotalLoss

import numpy as np
import time
import random
import yaml
from pathlib import Path


def validation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    # load the model
    model = net.TwinLiteNet(args)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        # model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
        
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], hyp["hgain"], hyp["sgain"], hyp["vgain"], valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    
    model.load_state_dict(torch.load(args.weight))
    # model.half()
    model.eval()
    example = torch.rand(16, 3, 384, 640).cuda().half() if args.half else torch.rand(16, 3, 384, 640).cuda()
    model = torch.jit.trace(model.half() if args.half else model, example)

    da_segment_results,ll_segment_results = val(valLoader, model,args.half)

    msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    print(msg)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--weight', default="pretrained_ema/small.pth")
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='hyperparameters path')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--type', default="nano", help='')
    parser.add_argument('--is320', action='store_true')
    parser.add_argument('--seda', action='store_true', help='sigle encoder for Drivable Segmentation')
    parser.add_argument('--sell', action='store_true', help='sigle encoder for Lane Segmentation')

    # args = parser.parse_args()
    validation(parser.parse_args())


