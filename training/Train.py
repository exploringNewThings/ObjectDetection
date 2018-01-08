import sys
import argparse
import os
import shutil
import time
import numpy as np
import scipy.misc

import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks.inception import inception_v3
from networks.SampleNetwork import SampleNetwork
from criterion.BaseLoss import BaseLoss
#from dataset import InputDataset
from tests import InputDataset_test as InputDataset
from dataset.readers import COCOReader
from transforms import individualfliphorizontal, imagefliphorizontal, brightness
from losses.CustomLoss import BoundingBoxLoss
from transforms.normalize import Normalize

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--map-freq', '-mf', default=1000, type=int,
                    metavar='N', help='output map save frequency (default: 1000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    '''
    if args.pretrained:
        model = inception_v3(pretrained=True)
    else:
        model = inception_v3()
    '''
    model = SampleNetwork()
    model = nn.DataParallel(model)
    model = model.cuda()
    #model.double().cuda()
    #model = torch.nn.parallel.DistributedDataParallel(model)
    

    # define loss function (criterion) and optimizer
    criterion = BoundingBoxLoss()#nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    '''
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    '''
    #FORMAT = '%(asctime)s : %(pathname)s : %(module)s : %(message)s'
#logging.basicConfig(level=logging.DEBUG, filename="test.log", filemode='w',format=FORMAT)

    reader = COCOReader.COCOReader(categories='person')
    df = reader.readfile("/data/stars/user/uujjwal/datasets/generic-detection/mscoco/annotations/instances_train2014.json", imagedir="/data/stars/user/uujjwal/datasets/generic-detection/mscoco/train2014")

    print(df.shape)
    reader.setsavepath('/data/stars/user/uujjwal/mscocogt')

    df = reader.addsaveinfo(df)

    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    fliptransform = imagefliphorizontal.ImageFlipHorizontal(prob=0.99)
    brighttransform = brightness.ObjectBrightness(prob=0.999, random_factor = 0.6)

    train_dataset = InputDataset.InputDataset(df, transforms=[normalize])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    '''
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''
    '''
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    '''
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
       # prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1
        #best_prec1 = max(prec1, best_prec1)
        #save_checkpoint({
        #    'epoch': epoch + 1,
        #    'arch': args.arch,
        #    'state_dict': model.state_dict(),
        #    'best_prec1': best_prec1,
        #    'optimizer' : optimizer.state_dict(),
        #}, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    #losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    file = open('/home/agoel/object_detection/Beta/training/loss.txt','w')
    for i, sample_batched in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_image_batch = sample_batched['image']
        bbox_batch = sample_batched['annotations']
        #print(len(input_image_batch))
        #print(bbox_batch)
        #print(bbox_batch[0])
        #print(bbox_batch[0][0])
        filename_batch = sample_batched['file_name']
        
        input_type = type(input_image_batch[0])
        print(input_type)
        target = bbox_batch#.cuda(async=True)
        input_var = input_image_batch.cuda()#torch.autograd.Variable(torch.IntTensor(input_image_batch)).cuda()
        target_var = target#torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        #print(input_image_batch.shape)
        loss,feat_map = criterion(output, target_var, sample_batched['orig_size'])
        
        #losses.update(loss,1)
        
        file.write(str(i)+'\t'+str(loss.data[0])+'\n')
        
        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        #losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))
        #top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % args.map_freq == 0:
            for batch in range(args.batch_size):
                filename = os.path.splitext(filename_batch[batch])[0]
                filename = filename[-27:]
                #np_output = output[batch].data.cpu().numpy()
                #print(filename)
                np.save('/home/agoel/object_detection/Beta/training/feature_maps/'+filename+'_featuremap_itr_'+str(i)+'.npy',feat_map)
                #scipy.misc.toimage(feat_map, cmin=0, cmax=...).save('/home/agoel/object_detection/Beta/training/feature_maps/'+filename+'_featuremap_itr_'+str(i)+'.jpg')
            
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time))
            print('Loss {}'.format(loss.data[0]))
    file.close()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
