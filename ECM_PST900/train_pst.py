import os
import shutil
import json
import time

from apex import amp
import apex
import copy

import numpy as np
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

#from toolbox import MscCrossEntropyLoss
from toolbox.lovasz_losses import lovasz_softmax
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
#from toolbox import Ranger
import torch_optimizer as optim
from toolbox import setup_seed
from toolbox import load_ckpt
from toolbox import group_weight_decay
#from toolbox import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, ProbOhemCrossEntropy2d, FocalLoss2d, \
#    LovaszSoftmax, LDAMLoss
from toolbox.loss import *

setup_seed(33)


def run(args):
    torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})'

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')


    model = get_model(cfg)
    device = torch.device(f'cuda:{args.cuda}')
    model.to(device)

    if args.resume is None:
        pass
    else:
        print('Load checkpoint successfully!')
        model = load_ckpt(args.resume, model)


    trainset, _, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    # val_loader = DataLoader(valset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
    #                         pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)

    params_list = model.parameters()
    optimizer = optim.Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)


    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)


    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0

    amp.register_float_function(torch, 'sigmoid')
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)


    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].to(device)
                label = sample['label'].to(device)
                binary_label = sample['binary_label'].to(device)
                targets = [label, binary_label]
                predict = model(image)
            else:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                binary_label = sample['binary_label'].to(device)
                targets = [label, binary_label]
                predict = model(image, depth)
            loss = train_criterion(predict, targets)
            ####################################################

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)

        # test
        with torch.no_grad():
            model.eval()
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    [d,e] = model(image, depth)[-1]
                    predict = d


                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)

        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        test_avg = test_miou#(test_macc + test_miou) / 2

        logger.info(
            f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')
        if test_avg > best_test:
           best_test = test_avg
           save_ckpt(logdir, model)
           # only for best ckpt
           print('Saving checkpoint successfully!!!')
        # save each ckpt
        # save_ckpt(logdir, model, prefix=str(ep+1)+'.')

   


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/ECM.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgbt', choices=['rgb', 'rgbt'])
    parser.add_argument("--resume", type=str, default=None,
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=0, help="set cuda device id")

    args = parser.parse_args()

    run(args)
