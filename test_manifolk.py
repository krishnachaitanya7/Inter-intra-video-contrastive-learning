import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from sqlite_utils import SQLDb
import torchvision
import torchvision.transforms as transforms
import lib.custom_transforms as custom_transforms
from sklearn.manifold import TSNE
import os
import argparse
import time

import models
import datasets
import math
import pandas as pd
import tensorboard_logger as tb_logger

from lib.NCEAverage import NCEAverage, NCEAverage_ori
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion, NCESoftmaxLoss
from lib.utils import AverageMeter  # , adjust_learning_rate

from datasets.ucf101 import UCF101Dataset
from datasets.ucf101_manifolk import UCF101DatasetManifolk
from datasets.hmdb51 import HMDB51Dataset
from models.c3d import C3D
from models.r21d import R2Plus1DNet
from models.r3d import R3DNet

from torch.utils.data import DataLoader, random_split

from gen_neg import preprocess
import random
import numpy as np
import ast
import copy


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--bs', type=int, default=1, help='batch_size for test')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--testsplit', type=str, default='2', help='dataset split')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='r3d', choices=['r3d', 'c3d', 'r21d'])
    parser.add_argument('--softmax', type=ast.literal_eval, default=True)
    parser.add_argument('--nce_k', type=int, default=1024)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=512, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51'])

    # specify folder
    # parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--ckpt', type=str,
                        default="/home/shivababa/PycharmProjects/Inter-intra-video-contrastive-learning/ckpt/r3d_res_repeat_cls.pth",
                        help='checkpoint path')
    parser.add_argument('--tb_path', type=str, default='./logs/', help='path to tensorboard')

    # add new views
    parser.add_argument('--debug', type=ast.literal_eval, default=True)
    parser.add_argument('--modality', type=str, default='res', choices=['rgb', 'res', 'u', 'v'])
    parser.add_argument('--intra_neg', type=ast.literal_eval, default=True)
    parser.add_argument('--neg', type=str, default='repeat', choices=['repeat', 'shuffle'])
    # parser.add_argument('--desp', type=str)
    parser.add_argument('--seed', type=int, default=632)

    opt = parser.parse_args()

    # if opt.intra_neg:
    #     print('[Warning] using intra-negative')
    #     opt.model_name = 'intraneg_{}_{}_{}'.format(opt.model, opt.modality, time.strftime('%m%d'))
    # else:
    #     print('[Warning] using baseline')
    #     opt.model_name = '{}_{}_{}'.format(opt.model, opt.modality, time.strftime('%m%d'))
    #
    # opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    # if not os.path.isdir(opt.model_folder):
    #     os.makedirs(opt.model_folder)
    #
    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    return opt


def id_to_label(x: int):
    class_idx_path = "./data/ucf101/split/classInd.txt"
    label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1][x + 1]
    return label


def set_model(args, n_data):
    # set the model
    if args.model == 'c3d':
        model = C3D(with_classifier=False)
    elif args.model == 'r3d':
        model = R3DNet(layer_sizes=(1, 1, 1, 1), with_classifier=True)
    elif args.model == 'r21d':
        model = R2Plus1DNet(layer_sizes=(1, 1, 1, 1), with_classifier=False)

    if args.intra_neg:
        contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    else:
        contrast = NCEAverage_ori(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)

    criterion_1 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_2 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    # GPU mode
    model = model.cuda()
    contrast = contrast.cuda()
    criterion_1 = criterion_1.cuda()
    criterion_2 = criterion_2.cuda()
    cudnn.benchmark = True

    return model, contrast, criterion_1, criterion_2


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def diff(x):
    shift_x = torch.roll(x, 1, 2)
    return ((x - shift_x) + 1) / 2


def test_manifolk(test_model):
    torch.set_grad_enabled(False)
    # We need to comment this for Grad Cam
    global epoch
    test_model.eval()
    all_tsne_features = []
    all_original_labels = []
    all_predicted_labels = []
    all_uids = []
    for i, data in enumerate(test_dataloader):
        # get inputs
        # Below you get rgb_clips with shape [Batch Size x Sub Videos x Channels x Clip Length x Width x Height]
        rgb_clips, u_clips, v_clips, targets, _, vid_uid = data
        rgb_clips = rgb_clips.cuda()
        rgb_clips = rgb_clips[0]
        i = 1
        for input_clip in rgb_clips:  # input_clip: [3, 16, 112, 112], input: [10, 3, 16, 112, 112]
            # Now our model accepts only 5D input. The input is 5D input, but the first dimension
            # is number of sub videos. which might be 10 in our case. Well we just wanna send one input
            # to the model for to calculate output. Hence we are looping through input to get each subclip.
            # But as told before if we loop through 5D input, it changes to 4D input, now we gotta take it back to
            # 5D input by using pytorch unsqueeze, that's what I am gonna do in my next line
            # Now the dimensions of input becomes [1 x Channels x Clip Length x Width x Height]
            input_clip = torch.unsqueeze(input_clip, 0)  # input_clip: [1, 3, 16, 112, 112]
            # This is a specific to this model. The inputs I gotta pass comes from the diff function.
            # Also as the model requires cuda inputs, we are converting everything to cuda arrays
            # Shape of output ideally would be 1x101
            output = test_model(diff(input_clip))  # output: [1, 101]
            # Now find the highest index of the output array, because that's the class that the model predicted
            out = torch.argmax(output, dim=1).cpu().data.numpy()[0]  # out = argmax int like 3 or 79 < 101
            # index = out  # same as out
            # Below we save the model's conv5 features into a variable
            features = test_model.tsne_layer_features[0].cpu().data.numpy()
            all_tsne_features.append(features)
            all_original_labels.append(id_to_label(targets.cpu().data.numpy()[0]))
            all_predicted_labels.append(id_to_label(int(out)))
            all_uids.append(f"{vid_uid[0]}_part{i}")
            break
    X_numpy = np.array(all_tsne_features, dtype=np.float32)
    X_embedded = TSNE(n_components=3).fit_transform(X_numpy)
    log_db.insert(epoch, X_embedded, all_original_labels, all_predicted_labels, all_uids)
    torch.set_grad_enabled(True)


def main():
    if not torch.cuda.is_available():
        raise Exception('Only support GPU mode')
    # parse the args
    args = parse_option()
    print(vars(args))

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    ''' Old version
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    '''
    # Fix all parameters for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # '''

    print('[Warning] The training modalities are RGB and [{}]'.format(args.modality))

    # Data
    train_transforms = transforms.Compose([
        transforms.Resize((128, 171)),  # smaller edge to 128
        transforms.RandomCrop(112),
        transforms.ToTensor()
    ])
    if args.dataset == 'ucf101':
        trainset = UCF101Dataset('./data/ucf101/', split='2', transforms_=train_transforms)
    else:
        trainset = HMDB51Dataset('./data/hmdb51/', transforms_=train_transforms)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor()
    ])
    test_dataset = UCF101DatasetManifolk('data/ucf101', args.cl, "2", False, test_transforms)
    global test_dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                 num_workers=1, pin_memory=True)

    n_data = trainset.__len__()

    # set the model
    model, contrast, criterion_1, criterion_2 = set_model(args, n_data)

    # set the optimizer
    # optimizer = set_optimizer(args, model)

    # optionally resume from a checkpoint
    # args.start_epoch = 1
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #         args.start_epoch = checkpoint['epoch'] + 1
    #         model.load_state_dict(checkpoint['model'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         contrast.load_state_dict(checkpoint['contrast'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #         del checkpoint
    #         torch.cuda.empty_cache()
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    # logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    #
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[45, 90, 125, 160], gamma=0.2)
    # routine
    global epoch
    epoch = 155
    pretrained_weights = torch.load(args.ckpt)
    model.load_state_dict(pretrained_weights, strict=True)
    test_manifolk(model)


if __name__ == '__main__':
    log_db = SQLDb(table_name="iic")
    test_dataloader = None
    epoch = None
    main()
    log_db.close_connection()
