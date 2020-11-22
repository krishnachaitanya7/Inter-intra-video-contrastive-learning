import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchsummary import summary
from lib.utils import AverageMeter
import matplotlib.pyplot as plt
from datasets.ucf101 import UCF101Dataset
from datasets.hmdb51 import HMDB51Dataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
import cv2
from time import sleep


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data.item()

    return n_correct_elems / batch_size


def diff(x):
    shift_x = torch.roll(x, 1, 2)
    return_tensor = x - shift_x
    return_tensor.requires_grad_(True)
    return return_tensor  # without rescaling
    # return ((x - shift_x) + 1) / 2


def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path)
    for name, params in pretrained_weights.items():
        if 'base_network' in name:
            name = name[name.find('.') + 1:]
            adjusted_weights[name] = params
            # print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


def plot_videos(video1, video2):
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.canvas.draw()
    plt.show(block=False)
    # You would be receiving videos of the shape CxWxH
    # And you need to convert them to WxHxC for plotting purposes

    # Let's show a black image at first
    video1 = video1.clone().cpu().data.numpy()
    video2 = video2.clone().cpu().data.numpy()
    video1 = np.moveaxis(video1, 0, -1)
    video2 = np.moveaxis(video2, 0, -1)
    black_image = np.zeros(video1.shape[1:], dtype=np.float)
    im1 = ax1.imshow(black_image)
    im2 = ax2.imshow(black_image)
    for frame1, frame2 in zip(video1, video2):
        im1.set_data(frame1)
        im2.set_data(frame2)
        f.canvas.draw()
        sleep(0.1)


class GradCam:
    def __init__(self, model):
        self.model = model

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        """
        Grad Cam Call Function. This function generates grad cam output
        Parameters
        ----------
        input: [Batch Size x Number of clip length(16 here) length Sub-Videos Possible x Channels
        x Clip Length x Width x Height].
        Here Number of clip length(16 here) length Sub-Videos Possible means that if you have 192 frames and the clip
        length you are going to input to the model is 16, then totally int(192/16) = 10 sub videos are possible
        index. Also in our case for this specific implementation the batch size is always 1, as we are inputting only
        one video.
        index: The output class int of each Sub Video clip
        Returns
        -------
        [Clip Length x Channels x W x H] array with OpenCV Viridis heat map superimposed on original input which can be
        played as a video

        """
        output = self.model(diff(input))
        features = self.model.features
        if not index:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        index = index.cpu().data.numpy()
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        grads_val = self.model.gradients

        target = features
        # Below line will just generate for first clip
        # TODO: make it a for loop
        target = target.cpu().data.numpy()[0, :]
        # TODO: Just taking first clip
        # Put it in a for loop
        # Convert gradient to numpy
        grads_val = grads_val.cpu().data.numpy()
        weights = np.mean(grads_val, axis=(3, 4))[0, :]
        cam = np.zeros((input.shape[2], *target.shape[2:]), dtype=np.float32)

        # The reshaping should be done only when you wanna
        # time frames aren't 16
        if input.shape[2] != target.shape[1]:
            weights = weights.reshape(input.shape[2], -1)
            target = target.reshape(input.shape[2], -1, target.shape[2], target.shape[3])
        for i, w in enumerate(weights):
            for j, w in enumerate(w):
                cam[i] += w * target[i, j, :, :]

        cam = np.maximum(cam, 0)
        # out_cam =
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def test(args, model, criterion, test_dataloader):
    # torch.set_grad_enabled(False)
    # We need to comment this for Grad Cam
    model.eval()
    if args.modality == 'res':
        print("[Warning]: using residual frames as input")
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    grad_cam = GradCam(model=model)
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        rgb_clips, u_clips, v_clips, targets, _ = data
        # Here I am gonna go ahead and assume that batch size isn't 16, we are just inputting one video
        # and that's what I am interested in. Nothing else. Nada!
        # Hence I will do rgb_clips = rgb_clips[0] to get first index, and also if you just pass one vide
        # you will see the shape[0] would be 1, so that should check out
        if args.modality == 'u':
            sampled_clips = u_clips
        elif args.modality == 'v':
            sampled_clips = v_clips
        else:  # rgb and res
            sampled_clips = rgb_clips
        sampled_clips = sampled_clips.cuda()
        targets = targets.cuda()
        outputs = []
        for clips in sampled_clips:
            # inputs = clips.float().cuda()
            # forward
            if args.modality == 'res':
                o = model(diff(clips))
            else:
                o = model(clips)
            o = torch.mean(o, dim=0)
            outputs.append(o)
            # Apply Gradcam
            mask = grad_cam(diff(clips), targets)
        outputs = torch.stack(outputs)
        # loss = criterion(outputs, targets)
        if i == 1:
            all_outputs = outputs
            all_targets = targets
        else:
            all_outputs = torch.cat((all_outputs, outputs), dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)
        # compute loss and acc
        print(f"Current i is {i}")
        break

        # total_loss += loss.item()
        # acc = calculate_accuracy(outputs, targets)
        # accuracies.update(acc, inputs.size(0))
        # print('Test: [{}/{}], {acc.val:.3f} ({acc.avg:.3f})'.format(i, len(test_dataloader), acc=accuracies), end='\r')
    # avg_loss = total_loss / len(test_dataloader)
    # print('\n[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, accuracies.avg))
    # return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune 3D CNN from pretrained weights')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='2', help='dataset split')
    parser.add_argument('--testsplit', type=str, default='4', help='dataset split')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ft_lr', type=float, default=1e-3, help='finetune learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--model_dir', type=str, default='./ckpt/', help='path to save model')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=150, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--modality', default='res', type=str, help='modality from [rgb, res, u, v]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    # Uncomment to fix all parameters for reproducibility
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

    ########### model ##############
    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51

    if args.model == 'c3d':
        model = C3D(with_classifier=True, num_classes=class_num).cuda()
    elif args.model == 'r3d':
        model = R3DNet(layer_sizes=(1, 1, 1, 1), with_classifier=True, num_classes=class_num).cuda()
        print(model)
    elif args.model == 'r21d':
        model = R2Plus1DNet(layer_sizes=(1, 1, 1, 1), with_classifier=True, num_classes=class_num).cuda()
    # pretrained_weights = load_pretrained_weights(args.ckpt)
    pretrained_weights = torch.load(args.ckpt)
    if args.mode == 'train':
        model.load_state_dict(pretrained_weights['model'], strict=False)
    else:
        # model.load_state_dict(pretrained_weights['model'], strict=True)
        model.load_state_dict(pretrained_weights, strict=True)

    # summary(model, (3, 16, 112, 112))

    if args.desp:
        exp_name = '{}_{}_cls_{}_{}'.format(args.model, args.modality, args.desp, time.strftime('%m%d'))
    else:
        exp_name = '{}_{}_cls_{}'.format(args.model, args.modality, time.strftime('%m%d'))
    print(exp_name)
    model_dir = os.path.join(args.model_dir, exp_name)
    if not os.path.isdir(model_dir) and args.mode == 'train':
        os.makedirs(model_dir)

    train_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.RandomCrop(112),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor()
    ])

    if args.dataset == 'ucf101':
        train_dataset = UCF101Dataset('data/ucf101', args.cl, args.split, True, train_transforms)
        test_dataset = UCF101Dataset('data/ucf101', args.cl, args.testsplit, False, test_transforms)
        val_size = 800
    elif args.dataset == 'hmdb51':
        train_dataset = HMDB51Dataset('data/hmdb51', args.cl, args.split, True, train_transforms)
        test_dataset = HMDB51Dataset('data/hmdb51', args.cl, args.split, False, test_transforms)
        val_size = 400

    # split val for 800 videos
    # train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - val_size, val_size))
    # print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
    # train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
    #                               num_workers=args.workers, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
    #                             num_workers=args.workers, pin_memory=True)

    ### loss funciton, optimizer and scheduler ###
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if
                    'linear' not in name and 'conv5' not in name and 'conv4' not in name]},
        {'params': [param for name, param in model.named_parameters() if
                    'linear' in name or 'conv5' in name or 'conv4' in name], 'lr': args.ft_lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)
    #
    # prev_best_val_loss = float('inf')
    # prev_best_model_path = None
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    test(args, model, criterion, test_dataloader)
