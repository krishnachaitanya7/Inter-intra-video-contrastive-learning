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
from datasets.ucf101_manifolk import UCF101DatasetManifolk
from datasets.hmdb51 import HMDB51Dataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
import cv2
from time import sleep
from sqlite_utils import SQLDb
from sklearn.manifold import TSNE


def show_cam_on_image(img, mask):
    # Because for colormap to apply correctly, the image needs to be in BGR. By default each frame of ours is in
    # RGB, so we change the order
    img = img[..., ::-1]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_RAINBOW)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def id_to_label(x: int):
    class_idx_path = "./data/ucf101/split/classInd.txt"
    label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1][x + 1]
    return label

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
    black_image = np.zeros(video1.shape[1:], dtype=np.float)
    im1 = ax1.imshow(black_image)
    im2 = ax2.imshow(black_image)
    for frame1, frame2 in zip(video1, video2):
        im1.set_data(frame1)
        im2.set_data(frame2)
        f.canvas.draw()
        sleep(0.1)

def test_manifolk(test_model, test_dataloader):
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
        j = 1
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
            all_uids.append(f"{vid_uid[0]}_part{j}")
            j += 1
        print(f"Current video number is: {i}")
    X_numpy = np.array(all_tsne_features, dtype=np.float32)
    X_embedded = TSNE(n_components=3).fit_transform(X_numpy)
    log_db.insert(155, X_embedded, all_original_labels, all_predicted_labels, all_uids)
    torch.set_grad_enabled(True)



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
    parser.add_argument('--ckpt', type=str,
                        default="/home/shivababa/PycharmProjects/Inter-intra-video-contrastive-learning/ckpt/r3d_res_repeat_cls.pth",
                        help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=150, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--modality', default='res', type=str, help='modality from [rgb, res, u, v]')
    args = parser.parse_args()
    return args


def main():
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
        test_dataset = UCF101DatasetManifolk('data/ucf101', args.cl, "2", False, test_transforms)
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
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=1, pin_memory=True)
    test_manifolk(model, test_dataloader)


if __name__ == "__main__":
    log_db = SQLDb(table_name="iic")
    main()
    log_db.close_connection()
