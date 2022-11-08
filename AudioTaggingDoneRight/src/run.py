# -*- coding: utf-8 -*-
# @Time    : 3/8/22 
# @Modified by : Juncheng B Li
# @Original Author  : Yuan Gong
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset","audioset_s", "esc50", "speechcommands"])
parser.add_argument("--n_mels", type=int, default=128, help="number of mel bins")
parser.add_argument('--mean', default=-12.7508, type=float, help='normalizing mean')
parser.add_argument('--std', default=11.7584, type=float, help='normalizing std')

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.999)
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

# arguments for TAL-trans Models:
parser.add_argument('--embedding_size', type = int, default = 1024) # this is the embedding size after a pooling layer
                                                                    # after a non-pooling layer, the embeddings size will be twice this much
parser.add_argument('--n_conv_layers', type = int, default = 10)
parser.add_argument('--n_trans_layers', type = int, default = 2)
parser.add_argument('--kernel_size', type = str, default = '3')     # 'n' or 'nxm'
parser.add_argument('--n_pool_layers', type = int, default = 5)     # the pooling layers will be inserted uniformly into the conv layers
                                                                    # the should be at least 2 and at most 6 pooling layers
                                                                    # the first two pooling layers will have stride (2,2); later ones will have stride (1,2)
parser.add_argument('--batch_norm', type = bool, default = True)
parser.add_argument('--dropout', type = float, default = 0.0)
parser.add_argument('--pooling', type = str, default = 'lin', choices = ['max', 'ave', 'lin', 'exp', 'att', 'h-att', 'all'])
parser.add_argument('--continue_from_ckpt', type = str, default = None)
parser.add_argument('--addpos', type = bool, default = False)
parser.add_argument('--transformer_dropout', type = float, default = 0.5)
#psla model
parser.add_argument("--eff_b", type=int, default=0, help="which efficientnet to use, the larger number, the more complex")
parser.add_argument("--att_head", type=int, default=4, help="number of attention heads")
parser.add_argument("--att_activation", type = str, default = 'sigmoid', choices = ['sigmoid','softmax','relu','linear'])
args = parser.parse_args()


print('now train a audio spectrogram transformer model')
# dataset spectrogram mean and std, used to normalize the input audioset_s yun feature:#-12.75089158, 11.75840071 gong yuan audioset: -4.2677393, 4.5689974
norm_stats = {'audioset':[args.mean, args.std], 'audioset_s':[args.mean, args.std], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
target_length = {'audioset':1024, 'audioset_s':400, 'esc50':512, 'speechcommands':128}
# if add noise for data augmentation, only use for speech commands
noise = {'audioset': False, 'audioset_s': False, 'esc50': False, 'speechcommands':True}

audio_conf = {'num_mel_bins': args.n_mels, 'target_length': target_length[args.dataset], 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1],
              'noise':noise[args.dataset]}
val_audio_conf = {'num_mel_bins': args.n_mels, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False}

if args.bal == 'bal':
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
if 'x' not in args.kernel_size:
    args.kernel_size = args.kernel_size + 'x' + args.kernel_size
args.kernel_size = tuple(int(x) for x in args.kernel_size.split('x'))
# transformer based model
if args.model == 'ast':
    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.n_mels,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size='base384')
elif args.model == 'fnet':
    audio_model = models.get_fnet()
elif args.model == 'TALtrans':
    args.target_length = target_length[args.dataset]
    audio_model = models.TransformerEncoder(args)
elif args.model == 'TAL':
    audio_model = models.TALNet(args)
elif args.model == 'efficientnet':
    audio_model = models.EffNetAttention(att_act=args.att_activation, label_dim=args.n_class, b=args.eff_b, pretrain=args.imagenet_pretrain, head_num=args.att_head)
elif args.model == 'resnet':
    args.target_length = target_length[args.dataset]
    audio_model = models.ResNetAttention(args)
elif args.model == 'mbnet':
    audio_model = models.MBNet(label_dim=args.n_class, pretrain=args.imagenet_pretrain)
elif args.model == 'linear':
    audio_model = models.LinearModel(n_layers=3, input_dim=args.n_mels, hidden_dim=128, label_dim=args.n_class)
elif args.model == 'mmt':
    audio_model = models.MMTEncoder(args)
print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    # best model on the validation set
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

