# Author: David Harwath
import argparse
import os
import pickle
import sys
import time
import torch

import dataloaders
import models
from steps import train, validate

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='',
        help="training data json")
parser.add_argument("--data-val", type=str, default='',
        help="validation data json")
parser.add_argument("--exp-dir", type=str, default="",
        help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="sgd",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=100, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100,
        help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="Davenet",
        help="audio model architecture", choices=["Davenet"])
parser.add_argument("--image-model", type=str, default="VGG16",
        help="image model architecture", choices=["VGG16"])
parser.add_argument("--pretrained-image-model", action="store_true",
    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--simtype", type=str, default="MISA",
        help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])

args = parser.parse_args()

resume = args.resume

if args.resume:
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        args = pickle.load(f)
args.resume = resume
        
print(args)

train_loader = torch.utils.data.DataLoader(
    dataloaders.ImageCaptionDataset(args.data_train),
    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloaders.ImageCaptionDataset(args.data_val, image_conf={'center_crop':True}),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

audio_model = models.Davenet()
image_model = models.VGG16(pretrained=args.pretrained_image_model)

if not bool(args.exp_dir):
    print("exp_dir not specified, automatically creating one...")
    args.exp_dir = "exp/Data-%s/AudioModel-%s_ImageModel-%s_Optim-%s_LR-%s_Epochs-%s" % (
        os.path.basename(args.data_train), args.audio_model, args.image_model, args.optim,
        args.lr, args.n_epochs)

if not args.resume:
    print("\nexp_dir: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

train(audio_model, image_model, train_loader, val_loader, args)
