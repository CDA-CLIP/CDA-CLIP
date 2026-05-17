import os
import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids')
parser.add_argument('--num_patches', type=int, default=196,
                    help='number of patches  (224/16) * (224/16) for ViT-B/16')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs')
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lr_clip', default=0.000001, type=float,
                    help='learning rate for any CLIP parameter that is unfrozen')
parser.add_argument('--lr_cda', default=5e-5, type=float,
                    help='learning rate for the Cross-Domain Attention module')
parser.add_argument('--lr_cda_fzCLIP', default=1e-3, type=float,
                    help='CDA learning rate when the CLIP backbone is frozen')
parser.add_argument('--warmup', default=10, type=int)
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay')
parser.add_argument('--model_saved_path', type=str, default='./models_saved')
parser.add_argument('--log_step', type=int, default=10, help='log_step')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--warm_up', default=8, type=int)

# Cross-Domain Attention hyper-parameters (see CDA.py and the manuscript).
parser.add_argument('--num_cda_layers', type=int, default=1,
                    help='Depth of the CDA module (paper ablates 1-3).')
parser.add_argument('--num_heads', type=int, default=8,
                    help='Number of attention heads in each CDA layer.')

# Dataset paths.
parser.add_argument('--INbreast_path', type=str, default='./data/INbreast.txt',
                    help='the directory of INbreast dataset')
parser.add_argument('--SIIMACR_path', type=str, default='./data/SIIMACR.txt',
                    help='the directory of SIIM-ACR dataset')
parser.add_argument('--CheXpert5x200_path', type=str,
                    default='./data/chexpert5x200.txt',
                    help='the directory of CheXpert dataset')
parser.add_argument('--ChestXray_path', type=str, default='./data/chestxray.txt',
                    help='the directory of ChestXray dataset')
parser.add_argument('--MIMICGAZE_path', type=str,
                    default='./data/MIMICGAZE_CLIP_train.txt',
                    help='the directory of MIMIC GAZE dataset')
parser.add_argument('--imagenet_path', type=str,
                    default='/media/datasets/ImageNet1K',
                    help='the directory of imagenet dataset')
parser.add_argument('--TinyImagenet_path', type=str,
                    default='/media/datasets/TinyImageNet',
                    help='the directory of tiny imagenet dataset')
parser.add_argument('--ImagenetV2', type=str,
                    default='/media/datasets/ImageNetV2-matched-frequency',
                    help='the directory of imagenetV2 dataset')
parser.add_argument('--TMED2_path', type=str, default='./data/TMED2.txt',
                    help='the directory of TMED2 dataset')

parser.add_argument('--vision_encoder', type=str, default='ViT-B/16',
                    help='RN50, RN101, RN50x4, RN50x16, RN50x64, '
                         'ViT-B/32, ViT-B/16')
opt = parser.parse_args()


print("Torch version:", torch.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
