import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import random

import copy
import shutil

import importlib
from tqdm import tqdm
import argparse
import torchvision.utils as vutils
import torch


#import lpips

def lpips_fn():
    return lpips.LPIPS(net='alex')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='zhang_exp1')
    parser.add_argument('--config', type=str, default='')
    args, unparsed = parser.parse_known_args()
    return args

def backup_files(args):
    # back up files
    shutil.copyfile('configs/'+args.config+'.py', os.path.join('checkpoints', args.exp, 'config.py'))
    shutil.copyfile('models/inpaint.py', os.path.join('checkpoints', args.exp, 'inpaint.py'))
    shutil.copyfile('models/memory.py', os.path.join('checkpoints', args.exp, 'memory.py'))
    shutil.copyfile('models/squid.py', os.path.join('checkpoints', args.exp, 'squid.py'))
    shutil.copyfile('models/discriminator.py', os.path.join('checkpoints', args.exp, 'discriminator.py'))
    shutil.copyfile('models/basic_modules.py', os.path.join('checkpoints', args.exp, 'basic_modules.py'))
    shutil.copyfile('main.py', os.path.join('checkpoints', args.exp, 'main.py'))
    shutil.copyfile('tools.py', os.path.join('checkpoints', args.exp, 'tools.py'))

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_disc(CONFIG):
    DISC = importlib.import_module('models.discriminator')

    if CONFIG.discriminator_type == 'basic':
        discriminator = DISC.SimpleDiscriminator(CONFIG.num_in_ch, size=CONFIG.size, num_layers=CONFIG.num_layers)
        print('Basic discriminator created.')

    elif CONFIG.discriminator_type == 'tiny':
        discriminator = DISC.TinyDiscriminator(CONFIG.num_in_ch, size=CONFIG.size)
        print('Tiny discriminator created.')

    elif CONFIG.discriminator_type == 'big':
        discriminator = DISC.BigDiscriminator(CONFIG.num_in_ch, size=CONFIG.size)
        print('Big discriminator created.')

    elif CONFIG.discriminator_type == 'extratiny':
        discriminator = DISC.ExtraTinyDiscriminator(CONFIG.num_in_ch, size=CONFIG.size)
        print('ExtraTiny discriminator created.')

    elif CONFIG.discriminator_type == 'patch':
        discriminator = DISC.PatchDiscriminator(CONFIG.num_in_ch)
        print('PatchGAN discriminator created.')

    elif CONFIG.discriminator_type == 'spectral':
        discriminator = DISC.SpectralDiscriminator(CONFIG.num_in_ch)
        print('SpectralNorm discriminator created.')

    else:
        discriminator = DISC.SimpleDiscriminator(CONFIG.num_in_ch, size=CONFIG.size, num_layers=CONFIG.num_layers)
        print('Unknown type, defaulting to Basic discriminator.')

    return discriminator


def log(log_file, msg):
    log_file.write(msg+'\n')

def log_loss(log_file, epoch, train_loss, val_loss):
    msg = 'Epoch: %d [TRAIN]' % epoch
    for k, v in train_loss.items():
        msg += ' %s: %.4f' % (k, v)
    msg += ' [VAL]'
    for k, v in val_loss.items():
        msg += ' %s: %.4f' % (k, v)
    log(log_file, msg)

def save_image(path, data, prefix="img"):
    os.makedirs(path, exist_ok=True)  # path artık sadece klasör olacak
    for idx, tensor in enumerate(data):
        filename = os.path.join(path, f"{prefix}_{idx:03d}.png")
        
        # Eğer filename klasörse, hata çıkmasın
        if os.path.isdir(filename):
            print(f"[Warning] {filename} is a directory. Skipping...")
            continue

        save_tensor_as_image(tensor, filename)


def save_tensor_as_image(tensor, path):
    tensor = tensor.clone().detach().cpu()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    # Önlem: Eğer aynı isimde klasör varsa silinemez; o yüzden hata almamak için uyarı
    if os.path.isdir(path):
        print(f"[Warning] Cannot save image, a directory exists: {path}")
        return

    vutils.save_image(tensor, path)



