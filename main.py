# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import PIL
import h5py
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vqgan.load_dataset import get_VgSceneGraphDataset
from vqgan.vqmodel.vqgan import VQModel
from vqgan.losses.lpips import LPIPS
from vqgan.losses.vqperceptual import VQLPIPSWithDiscriminator
from vqgan.discriminator.model import NLayerDiscriminator, weights_init
import os
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm


def data_to(sample, device):
    for i,s in enumerate(sample):
        sample[i] = s.to(device)
    return sample


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def train_VQModel(args=None):
    d_lr = args['d_lr']
    g_lr = args['g_lr']
    
    train_data, vocab, num_classes = get_VgSceneGraphDataset()
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=8)
    train_loader = sample_data(train_loader)
    device = torch.device('cuda:0')
    
    netG = VQModel(**args['VQConfig']).to(device)
    #loss = VQLPIPSWithDiscriminator(**args['lossconfig'])
    netD = NLayerDiscriminator(**args['DiscriminatorConfig']).apply(weights_init).to(device)
    P_Loss = LPIPS().to(device).eval()
    
    #initialize optimizer
    gen_params = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            gen_params += [{'params':[value], 'lr':g_lr}]
    g_optimizer = torch.optim.Adam(gen_params, betas=(0.5, 0.9))
    
    dis_params = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_params += [{'params':[value], 'lr':d_lr}]
    d_optimizer = torch.optim.Adam(dis_params, betas=(0.5, 0.9))
    
    pbar = range(args['total_epoch'])
    pbar = tqdm(pbar, initial=args['start_epoch'], dynamic_ncols=True, smoothing=0.01)
    for idx in pbar:
        data = next(train_loader)
        images, objects, boxes = data_to(data, device)

        # update D network
        netD.zero_grad()
        images_rec, qloss = netG(images)
        logits_real = netD(images.contiguous().detach())
        logits_fake = netD(images_rec.contiguous().detach())
        disc_factor = adopt_weight(1.0, idx, 5000)  # 在第一万轮之前不更新Discriminator
        d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)
        d_loss.backward()
        d_optimizer.step()
        
        # update G network
        netG.zero_grad()
        ploss = P_Loss(images.contiguous(), images_rec.contiguous())
        rec_loss = torch.abs(images.contiguous() - images_rec.contiguous()) + ploss
        rec_loss = torch.mean(rec_loss)
        logits_fake = netD(images_rec.contiguous())
        g_loss = -torch.mean(logits_fake) * disc_factor + rec_loss + qloss.mean()
        g_loss.backward()
        g_optimizer.step()
        
        pbar.set_description(
            (
                f'd:{d_loss:.4f};g:{g_loss:.4f}'
            )
        )
        
        if idx % 100 == 0:
            save_image(images, f"ori_{idx}")
            save_image(images_rec, f"rec_{idx}")
        if idx % 5000 == 0:
            state_dict = {
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'netG_optim': g_optimizer.state_dict(),
                'netD.optim': d_optimizer.state_dict()
            }
            torch.save(state_dict, os.path.join(args['model_path'], f'GD_{idx}.pth'))


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (torch.mean(F.softplus(-logits_real)) +
                    torch.mean(F.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def mkdir(dir):
    if os.path.exists(dir):
        return
    else:
        try:
            os.mkdir(dir)
        except FileNotFoundError:
            mkdir(os.path.split(dir)[0])
            os.mkdir(dir)


def save_image(img, im_name):
    image = make_grid(img.cpu().data, nrow=4).permute(1, 2, 0).contiguous().numpy()
    image = Image.fromarray(((image + 1.0) * 127.5).astype(np.uint8))
    image.save(os.path.join(args['image_path'], f'{im_name}.png'))


args = {
    'd_lr': 0.0001,
    'g_lr': 0.0001,
    'batch_size': 4,
    'start_epoch': 0,
    'total_epoch': 30000,
    'image_path': './output/images',
    'model_path': './output/models',
    'VQConfig': {
        'embed_dim': 256,
        'n_embed': 1024,
        'edconfig': {
            'double_z': False,
            'z_channels': 256,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1,1,2,2,4],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0
        },
    },
    'lossconfig': {
        'disc_conditional': False,
        'disc_in_channels': 3,
        'disc_start': 10000,
        'disc_weight': 0.8,
        'codebook_weight': 1.0
    },
    'DiscriminatorConfig': {
        'input_nc': 3,
        'ndf': 64,
        'n_layers': 3,
        'use_actnorm': False,
    },
}

mkdir(args['image_path'])
mkdir(args['model_path'])

train_VQModel(args)
