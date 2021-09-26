import torch
from torch.utils.data import DataLoader

from dataio import VgSceneGraphDataset
<<<<<<< HEAD
from model.generator_meta import G_NET_FULL
from model.discriminator import CombineDiscriminator128
import torch.nn as nn
from utils.losses import VGGLoss
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid
import os
import numpy as np
=======
from model.transformer import TransformerEncoder
from torch.autograd import Variable
from model.vec2mask import VecToMask
from model.generator_meta import G_obj
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301

import pdb

def data_to(sample, device):
		for i, s in enumerate(sample):
				sample[i] = s.to(device)
		return sample


<<<<<<< HEAD
def sample_data(loader):
	while True:
		for batch in loader:
			yield batch

def train(args=None):
		d_lr = 0.0001
		g_lr = 0.0001
		lamb_obj = 1.0
		lamb_img = 0.1

=======
def train(args=None):
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301
		h5_path = '../transLayout/data/train.h5'
		import json
		with open('../transLayout/data/vocab.json', 'r') as f:
			vocab = json.load(f)
<<<<<<< HEAD
			num_classes = len(vocab['object_name_to_idx'])
		image_dir = '../transLayout/data/images/'
		train_data = VgSceneGraphDataset(vocab, h5_path, image_dir, image_size=args['image_size'],
=======
		image_dir = '../transLayout/data/images/'
		train_data = VgSceneGraphDataset(vocab, h5_path, image_dir, image_size=(256, 256),
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301
								 normalize_images=True, max_objects=args['max_objs'], max_samples=None,
								 include_relationships=True, use_orphaned_objects=True,
								 left_right_flip=False)
		train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=8)
<<<<<<< HEAD
		train_loader = sample_data(train_loader)
		device = torch.device("cuda:0")

		netG = G_NET_FULL(args, num_classes).to(device)
		netD = CombineDiscriminator128(num_classes=num_classes).to(device)

		# noise1 = Variable(torch.FloatTensor(args['batch_size'], args['max_objs']+1, args['d_noise1'])).to(device)
		# noise2 = Variable(torch.FloatTensor(args['batch_size'], args['d_noise2'])).to(device)

		# initialize optimizer
		dis_parameters = []
		for key, value in dict(netD.named_parameters()).items():
			if value.requires_grad:
				dis_parameters += [{'params': [value], 'lr': d_lr}]
		d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

		gen_parameters = []
		for key, value in dict(netG.named_parameters()).items():
			if value.requires_grad:
				gen_parameters += [{'params': [value], 'lr': g_lr}]
		g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

		vgg_loss = VGGLoss()
		l1_loss = nn.L1Loss()

		pbar = range(args['iter'])
		pbar = tqdm(pbar, initial=args['start_iter'], dynamic_ncols=True, smoothing=0.01)
		for idx in pbar:
			noise1 = torch.randn(args['batch_size'], args['max_objs']+1, args['d_noise1'], requires_grad=True).to(device)
			noise2 = torch.randn(args['batch_size'], args['d_noise2'], requires_grad=True).to(device)
			data = next(train_loader)
			real_images, objects, boxes = data_to(data, device)

			# update D network
			netD.zero_grad()
			d_out_real, d_out_robj = netD(real_images, boxes, objects)
			d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
			d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()

			fake_images, mask = netG(objects=objects, boxes=boxes, latent_s=noise1, latent_a=noise2)
			d_out_fake, d_out_fobj = netD(fake_images.detach(), boxes, objects)
			d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
			d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()

			d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake)
			d_loss.backward()
			d_optimizer.step()

			# update G network
			if (idx % 1) == 0:
				netG.zero_grad()
				g_out_fake, g_out_obj = netD(fake_images, boxes, objects)
				g_loss_fake = - g_out_fake.mean()
				g_loss_obj = - g_out_obj.mean()

				pixel_loss = l1_loss(fake_images, real_images).mean()
				feat_loss = vgg_loss(fake_images, real_images).mean()

				g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + pixel_loss + feat_loss
				g_loss.backward()
				g_optimizer.step()

				pbar.set_description(
					(
						f'd: {d_loss:.4f}; g: {g_loss:.4f}'
					)
				)
			if (idx) % 1000 == 0:
				save_image(real_images, f"r_{idx}")
				save_image(fake_images, f"f_{idx}")
				save_image((mask > 0.5).long(), f"msk_{idx}")

			if (idx) % 10000 == 0:
				state_dict = {
					'netG': netG.state_dict(),
					'netD': netD.state_dict(),
					'netG_optim': g_optimizer.state_dict(),
					'netD_optim': d_optimizer.state_dict()
				}
				torch.save(state_dict, os.path.join(args['model_path'], f'GD_{idx}.pth'))

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

# 测试用参数
args = {
				'iter': 1200000,
				'start_iter': 0,
				'd_model': 128,
				'total_epoch': 1,
				'batch_size': 4,
				'image_size': (128, 128),
				'num_trans_layers': 1,
				'num_heads': 2,
				'd_ffn': 64,
				'd_noise1': 200,
				'd_noise2': 256,
				'max_objs': 5,
				'base_size': 128,
=======
		device = torch.device("cuda:0")
		image, objects, boxes = data_to(iter(train_loader).next(), device)

		gen = G_obj(args, len(vocab['object_name_to_idx'])).to(device)

		# net1 = TransformerEncoder(d_model=args['d_model'],
		# 													num_classes=len(vocab['object_name_to_idx']),
		# 													num_trans_layers=args['num_trans_layers'],
		# 													num_heads=args['num_heads'],
		# 													d_ffn=args['d_ffn'],
		# 													dropout=0.0).to(device)
		# vec2msk = VecToMask(in_features=args['d_model'], feature_grid_size=(args['d_model'], 8, 8),
		# 					 V2G_hidden_features=256, V2G_num_hidden_layers=3,
		# 					 Decoder_hidden_features=256, Decoder_num_hidden_layers=3,
		# 					 mask_image_size=(64, 64, 64)).to(device)
		#
		#
		noise1 = Variable(torch.FloatTensor(args['batch_size'], args['max_objs']+1, args['d_noise1'])).to(device)
		noise1.data.normal_(0, 1)
		noise2 = Variable(torch.FloatTensor(args['batch_size'], args['max_objs']+1, args['d_noise2'])).to(device)
		noise2.data.normal_(0, 1)

		gen(objects=objects, boxes=boxes, latent_s=noise1, latent_a=noise2)


		# output, attn_map = net1(objects, boxes, noise)
		# mask, mask_feat = vec2msk(output, boxes, args['base_size'])


		# for epoch in range(args['total_epoch']):
		# 		net1.train()
		#
		# 		for index, data in enumerate(train_loader):
		# 				image, objects, boxes = data
		# 				output = net1(objects, boxes)
		# 				# print(output)
		# 				break

# 测试用参数
args = {'d_model': 128,
				'total_epoch': 1,
				'batch_size': 4,
				'num_trans_layers': 1,
				'num_heads': 2,
				'd_ffn': 64,
				'd_noise1': 100,
				'd_noise2': 256,
				'max_objs': 10,
				'base_size':32,
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301
				'd_style': 512,
				'd_v2gh': 256,
				'l_v2gh': 3,
				'd_v2mh': 256,
				'l_v2mh': 3,
				'n_gmlp': 8,
				'n_emlp': 4,
<<<<<<< HEAD
	'image_path': './output/images',
	'model_path': './output/models'
				}

mkdir(args['image_path'])
mkdir(args['model_path'])

=======
				}
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301
train(args)