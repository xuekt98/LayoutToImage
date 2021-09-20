import torch
from torch.utils.data import DataLoader

from dataio import VgSceneGraphDataset
from model.transformer import TransformerEncoder
from torch.autograd import Variable
from model.vec2mask import VecToMask
from model.generator_meta import G_obj

import pdb

def data_to(sample, device):
		for i, s in enumerate(sample):
				sample[i] = s.to(device)
		return sample


def train(args=None):
		h5_path = '../transLayout/data/train.h5'
		import json
		with open('../transLayout/data/vocab.json', 'r') as f:
			vocab = json.load(f)
		image_dir = '../transLayout/data/images/'
		train_data = VgSceneGraphDataset(vocab, h5_path, image_dir, image_size=(256, 256),
								 normalize_images=True, max_objects=args['max_objs'], max_samples=None,
								 include_relationships=True, use_orphaned_objects=True,
								 left_right_flip=False)
		train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=8)
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
				'd_style': 512,
				'd_v2gh': 256,
				'l_v2gh': 3,
				'd_v2mh': 256,
				'l_v2mh': 3,
				'n_gmlp': 8,
				'n_emlp': 4,
				}
train(args)