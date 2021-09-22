import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb

'''全连接层Fully Connected(FC) Block'''


class FCBlock(nn.Module):
	"""Fully Connected(FC) Block"""

	def __init__(self, in_features, out_features, hidden_features, num_hidden_layers,
							 outermost_linear=False, nonlinearity='relu', weight_init=None):
		"""
Params:
		in_features: FC输入特征维度
		out_features: FC输出特征维度
		hidden_features: FC中间隐藏层特征维度
		num_hidden_layers: FC中间隐藏层个数(不包含第一层和最后一层)
		outermost_linear: FC最后一层是否需要激活函数
		nonlinearity: 激活函数
		weight_init: FC参数初始化方式
"""
		super(FCBlock, self).__init__()

		# 存储非线性激活函数与参数初始化方式的表
		nls_and_inits = {'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
										 'sigmoid': (nn.Sigmoid(), init_weights_normal, None)}
		nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

		if weight_init is not None:
			self.weight_init = weight_init
		else:
			self.weight_init = nl_weight_init

		self.net = []  # MLP中的所有模块

		# 第一层 从输入维度到隐藏层维度
		self.net.append(nn.Sequential(
			nn.Linear(in_features, hidden_features), nl
		))

		# 中间层
		for i in range(num_hidden_layers):
			self.net.append(nn.Sequential(
				nn.Linear(hidden_features, hidden_features), nl
			))

		# 最后一层 从隐藏层维度到输出维度
		if outermost_linear:
			self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
		else:
			self.net.append(nn.Sequential(
				nn.Linear(hidden_features, out_features), nl
			))

		self.net = nn.Sequential(*self.net)

		# 初始化模型参数
		if self.weight_init is not None:
			self.net.apply(self.weight_init)
		if first_layer_init is not None:
			self.net[0].apply(first_layer_init)

	def forward(self, model_input):
		output = self.net(model_input)
		return output


# 参数初始化方式 kaiming分布初始化
def init_weights_normal(m):
	if type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


# 参数初始化方式 Xavier分布初始化
def init_weights_xavier(m):
	if type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			nn.init.xavier_normal_(m.weight)


class VecToMask(nn.Module):
	"""利用Transformer Encoder的输出生成Mask"""

	def __init__(self, in_features=512, feature_grid_size=(8, 8, 8),
							 V2G_hidden_features=256, V2G_num_hidden_layers=3,
							 Decoder_hidden_features=256, Decoder_num_hidden_layers=3,):
		super(VecToMask, self).__init__()
		self.in_features = in_features
		self.feature_grid_size = feature_grid_size
		self.V2G_hidden_features = V2G_hidden_features
		self.V2G_num_hidden_layers = V2G_num_hidden_layers
		self.Decoder_hidden_features = Decoder_hidden_features
		self.Decoder_num_hidden_layers = Decoder_num_hidden_layers

		# 从TransformerEncoder输出的vector生成图像网格
		self.V2G_net = FCBlock(in_features=in_features, out_features=np.prod(feature_grid_size),
													 hidden_features=V2G_hidden_features, num_hidden_layers=V2G_num_hidden_layers,
													 outermost_linear=True, nonlinearity='relu')

		self.Decoder_net = FCBlock(in_features=feature_grid_size[0], out_features=1,
															 hidden_features=Decoder_hidden_features, num_hidden_layers=Decoder_num_hidden_layers,
															 outermost_linear=True, nonlinearity='sigmoid') # channel wise

	def forward(self, model_input, bbox, HH, WW=None):
		device = model_input.device
		if WW is None: WW = HH
		b, o, _ = model_input.size()
		model_input = model_input.view(-1, self.in_features)
		c, h, w = self.feature_grid_size
		mask_feat = self.V2G_net(model_input).view(-1, c, h, w)

		x, y = torch.arange(0, HH).expand(b, o, -1).to(device), torch.arange(0, WW).expand(b, o, -1).to(device)
		dx, dy = ((bbox[:, :, 2] - bbox[:, :, 0])/HH).unsqueeze(-1), ((bbox[:, :, 3] - bbox[:, :, 1])/WW).unsqueeze(-1)
		x, y = bbox[:, :, 0].unsqueeze(-1) + x * dx, bbox[:, :, 1].unsqueeze(-1) + y * dy
		bbox_gird = batch_meshgrid(x.view(b*o, -1), y.view(b*o, -1))

		# mask_feat_up = torch.einsum('bchw->bhwc', crop_bbox(mask_feat, bbox.view(-1, 4), HH, WW)).contiguous().reshape(-1, c)
		mask_feat_up = F.grid_sample(mask_feat, bbox_gird,
																 mode='bilinear',
																 padding_mode='border',
																 align_corners=True)
		# mask = torch.einsum('', self.Decoder_net(mask_feat).view(b, o, h, w))

		mask = self.Decoder_net(mask_feat_up.permute(0, 2, 3, 1).contiguous()).view(-1, HH, WW).unsqueeze(1) #(b*o, 1, HH, WW)

		return mask, mask_feat_up

def batch_meshgrid(x, y):
	b = x.size(0)
	h, w = x.size(-1), y.size(-1)
	gx = x.view(b, 1, -1, 1).repeat(1, 1, 1, w)
	gy = y.view(b, 1, 1, -1).repeat(1, 1, h, 1)
	out_grid = torch.cat((gx, gy), dim=1).permute(0, 2, 3, 1).contiguous()
	return out_grid
