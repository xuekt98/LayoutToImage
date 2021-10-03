import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb

class ResBlock(nn.Module):
	def __init__(self):
		super(ResBlock, self).__init__()

	def forward(self):
		pass

class ResBlocks(nn.Module):
	def __init__(self):
		super(ResBlocks, self).__init__()

	def forward(self):
		pass


class CoModConv(nn.Module):
	def __init__(self,
							 n_dl,
							 d_catin,
							 in_channels,
							 out_channels,
							 kernel_size,
							 stride=1,
							 padding=0,
							 dilation=1,
							 padding_mode='zeros'
							 ):
		super(CoModConv, self).__init__()
		self.conv = _DyConv(in_channels,
																 out_channels,
																 kernel_size,
																 stride=stride,
																 padding=padding,
																 dilation=dilation,
																 padding_mode=padding_mode)
		layers = list()
		layers.append(EqualLinear(d_catin, in_channels))
		for i in range(n_dl):
			layers.append(EqualLinear(in_channels, in_channels))

		self.dense = nn.Sequential()

	def forward(self, x, y):
		s = self.dense(torch.cat((x, y), dim=-1))

		# !!! prehold applying bias fused activ

		# !!!

		# modulate
		self.conv.assign_customized_params(s)

		# !!! prehold if applying demodulate

		# !!!

		out = self.conv(s)
		return out

class _DyConv(nn.Conv2d):
	def __init__(self,
							 in_channels,
							 out_channels,
							 kernel_size,
							 stride=1,
							 padding=0,
							 dilation=1,
							 padding_mode='zeros'
							 ):
		super(_DyConv, self).__init__(in_channels,
																 out_channels,
																 kernel_size,
																 stride=stride,
																 padding=padding,
																 dilation=dilation,
																 bias=False,
																 padding_mode=padding_mode)
		self.customized_w = None

	def forward(self, x):
		assert self.customized_w is not None
		B, C, H, W = x.shape
		x = x.view(1, -1, H, W) 		# move batch dim into channels
		out = self._conv_forward(x, self.customized_w, self.bias)
		H, W = out.shape[2:]
		out = out.view(B, -1, H, W)
		self.customized_w = None
		self.groups = 1
		return out

	def assign_customized_params(self, y):
		# y: BI, self.weight: OIkk
		B, I = y.shape
		O, k = self.weight.shape[0], self.weight.shape[-1]
		self.groups = B
		self.customized_w = self.weight.unsqueeze(0).repeat(B, 1, 1, 1, 1) * y.view(B, 1, I, 1, 1)
		self.customized_w = self.customized_w.view(-1, I, k, k)


class PixelNorm(nn.Module):
	def __init__(self):
			super().__init__()

	def forward(self, input):
			return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

# prehold
class EqualLinear(nn.Module):
	def __init__(self, d_in, d_out, activation='lrelu'):
		super(EqualLinear, self).__init__()
		self.linear = nn.Linear(d_in, d_out)
		if activation == 'lrelu':
			self.activ = nn.LeakyReLU()
		else:
			self.activ = self.dummy_activ

	def dummy_activ(self, x):
	# i.e. linear
		return x

	def forward(self, x):
		out = self.linear(x)
		out = self.activ(out)
		return out

dyconv = _DyConv(3, 64, (3, 3))
conv = nn.Conv2d(3, 64, (3, 3))
x = torch.ones((10, 3, 128, 128))
y = torch.randn(10, 3)
dyconv.assign_customized_params(y)
o1 = dyconv(x)
o2 = conv(x)