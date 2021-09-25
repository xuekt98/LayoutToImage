__all__ = ['CIPSskip',
					 'CIPSres',
					 ]

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock

import pdb

class CIPSskip(nn.Module):
	def __init__(self, size=256, ltnt_size=512, n_mlp=8, n_coaffin=3, vm_dim=256, lr_mlp=0.01,
							 activation=None, channel_multiplier=2, **kwargs):
		super(CIPSskip, self).__init__()

		self.size = size
		demodulate = True
		self.demodulate = demodulate

		self.channels = {
			0: 512,
			1: 512,
			2: 512,
			3: 512,
			4: 256 * channel_multiplier,
			5: 128 * channel_multiplier,
			6: 64 * channel_multiplier,
			7: 32 * channel_multiplier,
			8: 16 * channel_multiplier,
		}

		multiplier = 2
		in_channels = int(self.channels[0])
		self.conv1 = StyledConv(vm_dim, in_channels, 1, vm_dim, demodulate=demodulate,
														activation=activation)

		self.linears = nn.ModuleList()
		self.to_rgbs = nn.ModuleList()
		self.log_size = int(math.log(size, 2))

		self.n_intermediate = self.log_size - 1
		self.to_rgb_stride = 2
		for i in range(0, self.log_size - 1):
			out_channels = self.channels[i]
			self.linears.append(StyledConv(in_channels, out_channels, 1, vm_dim,
																		 demodulate=demodulate, activation=activation))
			self.linears.append(StyledConv(out_channels, out_channels, 1, vm_dim,
																		 demodulate=demodulate, activation=activation))
			self.to_rgbs.append(ToRGB(out_channels, vm_dim, upsample=False))

			in_channels = out_channels

		self.style_dim = vm_dim

		layers = [PixelNorm()]

		for i in range(n_mlp):
			layers.append(
				EqualLinear(
					ltnt_size, ltnt_size, lr_mul=lr_mlp, activation='fused_lrelu'
				)
			)

		self.style = nn.Sequential(*layers)

		ca_layers = [EqualLinear(vm_dim+ltnt_size, vm_dim, lr_mul=lr_mlp, activation='fused_lrelu')]
		for i in range(n_coaffin-1):
			ca_layers.append(
				EqualLinear(
					vm_dim, vm_dim, lr_mul=lr_mlp, activation='fused_lrelu'
				)
			)
		self.coaffine = nn.Sequential(*ca_layers)

	def forward(self,
							mask,
							v_code,
							mask_feat,
							latent,
							return_latents=False,
							truncation=1,
							truncation_latent=None,
							input_is_latent=False,
							return_rgb=False
							):
		# coords -> mask, latent->latent
		'''
			v_code: bod->bd (option: reshape+affine/mean) (here mean value temp currently)
			latent: bd
			mask: (b*o)1WH -> b1WH (mean o-axis-wise)
			mask_feat: bcWH
		'''
		batch_size, n_o, _, w, h = mask.shape

		# latent = latent[0]
		if truncation < 1:
			latent = truncation_latent + truncation * (latent - truncation_latent)

		if not input_is_latent:
			latent = self.style(latent)  ## generate W
			latent = self.coaffine(torch.cat((v_code.mean(dim=1), latent), dim=-1)).view(batch_size, -1)

		x = (mask_feat * mask).sum(dim=1)

		rgb = 0
		x = self.conv1(x, latent)
		for i in range(self.n_intermediate):
			for j in range(self.to_rgb_stride):
				x = self.linears[i*self.to_rgb_stride + j](x, latent)

			rgb = self.to_rgbs[i](x, latent, rgb)

		res = [x]
		if return_rgb:
			res.append(rgb)
		else:
			res.append(None)
		if return_latents:
			res.append(latent)
		else:
			res.append(None)
		return res

class CIPSskipObj(nn.Module):
	def __init__(self, size=256, ltnt_size=512, n_mlp=8, n_coaffin=3, vm_dim=256, lr_mlp=0.01,
							 activation=None, channel_multiplier=2, **kwargs):
		super(CIPSskipObj, self).__init__()

		self.size = size
		demodulate = True
		self.demodulate = demodulate

		self.channels = {
			0: 512,
			1: 512,
			2: 512,
			3: 512,
			4: 256 * channel_multiplier,
			5: 128 * channel_multiplier,
			6: 64 * channel_multiplier,
			7: 32 * channel_multiplier,
			8: 16 * channel_multiplier,
		}

		multiplier = 2
		in_channels = int(self.channels[0])
		self.conv1 = StyledConv(vm_dim, in_channels, 1, vm_dim, demodulate=demodulate,
														activation=activation)

		self.linears = nn.ModuleList()
		self.to_rgbs = nn.ModuleList()
		self.log_size = int(math.log(size, 2))

		self.n_intermediate = self.log_size - 1
		self.to_rgb_stride = 2
		for i in range(0, self.log_size - 1):
			out_channels = self.channels[i]
			self.linears.append(StyledConv(in_channels, out_channels, 1, vm_dim,
																		 demodulate=demodulate, activation=activation))
			self.linears.append(StyledConv(out_channels, out_channels, 1, vm_dim,
																		 demodulate=demodulate, activation=activation))
			self.to_rgbs.append(ToRGB(out_channels, vm_dim, upsample=False))

			in_channels = out_channels

		self.style_dim = vm_dim

		layers = [PixelNorm()]

		for i in range(n_mlp):
			layers.append(
				EqualLinear(
					ltnt_size, ltnt_size, lr_mul=lr_mlp, activation='fused_lrelu'
				)
			)

		self.style = nn.Sequential(*layers)

		ca_layers = [EqualLinear(vm_dim+ltnt_size, vm_dim, lr_mul=lr_mlp, activation='fused_lrelu')]
		for i in range(n_coaffin-1):
			ca_layers.append(
				EqualLinear(
					vm_dim, vm_dim, lr_mul=lr_mlp, activation='fused_lrelu'
				)
			)
		self.coaffine = nn.Sequential(*ca_layers)

	def forward(self,
							mask,
							v_code,
							mask_feat,
							latent,
							return_latents=False,
							truncation=1,
							truncation_latent=None,
							input_is_latent=False,
							return_rgb=False
							):
		# coords -> mask, latent->latent
		'''
			v_code: bd
			latent: bd
			mask: b1hw
			mask_feat: bchw
		'''


		batch_size, _, w, h = mask.shape

		# latent = latent[0]
		if truncation < 1:
			latent = truncation_latent + truncation * (latent - truncation_latent)

		if not input_is_latent:
			latent = self.style(latent)  ## generate W
			latent = self.coaffine(torch.cat((v_code, latent), dim=-1)).view(batch_size, -1)


		x = mask_feat * mask

		rgb = 0
		x = self.conv1(x, latent)
		for i in range(self.n_intermediate):
			for j in range(self.to_rgb_stride):
				x = self.linears[i*self.to_rgb_stride + j](x, latent) * mask

			rgb = self.to_rgbs[i](x, latent, rgb)

		res = [x]
		if return_rgb:
			res.append(rgb)
		else:
			res.append(None)
		if return_latents:
			res.append(latent)
		else:
			res.append(None)
		return res


class CIPSres(nn.Module):
	def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
							 activation=None, channel_multiplier=2, **kwargs):
		super(CIPSres, self).__init__()

		self.size = size
		demodulate = True
		self.demodulate = demodulate
		self.lff = LFF(int(hidden_size))
		self.emb = ConstantInput(hidden_size, size=size)

		self.channels = {
			0: 512,
			1: 512,
			2: 512,
			3: 512,
			4: 256 * channel_multiplier,
			5: 128 * channel_multiplier,
			6: 64 * channel_multiplier,
			7: 64 * channel_multiplier,
			8: 32 * channel_multiplier,
		}

		self.linears = nn.ModuleList()
		in_channels = int(self.channels[0])
		multiplier = 2
		self.linears.append(StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
																	 activation=activation))

		self.log_size = int(math.log(size, 2))
		self.num_layers = (self.log_size - 2) * 2 + 1

		for i in range(0, self.log_size - 1):
			out_channels = self.channels[i]
			self.linears.append(StyledResBlock(in_channels, out_channels, 1, style_dim, demodulate=demodulate,
																				 activation=activation))
			in_channels = out_channels

		self.to_rgb_last = ToRGB(in_channels, style_dim, upsample=False)

		self.style_dim = style_dim

		layers = [PixelNorm()]

		for i in range(n_mlp):
			layers.append(
				EqualLinear(
					style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
				)
			)

		self.style = nn.Sequential(*layers)

	def forward(self,
							coords,
							latent,
							return_latents=False,
							truncation=1,
							truncation_latent=None,
							input_is_latent=False,
							):

		latent = latent[0]

		if truncation < 1:
			latent = truncation_latent + truncation * (latent - truncation_latent)

		if not input_is_latent:
			latent = self.style(latent)

		x = self.lff(coords)

		batch_size, _, w, h = coords.shape
		if self.training and w == h == self.size:
			emb = self.emb(x)
		else:
			emb = F.grid_sample(
				self.emb.input.expand(batch_size, -1, -1, -1),
				coords.permute(0, 2, 3, 1).contiguous(),
				padding_mode='border', mode='bilinear',
			)
		out = torch.cat([x, emb], 1)

		for con in self.linears:
			out = con(out, latent)

		out = self.to_rgb_last(out, latent)

		if return_latents:
			return out, latent
		else:
			return out, None

