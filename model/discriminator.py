from torchvision.ops import roi_align
from torchvision.ops import roi_pool
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

class ROIAlign(nn.Module):
	def __init__(self, output_size, spatial_scale, sampling_ratio):
		super(ROIAlign, self).__init__()
		self.output_size = output_size
		self.spatial_scale = spatial_scale
		self.sampling_ratio = sampling_ratio

	def forward(self, input, rois):
		return roi_align(
			input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
		)


class ROIPool(nn.Module):
	def __init__(self, output_size, spatial_scale):
		super(ROIPool, self).__init__()
		self.output_size = output_size
		self.spatial_scale = spatial_scale

	def forward(self, input, rois):
		return roi_pool(input, rois, self.output_size, self.spatial_scale)

def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
	conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
	if spectral_norm:
		return nn.utils.spectral_norm(conv, eps=1e-4)
	else:
		return conv

class ObjDiscriminator64(nn.Module):
	def __init__(self, num_classes=81):
		super(ObjDiscriminator64, self).__init__()
		self.obD = ResnetDiscriminator64(num_classes=num_classes, input_dim=3)
<<<<<<< HEAD
		self.COUNT_F = 0
=======
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301

	def forward(self, images, bbox, label, mask=None):
		'''
		images: (b*o')chw (o' is the real no. of obj) or bchw (if this is for real image)
		bbox: bo4 (o is the max no. which is set in configuration of obj)
		mask: \prehold
		'''
		bbox = bbox * images.size(2)
		if images.shape[0] == bbox.shape[0]:
			idx = torch.arange(start=0, end=images.size(0), device=images.device)
			idx = idx.view(images.size(0), 1, 1).expand(-1, bbox.size(1), -1).float()
			bbox = torch.cat((idx, bbox.float()), dim=2)  # index to all obj for one image
			bbox = bbox.view(-1, 5)
		else:
			bbox = bbox.view(-1, 4)
<<<<<<< HEAD

=======
			idx = torch.arange(start=0, end=bbox.size(0), device=images.device)
			idx = idx.view(-1, 1).float()
			bbox = torch.cat((idx, bbox), dim=-1)
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301

		# only for input bbox: (x0, y0, h, w)
		# bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
		# bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
		label = label.view(-1)


		idx = (label != 0).nonzero().view(-1)
		bbox = bbox[idx]
		label = label[idx]
<<<<<<< HEAD

=======
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301
		d_out_obj, d_out_obj_c = self.obD(images, label, bbox)
		return d_out_obj, d_out_obj_c

class ResnetDiscriminator64(nn.Module):
	def __init__(self, num_classes=0, input_dim=3, ch=64):
		super(ResnetDiscriminator64, self).__init__()
		self.num_classes = num_classes

		self.block1 = OptimizedBlock(input_dim, ch, downsample=False)
		self.block2 = ResBlock(ch, ch * 2, downsample=False)
		self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
		self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
		self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
		# self.l_im = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
<<<<<<< HEAD
		self.activation = nn.Tanh()
=======
		self.activation = nn.ReLU()
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301

		# object path
		self.roi_align = ROIAlign((8, 8), 1.0 / 2.0, 0)
		self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=True)
		self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 8, 1))
		self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))
<<<<<<< HEAD
		self.COUNT_F = 0
=======
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301

		self.init_parameter()

	def forward(self, x, y=None, bbox=None):
		b = bbox.size(0)
		# 64x64
		x = self.block1(x)
		# 64x64
		x = self.block2(x)
		# 32x32
		x1 = self.block3(x)
		# 16x16
<<<<<<< HEAD
		# x = self.block4(x1)
		# 8x8
		# x = self.block5(x)
		# x = self.activation(x)
		# x = torch.mean(x, dim=(2, 3))
		# out_im = self.l_im(x)

		# obj path
		# if self.COUNT_F == 110 or self.COUNT_F == 0:
		# 	pdb.set_trace()
		obj_feat = self.roi_align(x1, list(bbox)) if bbox.dim() < 5 else self.roi_align(x1, bbox)
		obj_feat = self.block_obj4(obj_feat)
		obj_feat = self.activation(obj_feat)
		obj_feat = torch.mean(obj_feat, dim=(2, 3))
		out_obj = self.l_obj(obj_feat)
		out_obj_c = out_obj + torch.mean(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)
		self.COUNT_F += 1
=======
		x = self.block4(x1)
		# 8x8
		x = self.block5(x)
		x = self.activation(x)
		x = torch.mean(x, dim=(2, 3))
		# out_im = self.l_im(x)

		# obj path
		obj_feat = self.roi_align(x1, bbox)
		obj_feat = self.block_obj4(obj_feat)
		obj_feat = self.activation(obj_feat)
		obj_feat = torch.sum(obj_feat, dim=(2, 3))
		out_obj = self.l_obj(obj_feat)
		out_obj_c = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)
>>>>>>> 298b1038dc8f4d83ad944e45d69cfb891bbaa301

		return out_obj, out_obj_c

	def init_parameter(self):
		for k in self.named_parameters():
			if k[1].dim() > 1:
				torch.nn.init.orthogonal_(k[1])
			if k[0][-4:] == 'bias':
				torch.nn.init.constant_(k[1], 0)


class ResnetDiscriminator128(nn.Module):
	def __init__(self, num_classes=0, input_dim=3, ch=64):
		super(ResnetDiscriminator128, self).__init__()
		self.num_classes = num_classes

		self.block1 = OptimizedBlock(3, ch, downsample=True)
		self.block2 = ResBlock(ch, ch * 2, downsample=True)
		self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
		self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
		self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
		self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
		self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
		self.activation = nn.ReLU()

		self.roi_align_s = ROIAlign((8, 8), 1.0 / 4.0, int(0))
		self.roi_align_l = ROIAlign((8, 8), 1.0 / 8.0, int(0))

		self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
		self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
		self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
		self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
		self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))

	def forward(self, x, y=None, bbox=None):
		b = bbox.size(0)
		# 128x128
		x = self.block1(x)
		# 64x64
		x1 = self.block2(x)
		# 32x32
		x2 = self.block3(x1)
		# 16x16
		x = self.block4(x2)
		# 8x8
		x = self.block5(x)
		# 4x4
		x = self.block6(x)
		x = self.activation(x)
		x = torch.sum(x, dim=(2, 3))
		out_im = self.l7(x)

		# obj path
		# seperate different path
		s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64)
		bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
		y_l, y_s = y[~s_idx], y[s_idx]

		obj_feat_s = self.block_obj3(x1)
		obj_feat_s = self.block_obj4(obj_feat_s)
		obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

		obj_feat_l = self.block_obj4(x2)
		obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

		obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
		y = torch.cat([y_l, y_s], dim=0)
		obj_feat = self.block_obj5(obj_feat)
		obj_feat = self.activation(obj_feat)
		obj_feat = torch.sum(obj_feat, dim=(2, 3))
		out_obj = self.l_obj(obj_feat)
		out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

		return out_im, out_obj

class CombineDiscriminator128(nn.Module):
	def __init__(self, num_classes=81):
		super(CombineDiscriminator128, self).__init__()
		self.obD = ResnetDiscriminator128(num_classes=num_classes, input_dim=3)

	def forward(self, images, bbox, label, mask=None):
		idx = torch.arange(start=0, end=images.size(0),
											 device=images.device).view(images.size(0),
																									1, 1).expand(-1, bbox.size(1), -1).float()
		bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
		bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
		bbox = bbox * images.size(2)
		bbox = torch.cat((idx, bbox.float()), dim=2)
		bbox = bbox.view(-1, 5)
		label = label.view(-1)

		idx = (label != 0).nonzero().view(-1)
		bbox = bbox[idx]
		label = label[idx]
		d_out_img, d_out_obj = self.obD(images, label, bbox)
		return d_out_img, d_out_obj

class OptimizedBlock(nn.Module):
	def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
		super(OptimizedBlock, self).__init__()
		self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
		self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
		self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
		self.activation = nn.ReLU()
		self.downsample = downsample

	def forward(self, in_feat):
		x = in_feat
		x = self.activation(self.conv1(x))
		x = self.conv2(x)
		if self.downsample:
			x = F.avg_pool2d(x, 2)
		return x + self.shortcut(in_feat)

	def shortcut(self, x):
		if self.downsample:
			x = F.avg_pool2d(x, 2)
		return self.c_sc(x)

class ResBlock(nn.Module):
	def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
		super(ResBlock, self).__init__()
		self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
		self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
		self.activation = nn.ReLU()
		self.downsample = downsample
		self.learnable_sc = (in_ch != out_ch) or downsample
		if self.learnable_sc:
			self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)

	def residual(self, in_feat):
		x = in_feat
		x = self.conv1(self.activation(x))
		x = self.conv2(self.activation(x))
		if self.downsample:
			x = F.avg_pool2d(x, 2)
		return x

	def shortcut(self, x):
		if self.learnable_sc:
			x = self.c_sc(x)
			if self.downsample:
				x = F.avg_pool2d(x, 2)
		return x

	def forward(self, in_feat):
		return self.residual(in_feat) + self.shortcut(in_feat)