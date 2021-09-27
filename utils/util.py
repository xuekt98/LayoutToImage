import torch
from torchvision.transforms import Resize
import pdb

def extract_obj_layer(images, bbox, label, base_size):
	o = bbox.size(1)
	bbox = bbox * base_size
	bbox = bbox.view(-1, 4)
	label = label.view(-1)

	idx = (label != 0).nonzero().view(-1)
	bbox = bbox[idx].long()

	images = Resize(base_size)(images).unsqueeze(1).repeat(1, o, 1, 1, 1)
	images = images.view((-1, ) + images.shape[2:])[idx]

	mask = torch.zeros_like(images[:, 0, :, :]).unsqueeze(1).float()
	for i in range(mask.size(0)):
		mask[i, :, bbox[i, 0]:bbox[i, 2], bbox[i, 1]:bbox[i, 3]] = 1

	images = mask * images

	return images

def image_resize(images, size):
	return Resize(size)(images)



