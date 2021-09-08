import torch
import torch.nn
import numpy as np

"""
	!!!选用什么方式实现位置编码还需要再实现
	这里先采用Transformer最初的sin,cos固定位置编码实现
"""
class PositionalEncoding(nn.Module):
    """位置编码部分，取bounding box的中心进行编码"""
    def __init__(self, d_model=512, max_image_size=1024):
    	"""
			Params:
				d_model: 模型维度，默认512
				max_image_size: 生成图像的最大尺寸，默认1024，即最大1024*1024像素的图像
    	"""
        super(PositionalEncoding, self).__init__()
        PE = np.array([
        	[pos / np.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
        	for pos in range(max_image_size)])
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])

        self.PE_net = nn.Embedding(max_image_size + 1, d_model)
        self.PE_net.weight = nn.Parameter(PE, require_grad=False)
    
    def forward(self, bb_center):
        return self.PE_net(center)