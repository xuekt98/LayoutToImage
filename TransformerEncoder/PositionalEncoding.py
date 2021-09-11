"""
	!!!选用什么方式实现位置编码还需要再实现
	这里先采用Transformer最初的sin,cos固定位置编码实现
"""
class PositionalEncoding(nn.Module):
		"""位置编码部分，取bounding box的中心进行编码"""

		def __init__(self, d_model=512, max_image_size=(256, 256)):
				"""
				Params:
					d_model: 模型维度，默认512
					max_image_size: 生成图像的最大尺寸，默认1024，即最大1024*1024像素的图像
				"""
				super(PositionalEncoding, self).__init__()
				self.H, self.W = max_image_size
				self.d_model = d_model
                
                # 这样写会阻塞
				# PE = np.array([
				# 		[pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
				# 		for pos in range(np.prod(max_image_size))])


				# self.PE_net = nn.Embedding(max_image_size + 1, d_model) 不需要使用
				# self.PE_net.weight = nn.Parameter(PE, requires_grad=False)
                
		def forward(self, bb_center):
				'''
					bb_center: B x 2
				'''
				x, y = bb_center[:, 0], bb_center[:, 1]
				pos = (x * self.W + y).unsqueeze(1)
				pe_fe = torch.arange(self.d_model).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
				PE = pos / torch.pow(10000, 2.0 * (pe_fe // 2) / self.d_model)
				PE[:, 0::2] = torch.sin(PE[:, 0::2])
				PE[:, 1::2] = torch.cos(PE[:, 1::2])
				return PE
