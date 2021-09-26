from model.transformer import TransformerEncoder
from model.vec2mask import VecToMask
from model.blocks import *
from model.GeneratorsCIPS import CIPSskip, CIPSskipObj

class G_NET_obj(nn.Module):
	def __init__(self, cfg, vocab_len):
		super(G_NET_obj, self).__init__()
		mapping = [PixelNorm()]

		for i in range(cfg['n_emlp']):
			mapping.append(
				EqualLinear(
					cfg['d_noise1'], cfg['d_noise1'], activation='fused_lrelu', lr_mul=0.01
				)
			)
		self.mapping = nn.Sequential(*mapping)

		self.encoder = TransformerEncoder(d_model=cfg['d_model'], d_noise=cfg['d_noise1'],
																			num_classes=vocab_len,
																			num_trans_layers=cfg['num_trans_layers'],
																			num_heads=cfg['num_heads'],
																			d_ffn=cfg['d_ffn'],
																			dropout=0.0)

		self.vec2mask = VecToMask(in_features=cfg['d_model'], feature_grid_size=(cfg['d_model'], 8, 8),
															V2G_hidden_features=cfg['d_v2gh'], V2G_num_hidden_layers=cfg['l_v2gh'],
															Decoder_hidden_features=cfg['d_v2mh'], Decoder_num_hidden_layers=cfg['l_v2mh'], )

		self.gen = CIPSskipObj(size=cfg['base_size'], ltnt_size=cfg['d_noise2'],
												style_dim=cfg['d_style'], n_mlp=cfg['n_gmlp'],
												vm_dim=cfg['d_model'],
												activation=None, channel_multiplier=2, )

		self.base_size = cfg['base_size']
		self.d_latent_a = cfg['d_noise2']
		self.COUNT_F = 0

	def forward(self, objects, boxes, latent_s):
		# if self.COUNT_F == 9:
		# 	pdb.set_trace()
		b, o, d = latent_s.shape
		latent_s = self.mapping(latent_s.view(-1, d)).view(b, o, d)

		v_code, attn_map = self.encoder(objects, boxes, latent_s)  # (b*o)chw, list: len->n_layer of encoder, per (b*n_h)oo
		mask, mask_feat = self.vec2mask(v_code, boxes, self.base_size)

		# eliminate obj padding
		objects = objects.view(-1)
		idx = (objects != 0).nonzero().view(-1)
		mask = mask[idx]
		v_code = v_code.view(-1, v_code.shape[-1])[idx]
		mask_feat = mask_feat[idx]
		latent_a = torch.randn(mask.size(0), self.d_latent_a, device=objects.device)


		im_fm, im_rgb, style_ltnt = self.gen(mask=mask,
																				 v_code=v_code,
																				 mask_feat=mask_feat,
																				 latent=latent_a,
																				 return_latents=True,
																				 truncation=1,
																				 truncation_latent=None,
																				 input_is_latent=False,
																				 return_rgb=True
																				 )
		self.COUNT_F += 1
		return im_rgb, mask		# im_rgb: ((b*o')chw), mask: ((b*o')1hw)

class G_NET_agg(nn.Module):
	def __init__(self):
		super(G_NET_agg, self).__init__()

	def forward(self):
		pass

class G_NET_FULL(nn.Module):
	def __init__(self, cfg, vocab_len):
		super(G_NET_FULL, self).__init__()
		mapping = [PixelNorm()]

		for i in range(cfg['n_emlp']):
			mapping.append(
				EqualLinear(
					cfg['d_noise1'], cfg['d_noise1'], activation='lrelu'
				)
			)
		self.mapping = nn.Sequential(*mapping)

		self.encoder = TransformerEncoder(d_model=cfg['d_model'], d_noise=cfg['d_noise1'],
															num_classes=vocab_len,
															num_trans_layers=cfg['num_trans_layers'],
															num_heads=cfg['num_heads'],
															d_ffn=cfg['d_ffn'],
															dropout=0.0)

		self.vec2mask = VecToMask(in_features=cfg['d_model'], feature_grid_size=(cfg['d_model'], 8, 8),
												V2G_hidden_features=cfg['d_v2gh'], V2G_num_hidden_layers=cfg['l_v2gh'],
												Decoder_hidden_features=cfg['d_v2mh'], Decoder_num_hidden_layers=cfg['l_v2mh'],)

		self.gen = CIPSskip(size=cfg['base_size'], ltnt_size=cfg['d_noise2'],
												style_dim=cfg['d_style'], n_mlp=cfg['n_gmlp'],
												vm_dim=cfg['d_model'],
													activation=None, channel_multiplier=2,)


		self.base_size = cfg['base_size']


	def forward(self, objects, boxes, latent_s, latent_a):
		latent_s = self.mapping(latent_s)
		n_o = boxes.shape[1]
		device = objects.device
		# mask = torch.FloatTensor(4, 6, 1, 128, 128).to(device)
		# v_map = torch.FloatTensor(4, 6, 128).to(device)
		# mask_feat = torch.FloatTensor(4, 6, 128, 128, 128).to(device)

		v_map, attn_map = self.encoder(objects, boxes, latent_s)		# (b*o)chw, list: len->n_layer of encoder, per (b*n_h)oo
		mask, mask_feat = self.vec2mask(v_map, boxes, self.base_size)
		# mask = self.view2bo(mask, n_o)
		# mask_feat = self.view2bo(mask_feat, n_o)


		im_fm, im_rgb, style_ltnt = self.gen(mask=self.view2bo(mask, n_o),
																					v_map=v_map,
																					mask_feat=self.view2bo(mask_feat, n_o),
																					latent=latent_a,
																					return_latents=True,
																					truncation=1,
																					truncation_latent=None,
																					input_is_latent=False,
																					return_rgb=True
													)
		return im_rgb, mask

	def view2bo(self, x, n_o):
		return x.view((-1, n_o) + x.shape[1:])

