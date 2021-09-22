from model.transformer import TransformerEncoder
from model.vec2mask import VecToMask
from model.blocks import *
from model.GeneratorsCIPS import CIPSskip

class G_NET(nn.Module):
	def __init__(self, cfg, vocab_len):
		super(G_NET, self).__init__()
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

