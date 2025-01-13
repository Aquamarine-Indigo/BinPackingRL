import torch
import torch.nn as nn
import numpy as np
from PackingUtils import *
from ModelUtils import *
from torchinfo import summary

class BPP_Encoder(nn.Module):
	def __init__(self, 
		image_size, patch_size, embed_dim, 
		num_heads, hidden_dim, num_layers, output_dim,
		batch_size,
		mlp_dimensions = [256, 256, 256]
	):
		super(BPP_Encoder, self).__init__()
		self.batch_size = batch_size
		self.height_map_encode_vit = EncodeViT(
			image_size, patch_size, 1, embed_dim,
			num_heads, hidden_dim, num_layers, output_dim,
			batch_size
		)
		self.feasibility_encode_vit = EncodeViT(
			image_size, patch_size, 2, embed_dim,
			num_heads, hidden_dim, num_layers, output_dim,
			batch_size
		)
		self.coming_item_encode = nn.Sequential(
			nn.Linear(3, 64),
			nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, output_dim),
		)
		height_mlp = [nn.Linear(output_dim, mlp_dimensions[0]), nn.LayerNorm(mlp_dimensions[0]), nn.ReLU()]
		last_dim = mlp_dimensions[0]
		for i in mlp_dimensions[1:]:
			height_mlp.append(nn.Linear(last_dim, i))
			height_mlp.append(nn.LayerNorm(i))
			height_mlp.append(nn.ReLU())
			last_dim = i
		height_mlp.append(nn.Linear(last_dim, output_dim))

		self.height_map_mlp = nn.Sequential(*height_mlp)

		feasible_mlp = [nn.Linear(output_dim, mlp_dimensions[0]), nn.LayerNorm(mlp_dimensions[0]), nn.ReLU()]
		last_dim = mlp_dimensions[0]
		for i in mlp_dimensions[1:]:
			feasible_mlp.append(nn.Linear(last_dim, i))
			feasible_mlp.append(nn.LayerNorm(i))
			feasible_mlp.append(nn.ReLU())
			last_dim = i
		feasible_mlp.append(nn.Linear(last_dim, output_dim))

		self.feasibility_mlp = nn.Sequential(*feasible_mlp)

	def forward(self, height_map, coming_item, feasibility_map):
		# height_map, coming_item, feasibility_map = state_encoding
		xh = self.height_map_encode_vit(height_map)
		xf = self.feasibility_encode_vit(feasibility_map)
		xi = self.coming_item_encode(coming_item)
		xh = self.height_map_mlp(xh) + xh
		xf = self.feasibility_mlp(xf) + xf
		return xh, xi, xf
	
class BPP_Encoder_CNN(nn.Module):
	def __init__(self,
		image_size, conv_out_channels, out_dim, batch_size,
		conv_layers=[64, 64, 128, 128, 256, 256],
		conv_mlp_layers=[1024, 512, 256],
		mlp_dimensions = [256, 256, 256] 
	):
		super(BPP_Encoder_CNN, self).__init__()
		self.image_size = image_size
		self.batch_size = batch_size
		self.conv_layers = conv_layers
		self.conv_mlp_layers = conv_mlp_layers
		self.out_dim = out_dim
		self.height_map_encode = CNN_Encoder(
			input_channels=1, output_dim=out_dim,
			hidden_layer_channels=conv_layers, 
			mlp_layers=conv_mlp_layers,
		)
		self.feasibility_encode = CNN_Encoder(
			input_channels=2, output_dim=out_dim,
			hidden_layer_channels=conv_layers,
			mlp_layers=conv_mlp_layers,
		)
		coming_item_layers = [nn.Linear(3, mlp_dimensions[0]), nn.ReLU()]
		for i in range(len(mlp_dimensions) - 1):
			coming_item_layers.append(nn.Linear(mlp_dimensions[i], mlp_dimensions[i+1]))
			coming_item_layers.append(nn.ReLU())
		self.coming_item_mlp = nn.Sequential(*coming_item_layers)
	
	def forward(self, height_map, coming_item, feasibility_map):
		xh = self.height_map_encode(height_map)
		xi = self.coming_item_mlp(coming_item)
		xf = self.feasibility_encode(feasibility_map)
		return xh, xi, xf

	
class BPP_Decoder_CNN(nn.Module):
	def __init__(self, 
	      sequence_len, embed_dim, num_heads, output_dim, batch_size, 
	      mlp_dimensions=[1024, 2048, 4096],
	      output_size=(100, 100, 2),
	):
		super(BPP_Decoder_CNN, self).__init__()
		self.seq_len = sequence_len
		self.output_dim = output_dim
		self.embed_dim = embed_dim
		self.output_size = (batch_size, *output_size)
		self.batch_size = batch_size
		assert self.output_size[1] * self.output_size[2] * 2 == output_dim, "output_dim must be equal to output_size[0] * output_size[1]"

		self.cross_attention_hf = CrossAttention(embed_dim, num_heads)
		self.layer_norm_1 = nn.LayerNorm(embed_dim)
		self.cross_attention_fh = CrossAttention(embed_dim, num_heads)
		self.layer_norm_2 = nn.LayerNorm(embed_dim)
		self.cross_attention_ih = CrossAttention(embed_dim, num_heads)
		self.layer_norm_3 = nn.LayerNorm(embed_dim)

		# self.start_mlp = mlp_dimensions[0]
		mlp_layers = [nn.Linear(embed_dim*self.seq_len, mlp_dimensions[0])]
		last_dim = mlp_dimensions[0]
		for i in mlp_dimensions[1:]:
			mlp_layers.append(nn.Linear(last_dim, i))
			mlp_layers.append(nn.LayerNorm(i))
			mlp_layers.append(nn.ReLU())
			last_dim = i
		mlp_layers.append(nn.Linear(last_dim, output_dim))
		self.mlp = nn.Sequential(*mlp_layers)

	def forward(self, xh, xi, xf, feasibility_mask):
		# cross attention: (batch_size, seq_len, embed_dim)
		xh = xh.permute(0, 2, 1)
		xi = torch.reshape(torch.concat([xi for _ in range(self.embed_dim)]), (xi.shape[0], xi.shape[1], self.embed_dim))
		xf = xf.permute(0, 2, 1)
		# print(xh.shape, xi.shape, xf.shape)
		x1 = self.cross_attention_hf(
			query=xh, key=xf, value=xf
		) + xh
		x1 = self.layer_norm_1(x1)
		x2 = self.cross_attention_fh(
			query=xf, key=x1, value=x1
		) + xf
		x2 = self.layer_norm_2(x2)
		x3 = self.cross_attention_ih(
			query=xi, key=x2, value=x2
		) + xi
		x3 = self.layer_norm_3(x3)
		x3_flattened = x3.view(self.batch_size, -1)
		# print("x3 shape: ", x3.shape)
		# print("x3 flattened shape: ", x3_flattened.shape)
		x = self.mlp(x3_flattened)
		# x = self.layer_norm_4(x)
		# return self.output_mlp(x).resize(self.output_size)
		# return x.view(self.output_size)
		# mask element wise multiplication
		# x = x * feasibility_mask
		return F.softmax(x, dim=1)
	
class BPP_Decoder(nn.Module):
	def __init__(self, 
	      sequence_len, embed_dim, num_heads, output_dim, batch_size, 
	      mlp_dimensions=[1024, 2048, 4096],
	      output_size=(100, 100, 2),
	):
		super(BPP_Decoder, self).__init__()
		self.seq_len = sequence_len
		self.output_dim = output_dim
		self.embed_dim = embed_dim
		self.output_size = (batch_size, *output_size)
		self.batch_size = batch_size
		assert self.output_size[1] * self.output_size[2] * 2 == output_dim, "output_dim must be equal to output_size[0] * output_size[1]"

		self.cross_attention_hf = CrossAttention(embed_dim, num_heads)
		self.layer_norm_1 = nn.LayerNorm(embed_dim)
		self.cross_attention_fh = CrossAttention(embed_dim, num_heads)
		self.layer_norm_2 = nn.LayerNorm(embed_dim)
		self.cross_attention_ih = CrossAttention(embed_dim, num_heads)
		self.layer_norm_3 = nn.LayerNorm(embed_dim)

		# self.start_mlp = mlp_dimensions[0]
		mlp_layers = [nn.Linear(embed_dim*self.seq_len, mlp_dimensions[0])]
		last_dim = mlp_dimensions[0]
		for i in mlp_dimensions[1:]:
			mlp_layers.append(nn.Linear(last_dim, i))
			mlp_layers.append(nn.LayerNorm(i))
			mlp_layers.append(nn.ReLU())
			last_dim = i
		mlp_layers.append(nn.Linear(last_dim, output_dim))
		self.mlp = nn.Sequential(*mlp_layers)
		# self.layer_norm_4 = nn.LayerNorm(output_dim)
		# self.output_mlp = nn.Sequential(
		# 	nn.Linear(embed_dim, output_dim)
		# )

	def forward(self, xh, xi, xf, feasibility_mask):
		# cross attention: (batch_size, seq_len, embed_dim)
		xh = xh.permute(0, 2, 1)
		xi = torch.reshape(torch.concat([xi for _ in range(self.embed_dim)]), (xi.shape[0], xi.shape[1], self.embed_dim))
		xf = xf.permute(0, 2, 1)
		# print(xh.shape, xi.shape, xf.shape)
		x1 = self.cross_attention_hf(
			query=xh, key=xf, value=xf
		) + xh
		x1 = self.layer_norm_1(x1)
		x2 = self.cross_attention_fh(
			query=xf, key=x1, value=x1
		) + xf
		x2 = self.layer_norm_2(x2)
		x3 = self.cross_attention_ih(
			query=xi, key=x2, value=x2
		) + xi
		x3 = self.layer_norm_3(x3)
		x3_flattened = x3.view(self.batch_size, -1)
		# print("x3 shape: ", x3.shape)
		# print("x3 flattened shape: ", x3_flattened.shape)
		x = self.mlp(x3_flattened)
		# x = self.layer_norm_4(x)
		# return self.output_mlp(x).resize(self.output_size)
		# return x.view(self.output_size)
		# mask element wise multiplication
		# x = x * feasibility_mask
		return F.softmax(x, dim=1)
	
class BPP_Model(nn.Module):
	def __init__(self, 
	      image_size, patch_size, embed_dim, num_heads, 
	      hidden_dim, num_layers, encoded_dim, batch_size,
	      encoder_mlp_dims=[256, 256, 256],
	      decoder_mlp_dims=[2048, 2048, 2048],
	      output_size=(50, 50, 2)
	      ):
		super(BPP_Model, self).__init__()
		self.batch_size = batch_size
		self.encoder = BPP_Encoder(
			image_size=image_size, 
			patch_size=patch_size, 
			embed_dim=embed_dim, 
			num_heads=num_heads, 
			hidden_dim=hidden_dim, 
			num_layers=num_layers, 
			output_dim=encoded_dim, 
			batch_size=batch_size,
			mlp_dimensions=encoder_mlp_dims
		)
		self.decoder = BPP_Decoder(
			sequence_len=encoded_dim,
			embed_dim=embed_dim, 
			num_heads=num_heads, 
			output_dim=output_size[0]*output_size[1]*output_size[2],
			batch_size=batch_size,
			mlp_dimensions=decoder_mlp_dims
		)
	
	def forward(self, height_map, coming_item, feasibility_map):
		xh, xi, xf = self.encoder(height_map, coming_item, feasibility_map)
		feasibility_mask = feasibility_map.permute(0, 2, 3, 1).reshape(self.batch_size, -1)
		return self.decoder(xh, xi, xf, feasibility_mask)
	
def test_model_size():
	batch_size = 4
	embed_dim = 16
	seq_len = 4096
	encoder = BPP_Encoder(
		image_size=(100, 100), 
		patch_size=2, 
		embed_dim=embed_dim, 
		num_heads=8, 
		hidden_dim=16, 
		num_layers=5, 
		output_dim=seq_len, 
		batch_size=batch_size
	)
	decoder = BPP_Decoder(
		sequence_len=seq_len, 
		embed_dim=16, 
		num_heads=8, 
		output_dim=100*100*2, 
		batch_size=batch_size
	)
	encoder_summary = summary(encoder, ((batch_size, 1, 100, 100), (batch_size, 3,), (batch_size, 2, 100, 100)))
	decoder_summary = summary(decoder, ((batch_size, 16, 4096), (batch_size, 4096,), (batch_size, 16, 4096)))
	# print(encoder_summary)

def test_model_whole():
	batch_size = 4
	bpp_model = BPP_Model(
		image_size=(100, 100),
		patch_size=2,
		embed_dim=16,
		num_heads=8,
		hidden_dim=16,
		num_layers=5,
		encoded_dim=4096,
		batch_size=batch_size
	)
	summary(bpp_model, ((batch_size, 1, 100, 100), (batch_size, 3,), (batch_size, 2, 100, 100)))

if __name__ == "__main__":
	# pass
	test_model_whole()