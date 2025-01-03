import torch
import torch.nn as nn
from torchinfo import summary
from ModelUtils import CNN_Encoder

"""
classes Attention, TransformerBlock and EncoderBlock, and function init_ are copied from the following repo:
https://github.com/Xiong5Heng/GOPT/blob/main/model.py
From the following paper:
GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning,
IEEE ROBOTICS AND AUTOMATION LETTERS. PREPRINT VERSION, ACCEPTED SEPTEMBER, 2024
"""

def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	bias_init(module.bias.data)
	return module

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))

class Attention(nn.Module):
	def __init__(self, embed_size, heads):
		super(Attention, self).__init__()
		self.embed_size = embed_size
		self.heads = heads
		self.head_dim = embed_size // heads

		assert (
			self.head_dim * heads == embed_size
		), "Embedding size needs to be divisible by heads"

		self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

	def forward(self, query, keys, values, pad_mask=None):
		# A.P.: Get number of training examples
		N = query.shape[0]

		value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

		#A.P.: Split the embedding into self.heads different pieces
		values = values.reshape(N, value_len, self.heads, self.head_dim)
		keys = keys.reshape(N, key_len, self.heads, self.head_dim)
		query = query.reshape(N, query_len, self.heads, self.head_dim)

		values = self.values(values)  # A.P.: (N, value_len, heads, head_dim)
		keys = self.keys(keys)        # A.P.: (N, key_len, heads, head_dim)
		queries = self.queries(query) # A.P.: (N, query_len, heads, heads_dim)

		# A.P.: Einsum does matrix mult. for query*keys for each training example
		# with every other training example, don't be confused by einsum
		# it's just how I like doing matrix multiplication & bmm

		energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
		# A.P.: queries shape: (N, query_len, heads, heads_dim),
		# A.P.: keys shape: (N, key_len, heads, heads_dim)
		# A.P.: energy: (N, heads, query_len, key_len)

		# Mask padded indices so their weights become 0
		if pad_mask is not None:
			pad_mask = pad_mask.unsqueeze(-1).expand(N, query_len, key_len)
			pad_mask = pad_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
			energy = energy.masked_fill(pad_mask==0, -1e18)
			# energy = energy.masked_fill(pad_mask==0, float("-inf"))

		# A.P.: Normalize energy values similarly to seq2seq + attention
		# so that they sum to 1. Also divide by scaling factor for
		# better stability
		attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
		# A.P.: attention shape: (N, heads, query_len, key_len)

		out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
			N, query_len, self.heads * self.head_dim
		)
		# A.P.: attention shape: (N, heads, query_len, key_len)
		# A.P.: values shape: (N, value_len, heads, heads_dim)
		# A.P.: out after matrix multiply: (N, query_len, heads, head_dim), then
		# we reshape and flatten the last two dimensions.

		out = self.fc_out(out)
		# A.P.: Linear layer doesn't modify the shape, final shape will be (N, query_len, embed_size)

		return out


class TransformerBlock(nn.Module):
	def __init__(self, embed_size, heads, dropout, forward_expansion):
		super(TransformerBlock, self).__init__()
		self.attention = Attention(embed_size, heads)
		self.norm1 = nn.LayerNorm(embed_size)
		self.norm2 = nn.LayerNorm(embed_size)

		self.feed_forward = nn.Sequential(
			nn.Linear(embed_size, forward_expansion * embed_size),
			nn.ReLU(),
			nn.Linear(forward_expansion * embed_size, embed_size),
		)

		self.dropout = nn.Dropout(dropout)

	def forward(self, query, key, value, pad_mask=None):
		attention = self.attention(query, key, value, pad_mask)

		# A.P.: Add skip connection, run through normalization and finally dropout
		x = self.dropout(self.norm1(attention + query))
		forward = self.feed_forward(x)
		out = self.dropout(self.norm2(forward + x))
		return out


class EncoderBlock(nn.Module):
	def __init__(self, embed_size, heads, forward_expansion, dropout):
		super(EncoderBlock, self).__init__()
		
		self.item_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
		self.ems_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
		self.ems_on_item = TransformerBlock(embed_size, heads, dropout, forward_expansion)
		self.item_on_ems = TransformerBlock(embed_size, heads, dropout, forward_expansion)

	def forward(self, item_feature, ems_feature, mask=None):
		# self-attention
		item_embedding = self.item_embedding(item_feature, item_feature, item_feature)
		ems_embedding = self.ems_embedding(ems_feature, ems_feature, ems_feature, mask)
		# cross-attention
		ems_on_item = self.ems_on_item(ems_embedding, item_embedding, item_embedding, mask) 
		item_on_ems = self.item_on_ems(item_embedding, ems_embedding, ems_embedding)

		return item_on_ems, ems_on_item
	
class HeightMapEncoder(nn.Module):
	def __init__(self, embed_size, heads, forward_expansion, dropout):
		super(HeightMapEncoder, self).__init__()

		self.ems_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
		self.hm_encoder = CNN_Encoder(
			input_channels=1, output_dim=embed_size, conv_out_channels=32, 
			hidden_layer_channels=[32, 64, 128], mlp_layers=[256, 64]
		)
		self.ems_on_hm = TransformerBlock(embed_size, heads, dropout, forward_expansion)

	def forward(self, ems_feature, height_map, mask=None):
		ems_embedding = self.ems_embedding(ems_feature, ems_feature, ems_feature, mask)
		# print(height_map.shape)
		hm_embedding = self.hm_encoder(height_map)
		hm_embedding = hm_embedding.reshape(hm_embedding.size(0), 1, -1)
		ems_embedding = self.ems_on_hm(ems_embedding, hm_embedding, hm_embedding, mask)
		return ems_embedding, hm_embedding
	
class BPP_Model_EMS(nn.Module):
	def __init__(self, 
	      num_placement=100,
	      embed_size=32,
	      num_encoders=3,
	      num_heads=4,
	      forward_expansion=4,
	      dropout=0.1,
	      mlp_layers=[32, 64],
	      batch_size=8,
	      feature_mlp_layers=[128, 128],
	      feature_mlp_output=32,
	      ):
		super(BPP_Model_EMS, self).__init__()
		self.n_placements = num_placement
		self.embed_size = embed_size
		self.num_heads = num_heads
		self.forward_expansion = forward_expansion
		self.dropout = dropout
		self.batch_size = batch_size
		
		ems_layers = [init_(nn.Linear(5, mlp_layers[0])), nn.LayerNorm(mlp_layers[0]), nn.LeakyReLU()]
		for i in range(len(mlp_layers)-1):
			ems_layers.append(init_(nn.Linear(mlp_layers[i], mlp_layers[i+1])))
			ems_layers.append(nn.LayerNorm(mlp_layers[i+1]))
			ems_layers.append(nn.LeakyReLU())
		ems_layers.append(init_(nn.Linear(mlp_layers[-1], embed_size)))
		self.ems_encoder = nn.Sequential(*ems_layers)

		item_layers = [init_(nn.Linear(3, mlp_layers[0])), nn.LayerNorm(mlp_layers[0]), nn.LeakyReLU()]
		for i in range(len(mlp_layers)-1):
			item_layers.append(init_(nn.Linear(mlp_layers[i], mlp_layers[i+1])))
			item_layers.append(nn.LayerNorm(mlp_layers[i+1]))
			item_layers.append(nn.LeakyReLU())
		item_layers.append(init_(nn.Linear(mlp_layers[-1], embed_size)))
		self.item_encoder = nn.Sequential(*item_layers)

		self.HMEncoder = HeightMapEncoder(
			embed_size=embed_size,
			heads=num_heads,
			forward_expansion=forward_expansion,
			dropout=dropout
		)
		self.GeneralEncoder = nn.ModuleList([
			EncoderBlock(
				embed_size=embed_size,
				heads=num_heads,
				forward_expansion=forward_expansion,
				dropout=dropout
			) for _ in range(num_encoders)
		])

		ems_feature_layers = [init_(nn.Linear(embed_size, feature_mlp_layers[0])), nn.LayerNorm(feature_mlp_layers[0]), nn.LeakyReLU()]
		for i in range(len(feature_mlp_layers)-1):
			ems_feature_layers.append(init_(nn.Linear(feature_mlp_layers[i], feature_mlp_layers[i+1])))
			ems_feature_layers.append(nn.LayerNorm(feature_mlp_layers[i+1]))
			ems_feature_layers.append(nn.LeakyReLU())
		ems_feature_layers.append(init_(nn.Linear(feature_mlp_layers[-1], feature_mlp_output)))
		self.ems_feature_mlp = nn.Sequential(*ems_feature_layers)

		item_feature_layers = [init_(nn.Linear(embed_size, feature_mlp_layers[0])), nn.LayerNorm(feature_mlp_layers[0]), nn.LeakyReLU()]
		for i in range(len(feature_mlp_layers)-1):
			item_feature_layers.append(init_(nn.Linear(feature_mlp_layers[i], feature_mlp_layers[i+1])))
			item_feature_layers.append(nn.LayerNorm(feature_mlp_layers[i+1]))
			item_feature_layers.append(nn.LeakyReLU())
		item_feature_layers.append(init_(nn.Linear(feature_mlp_layers[-1], feature_mlp_output)))
		self.item_feature_mlp = nn.Sequential(*item_feature_layers)
		self.output_softmax = nn.Softmax(dim=1)
	
	def forward(self, ems_placements, coming_items, height_map, mask):
		# mask: (batch_size, 2, n_placements) -> (batch_size, 2 * n_placements)
		# ems_mask: (batch_size, n_placements)
		ems_mask = torch.any(mask.bool(), dim=1)
		mask = mask.bool().reshape(self.batch_size, -1)
		ems_state = self.ems_encoder(ems_placements)
		item_state = self.item_encoder(coming_items)
		# print(ems_placements.shape, ems_state.shape, item_state.shape)
		# height_state = height_map.clone()
		# for encoder in self.HMEncoder:
		ems_state, height_state = self.HMEncoder(ems_state, height_map, ems_mask)
		for encoder in self.GeneralEncoder:
			item_state, ems_state = encoder(item_state, ems_state, ems_mask)
		# item_state: (batch_size, 2, embed_size)
		# ems_state: (batch_size, n_placements, embed_size)
		ems_feature = self.ems_feature_mlp(ems_state)
		item_feature = self.item_feature_mlp(item_state).permute(0, 2, 1)
		# print(ems_feature.shape)
		# print(item_feature.shape)
		logits = torch.bmm(ems_feature, item_feature).reshape(self.batch_size, -1)
		# masked_logits = self.output_softmax(logits * mask)
		masked_logits = logits.masked_fill(mask == False, float(-1e9))
		masked_logits = self.output_softmax(masked_logits)
		# print(masked_logits.shape)
		return ems_state, item_state, ems_feature, item_feature, logits, masked_logits
	
def test_model_dims():
	batch_size = 13
	bpp_model = BPP_Model_EMS(batch_size=batch_size).to(torch.device('cuda'))
	summary(bpp_model, [(batch_size, 100, 5), (batch_size, 2, 3), (batch_size, 1, 100, 100), (batch_size, 2, 100)])

if __name__ == '__main__':
	test_model_dims()
