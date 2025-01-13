import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

class PatchEmbedding(nn.Module):
	def __init__(self, image_size, patch_size, in_channels, embed_dim):
		super().__init__()
		self.image_size = image_size
		self.patch_size = (patch_size, patch_size)
		self.in_channels = in_channels
		self.embed_dim = embed_dim
		self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
		self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
	
	def forward(self, x):
		x = self.projection(x)
		# print("original shape: ", x.shape)
		# print("flattened shape: ", x.flatten(1).shape)
		# x = x.flatten(2).transpose(1, 2)
		x = x.flatten(-2)
		return x

class PositionalEncoding(nn.Module):
	def __init__(self, embed_dim, num_patches, batch_size):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_patches = num_patches
		self.position_embedding = nn.Parameter(torch.zeros(batch_size, embed_dim, num_patches))

	def forward(self, x):
		# print(x.shape)
		# print(self.position_embedding.shape)
		x = x + self.position_embedding
		return x
	
class TransformerEncoder(nn.Module):
	def __init__(self, embed_dim, num_heads, hidden_dim, num_layers):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.encoder_layers = nn.TransformerEncoderLayer(
			d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim
		)
		self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

	def forward(self, x):
		# x = x.transpose(1, 2)
		# should be: seq len, batch size, embed dim
		x = x.permute(2, 0, 1)
		# print("encoder shape, ", x.shape)
		x = self.encoder(x)
		# should be: batch size, embed dim, seq len
		x = x.permute(1, 2, 0)
		# print("encoded, shape: ", x.shape)
		# x = x.transpose(1, 2)
		return x
	
class EncodeViT(nn.Module):
	def __init__(self, 
		image_size, patch_size, in_channels, embed_dim, 
		num_heads, hidden_dim, num_layers, output_dim, batch_size
	):
		super().__init__()
		self.batch_size = batch_size
		self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
		self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
		self.position_encoding = PositionalEncoding(embed_dim, self.num_patches, batch_size)
		self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, hidden_dim, num_layers)
		self.fc = nn.Linear(self.num_patches, output_dim)

	def forward(self, x):
		x = self.patch_embedding(x)
		x = self.position_encoding(x)
		x = self.transformer_encoder(x)
		# print(x.shape)
		x = self.fc(x)
		return x

class CrossAttention(nn.Module):
	def __init__(self, embed_dim, num_heads):
		super(CrossAttention, self).__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads

		assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

		# Linear layers for Q, K, V
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)

		# Output projection
		self.out_proj = nn.Linear(embed_dim, embed_dim)

	def forward(self, query, key, value, mask=None):
		"""
		Args:
		query: Tensor of shape (batch_size, seq_len_query, embed_dim)
		key, value: Tensors of shape (batch_size, seq_len_key, embed_dim)
		mask: Optional mask for attention (batch_size, seq_len_query, seq_len_key)
		
		Returns:
		Tensor of shape (batch_size, seq_len_query, embed_dim)
		"""
		batch_size = query.size(0)

		# Project Q, K, V
		Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
		K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
		V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

		# Scaled dot-product attention
		scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
		if mask is not None:
			scores = scores.masked_fill(mask == 0, float('-inf'))
		attention_weights = F.softmax(scores, dim=-1)

		# Attention output
		attention_output = torch.matmul(attention_weights, V)

		# Concatenate heads and project output
		attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
		output = self.out_proj(attention_output)

		return output
	
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=True):
		super(ResidualBlock, self).__init__()
		# Main path: two convolutional layers
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
		self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
		self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
		# Shortcut path: match dimensions using a 1x1 convolution if needed
		if in_channels != out_channels or stride != 1:
			self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
		else:
			self.shortcut = nn.Identity()
		
	def forward(self, x):
		# Save the input as the shortcut
		shortcut = self.shortcut(x)
		# Pass through the main path
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		# Add the shortcut and apply ReLU
		out += shortcut
		out = F.relu(out)
		return out

class CNN_Encoder(nn.Module):
	def __init__(self, 
	      input_channels, output_dim, conv_out_channels, 
	      kernel_size=(3, 3), stride=1, padding=1, pooling_out=(32, 32),
	      hidden_layer_channels=[64, 64, 128, 128, 256, 256],
	      mlp_layers=[1024, 512, 256]
	):
		super(CNN_Encoder, self).__init__()
		self.initial_conv = nn.Conv2d(
			input_channels, 
			hidden_layer_channels[0], 
			kernel_size=kernel_size,
			stride=stride,
			padding=padding
		)
		self.initial_bn = nn.BatchNorm2d(hidden_layer_channels[0])
		self.hidden_layers = nn.ModuleList([
			ResidualBlock(hidden_layer_channels[i], hidden_layer_channels[i+1], stride=1) for i in range(len(hidden_layer_channels)-1)
		])
		self.final_conv = nn.Conv2d(hidden_layer_channels[-1], conv_out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
		self.final_bn = nn.BatchNorm2d(conv_out_channels)
		self.pooling = nn.AdaptiveAvgPool2d(pooling_out)

		mlp_initial_size = conv_out_channels * pooling_out[0] * pooling_out[1]

		mlp_list = [nn.Linear(mlp_initial_size, mlp_layers[0]), nn.GELU()]
		for i in range(len(mlp_layers)-1):
			mlp_list.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
			mlp_list.append(nn.GELU())
		mlp_list.append(nn.Linear(mlp_layers[-1], output_dim))
		self.mlp_layers = nn.Sequential(*mlp_list)

	def forward(self, x):
		x = self.initial_conv(x)
		x = self.initial_bn(x)
		x = F.relu(x)
		for layer in self.hidden_layers:
			x = layer(x)
		x = self.final_conv(x)
		x = self.final_bn(x)
		x = F.relu(x)
		x = self.pooling(x)
		x = x.view(x.size(0), -1)
		x = self.mlp_layers(x)
		return x
	
if __name__ == '__main__':
	cnn_model = CNN_Encoder(
		input_channels=1,
		output_dim=32,
		conv_out_channels=32,
		hidden_layer_channels=[8, 32, 128],
		# mlp_layers=[64, 32],
	)
	summary(cnn_model, (1, 1, 100, 100))