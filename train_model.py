import torch
import json
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from ModelUtils import *
from PackingUtils import *
from ModelCrossAttention import *
from tqdm import tqdm
import datetime
from sklearn.model_selection import train_test_split
import os

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def train_process(model, train_loader, num_epochs, criterion, optimizer, scheduler, device):
	model.train()
	progress_bar = tqdm(range(num_epochs), desc='Training')
	for epoch in progress_bar:
		running_loss = 0.0
		for i, (hm, isz, feas, labels) in enumerate(train_loader):
			hm, isz, feas = hm.to(device), isz.to(device), feas.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = model(hm, isz, feas)
			# print(outputs.shape, labels.shape)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			# scheduler.step()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=100, norm_type=2)
			running_loss += loss.item()

		epoch_loss = running_loss / len(train_loader.dataset)
		scheduler.step(epoch_loss)
		progress_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
		progress_bar.set_postfix(Loss=f"{epoch_loss:.8f}")


		if (epoch+1) % 5 == 0:
			torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}.pth')

def get_dataloader(file_path="dataset_map", batch_size=8):
	height_map = np.load(os.path.join(file_path, "height_map.npy"))
	item_size = np.load(os.path.join(file_path, "item_size.npy"))
	feasibility = np.load(os.path.join(file_path, "feasibility.npy"))
	labels = np.load(os.path.join(file_path, "labels.npy"))
	labels = labels.reshape(labels.shape[0], -1)
	print(labels.shape)
	hm_train, hm_test, is_train, is_test, feas_train, feas_test, y_train, y_test = train_test_split(height_map, item_size, feasibility, labels, train_size=0.9)

	hm_tensor = torch.tensor(hm_train, dtype=torch.float32)
	is_tensor = torch.tensor(is_train, dtype=torch.float32)
	feas_tensor = torch.tensor(feas_train, dtype=torch.float32)
	y_tensor = torch.tensor(y_train, dtype=torch.float32)

	hm_test_tensor = torch.tensor(hm_test, dtype=torch.float32)
	is_test_tensor = torch.tensor(is_test, dtype=torch.float32)
	feas_test_tensor = torch.tensor(feas_test, dtype=torch.float32)
	y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

	batch_size = 8
	train_dataset = TensorDataset(hm_tensor, is_tensor, feas_tensor, y_tensor)
	test_dataset = TensorDataset(hm_test_tensor, is_test_tensor, feas_test_tensor, y_test_tensor)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size)
	return train_loader, test_loader

def train_model(file_path="dataset_map"):
	batch_size = 8
	train_loader, test_loader = get_dataloader(file_path, batch_size)
	
	device = torch.device('cuda')

	model = BPP_Model(
		image_size=(100, 100),
		patch_size=2,
		embed_dim=8,
		num_heads=4,
		hidden_dim=16,
		num_layers=3,
		encoded_dim=512,
		batch_size=batch_size,
		encoder_mlp_dims=[256, 256],
		decoder_mlp_dims=[1024, 2048, 4096],
		output_size=(100, 100, 2),
	)

	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

	num_epochs = 100
	train_process(
		model = model,
		train_loader = train_loader,
		optimizer = optimizer,
		criterion = criterion,
		num_epochs = num_epochs,
		device = device,
		scheduler = scheduler
	)
	torch.save(model.state_dict(), f'models/model_{current_time}.pth')

if __name__ == '__main__':
	train_model()
