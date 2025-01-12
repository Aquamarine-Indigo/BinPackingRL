import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PackingUtils import *
from ModelEMS import *
from tqdm import tqdm
import datetime
import random

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class BinPacking_Environment:
	def __init__(self, container_dim, items_list: list[np.ndarray], clip_num=100, device='cuda'):
		self.container_dim = container_dim
		self.items_list = items_list
		self.height_map = np.zeros(container_dim[:2])
		self.packed_volume = 0
		# self.max_box_height = 0
		self.total_volume = np.prod(container_dim)
		self.item_ptr = 0
		self.clip_num = clip_num
		self.device = torch.device(device)
		self.used = 0
	
	def step(self, action, invalid_quit=True):
		# action is a tuple of (x, y, rotate)
		# if not self.is_valid(action):
		# 	return -1, False
		if not self.is_valid(action):
			return 0, invalid_quit
		# print("find height: ", find_min_height(self.height_map, action[:2], self.items_list[self.item_ptr].tolist(), action[2]))
		self.height_map = update_height_map(
			self.height_map, 
			self.items_list[self.item_ptr].tolist(),
			action[:2],
			action[2]
		)
		old_util = self.packed_volume / self.total_volume
		self.packed_volume += np.prod(self.items_list[self.item_ptr])
		used_positions = np.sum(self.height_map > 0)
		max_height = np.max(self.height_map)
		used_volume = used_positions * max_height
		# reward = self.packed_volume / used_volume
		# reward = self.packed_volume / self.total_volume - old_util
		reward = np.prod(self.items_list[self.item_ptr])
		# state = self.get_state(self.clip_num)
		self.item_ptr += 1
		self.used += 1
		done = self.item_ptr == len(self.items_list)
		return reward, done

	def is_valid(self, action):
		li, wi, hi = self.items_list[self.item_ptr].tolist()
		x, y, r = action
		if r == 1:
			li, wi = wi, li
		# print(li, wi, hi, action)
		if x-li+1 < 0 or y+wi > self.container_dim[1]:
			return False
		stable_cnt = positionally_stable(
			self.height_map[x-li+1:x+1, y:y+wi],
			li, wi, hi, self.container_dim[2]
		)
		if stable_cnt > 0:
			return True
		return False
	
	def get_state(self):
		new_item = self.items_list[self.item_ptr].tolist()
		# placement: (N, 6)
		# mask: (2, N)
		# print(self.height_map)
		placement, mask = generate_EMS_and_mask(
			self.height_map, 
			new_item,
			self.container_dim[2],
			self.clip_num
		)
		# item_state: (2, 3)
		item_state = np.array([
			new_item, 
			[new_item[1], new_item[0], new_item[2]]
		])
		# print("ENV: ", placement.shape)
		height_map = torch.Tensor(self.height_map).reshape(1, 1, self.container_dim[0], self.container_dim[1]).to(self.device)
		placement = torch.Tensor(placement).reshape(1, -1, 5).to(self.device)
		item_state = torch.Tensor(item_state).reshape(1, 2, 3).to(self.device)
		mask = torch.Tensor(mask).reshape(1, 2, -1).to(self.device)
		return placement, item_state, height_map, mask

	def get_utilization(self):
		return self.packed_volume / self.total_volume

	def get_remaining_cnt(self):
		return len(self.items_list) - self.used
	def get_item_cnt(self):
		return len(self.items_list)
	def masked_all(self, feasibility_mask):
		return torch.sum(feasibility_mask) == 0
	
def get_action_from_idx(placement, idx, clip_num, height_map=None, item_size=None, return_height=False):
	if idx >= clip_num:
		place = placement[0, idx - clip_num, :3].cpu().detach().numpy().astype(np.int32)
		rotate = 1
		# if return_height == True:
		# 	return (place[0], place[1], 0, place[2])
		# return (place[0], place[1], 1)
	else:
		place = placement[0, idx, :3].cpu().detach().numpy().astype(np.int32)
		rotate = 0
	if return_height == True:
		# print(item_size, place)
		try:
			height = find_min_height(height_map, (place[0], place[1]), item_size, rotate)
		except:
			# print("invalid placement")
			return (place[0], place[1], 0, -1)
		return (place[0], place[1], rotate, height)
	return (place[0], place[1], rotate)

def train(item_lists, gamma=0.99, lr=0.001, clip_num=50, max_episode=300, max_steps=100, device_name='cpu'):
	device = torch.device(device_name)
	model = BPP_Model_EMS(num_placement=clip_num, batch_size=1, embed_size=128, feature_mlp_layers=[256, 128], feature_mlp_output=64).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-6, weight_decay=1e-4)
	progress_bar = tqdm(range(max_episode), desc='Training')
	# max_steps = 200
	n_item_lists = len(item_lists)
	for i in range(n_item_lists):
		item_lists[i].sort(key=lambda x: max(x[0], x[1], x[2]), reverse=True)
	model.train()
	for episode in progress_bar:
		# random.shuffle(item_list)
		item_list = item_lists[episode % n_item_lists]
		env = BinPacking_Environment((100, 100, 100), item_list, clip_num=clip_num, device=device_name)
		step_cnt = 0
		done = False
		states, actions, rewards = [], [], []
		while not done:
			placement, item_info, height_map, feasibility_mask = env.get_state()
			if env.masked_all(feasibility_mask) == True:
				# print("All masked")
				env.item_ptr += 1
				if env.item_ptr >= len(item_list):
					done = True
					break
				continue
			ems_state, item_state, ems_feature, item_feature, logits_raw_, logits = model(
				placement,
				item_info,
				height_map, 
				feasibility_mask
			)
			# print(feasibility_mask)
			# print(logits)
			# if torch.any(logits < 0) or torch.any(logits == torch.nan) or torch.any(logits == torch.inf):
			# 	print(f"logits < 0 exists")
			# 	print(logits)
			# print(logits)
			try:
				action_idx = torch.multinomial(logits, 1).item()
			except Exception as e:
				print(f"Error: {e}")
				print(item_list)
				print(logits)
				print(logits_raw_)
				# print(feasibility_mask.cpu().detach().numpy().tolist())
				print("feasibility mask: ", feasibility_mask)
				print("placement: ", placement)
				print("item_info: ", item_info)
				print("EMS state: ", ems_state)
				print("Item state: ", item_state)
				print("height_map: ", height_map)

				print("EMS feature: ", ems_feature)
				print("Item feature: ", item_feature)
				print("step count: ", step_cnt)
				torch.save(model.state_dict(), f"checkpoints_size_reward/model_nan.pth")
				exit()
			# print(action_idx)
			action = get_action_from_idx(placement, action_idx, clip_num)
			reward, done = env.step(action, invalid_quit=False)
			if reward == 0:
				# print(f"skip this step: item {item_state.cpu().detach().numpy()}")
				env.item_ptr += 1
				if env.item_ptr >= len(item_list):
					done = True
			else:
				states.append((placement, item_info, height_map, feasibility_mask))
				rewards.append(reward)
				actions.append(action_idx)

			if step_cnt >= max_steps and not done:
				done = True
				# rewards[-1] -= env.get_remaining_cnt() * 0.1
				break
			step_cnt += 1
		
		# print(f"height_map: \n{env.height_map}")
		# state_end = env.get_state()
		# # print(f"ems: \n{state_end[0]}")
		# print(f"next_item: \n{state_end[1]}")

		# compute returns
		# print("len_rewards", len(rewards))
		print("reward ", rewards)
		cumulative_rewards = []
		R = 0
		for r in reversed(rewards):
			R = gamma * R + r
			cumulative_rewards.insert(0, R)
		cumulative_rewards = torch.tensor(cumulative_rewards)
		cumulative_rewards = (cumulative_rewards - cumulative_rewards.mean()) / (cumulative_rewards.std() + 1e-8)
		print("cumulative_rewards", cumulative_rewards)

		optimizer.zero_grad()
		for state, action, R in zip(states, actions, cumulative_rewards):
			_, _, _, _, logits_raw_, logits = model(
				state[0],
				state[1],
				state[2],
				state[3]
			)
			prob = logits[0, action] + 1e-8
			log_prob = torch.log(prob)
			loss = -log_prob * R
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
			optimizer.step()
		progress_bar.set_description(f"Episode {episode + 1}, space_util: {env.get_utilization()*100:.4f}%, unused: {env.get_remaining_cnt()}/{env.get_item_cnt()}")
		if (episode + 1) % 10 == 0:
			print(f"Episode {episode + 1} loss: {loss.item()}")
			torch.save(model.state_dict(), f"checkpoints_size_reward/model_{episode + 1}.pth")
	return model

def load_item_lists(filenames):
	item_lists = []
	for filename in filenames:
		items = np.load(filename)
		# print(items)
		item_list_ = items.tolist()
		print(item_list_)
		item_list = []
		for item in item_list_:
			item_list.append(np.array(item))
		item_lists.append(item_list)
	return item_lists
		

if __name__ == "__main__":
	# items = np.load("dataset_map/item_list.npy")
	# # print(items)
	# item_list_ = items.tolist()
	# print(item_list_)
	# item_list = []
	# for item in item_list_:
	# 	item_list.append(np.array(item))
	# # train()
	item_lists = load_item_lists(["dataset_map/item_list.npy", "dataset_map/item_list_.npy"])
	model = train(item_lists, clip_num=100, device_name='cuda', max_episode=500)
	torch.save(model.state_dict(), f"models/model_{current_time}_.pth")
