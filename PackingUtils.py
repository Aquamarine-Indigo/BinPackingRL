import numpy as np
import torch

def equal(a, b):
    return np.abs(a - b) < 1e-6

def find_min_height(height_map, xypos, lwh, rotate):
	x, y = xypos
	l, w, h = lwh
	if rotate == 1:
		l, w = w, l
	return np.max(height_map[x-l+1:x+1, y:y+w])

def positionally_stable(heights_, li, wi, hi, height_limit):
	max_height = np.max(heights_)
	if max_height + hi > height_limit:
		return False
	max_height_cnt = 0
	# print(f"posiitonally stable: max_height = {max_height}, hi = {hi}, height_limit = {height_limit}")
	for i in range(li):
		for j in range(wi):
			if equal(heights_[i, j], max_height):
				max_height_cnt += 1
	if max_height_cnt >= int((li*wi + 1) * 0.5):
		return max_height_cnt
	else:
		return -1

def generate_feasibility_mask(height_map: np.ndarray, comming_item: np.ndarray, height_limit):
	"""
	Place the item with front-left-bottom corner at (x, y)
	:param height_map: (length, width)
	:param comming_item: length, width, height (3,)
	:return: (length, width, 2)
	"""


	# Orientation 1: 
	length, width = height_map.shape
	feasibility_mask = np.zeros((2, length, width))
	li, wi, hi = comming_item
	for x in range(li - 1, length):
		for y in range(width - wi + 1):
			stable_cnt = positionally_stable(height_map[x-li+1:x+1, y:y+wi], li, wi, hi, height_limit)
			if stable_cnt > 0:
				feasibility_mask[0, x, y] = 1
	# Orientation 2:
	li, wi = wi, li
	for x in range(li - 1, length):
		for y in range(width - wi + 1):
			stable_cnt = positionally_stable(height_map[x-li+1:x+1, y:y+wi], li, wi, hi, height_limit)
			if stable_cnt > 0:
				feasibility_mask[1, x, y] = 1
	return feasibility_mask
				

def generate_state_encoding(height_map: np.ndarray, coming_item: np.ndarray, height_limit):
	"""
	:param height_map: (length, width)
	:param comming_item: length, width, height (3,)
	:return: (length, width, 6)
	"""
	length, width = height_map.shape
	# print(f"(L = {length}, W = {width}), height = {height_limit}")
	item_dim = coming_item.shape[0]
	item_mat = []
	for i in range(item_dim):
		item_mat.append(np.full((length, width), coming_item[i]))
	item_mat = np.array(item_mat)
	feasibility_mask = generate_feasibility_mask(height_map, coming_item, height_limit)
	height_map_resize = np.array([height_map])
	# print(height_map_resize.shape)
	# print(item_mat.shape)
	# print(feasibility_mask.shape)

	state_encoding = np.concatenate([height_map_resize, item_mat, feasibility_mask], axis=0).transpose(1, 2, 0)
	# print(state_encoding.shape)
	return state_encoding
	
def update_height_map(height_map: np.ndarray, item_size, item_position, item_orientation):
	"""
	:param height_map: (length, width)
	:param item_size: (3,)
	:param item_position: (2,)
	:param item_orientation: 0 or 1
	:return:
	"""
	result = height_map.copy()
	length, width = height_map.shape
	x, y = item_position
	if item_orientation == 0:
		li, wi, zi = item_size
	else:
		wi, li, zi = item_size
	# print(f"length {li}, width {wi}, height {zi}")
	# print(f"x {x}, y {y}")
	area_max_height = np.max(height_map[x-li+1:x+1, y:y+wi])
	for i in range(x - li + 1, x + 1):
		for j in range(y, y + wi):
			result[i, j] = area_max_height + zi
	# print(f"Update Height Map: area_max_height = {area_max_height}, bin height = {zi}, sum = {area_max_height + zi}")
	return result

def generate_label(map_dim, item_position, item_orientation):
	length, width = map_dim
	x, y = item_position
	result = np.zeros((length, width, 2))
	result[x, y, item_orientation] = 1
	# print(f"[{x}, {y}, {item_orientation}] = {result[x, y, item_orientation]}")
	return result

def generate_EMS(height_map, height_limit=100, clip_num=100):
	length, width = height_map.shape
	placement_list = []
	heights = []
	for i in range(length):
		for j in range(width):
			if height_map[i, j] not in heights:
				heights.append(height_map[i, j])
	heights.sort()
	# print(heights)
	# x0: max j
	height_ptr = 0
	while height_ptr < len(heights):
		map_h = height_map.copy() - heights[height_ptr]
		max_j = []
		for i in range(length):
			flag = False
			for j in range(width):
				if map_h[i, j] > 0:
					max_j.append(j-1)
					flag = True
					break
			if not flag:
				max_j.append(width - 1)
		# print(f"height = {heights[height_ptr]}, max_j = {max_j}")
		# i_ranges = []
		for i in range(len(max_j)):
			maxj = max_j[i]
			if maxj == -1:
				continue
			# print(f"i = {i}, maxj = {maxj}")
			i_ptr_1 = i-1
			while i_ptr_1 >= 0:
				if map_h[i_ptr_1, maxj] <= 0:
					if max_j[i_ptr_1] == maxj:
						max_j[i_ptr_1] = -1
					elif max_j[i_ptr_1] < maxj:
						break
					i_ptr_1 -= 1
				else:
					break
			i_ptr_2 = i+1
			while i_ptr_2 < length:
				if map_h[i_ptr_2, maxj] <= 0:
					if max_j[i_ptr_2] == maxj:
						max_j[i_ptr_2] = -1
					elif max_j[i_ptr_2] < maxj:
						break
					i_ptr_2 += 1
				else:
					break
			# i_ranges.append([i_ptr_1+1, i_ptr_2-1])
			placement_list.append((
				i_ptr_2-1, 0, heights[height_ptr],
				i_ptr_1+1, maxj, height_limit
			))
		height_ptr += 1
	# x1: min j
	height_ptr = 0
	while height_ptr < len(heights):
		map_h = height_map.copy() - heights[height_ptr]
		max_j = []
		for i in range(length):
			flag = False
			for j in range(1, width+1):
				if map_h[i, width-j] > 0:
					max_j.append(width-j+1)
					flag = True
					break
			if not flag:
				max_j.append(0)
		# i_ranges = []
		for i in range(len(max_j)):
			maxj = max_j[i]
			if maxj == width:
				continue
			i_ptr_1 = i-1
			while i_ptr_1 >= 0:
				if map_h[i_ptr_1, maxj] <= 0:
					if max_j[i_ptr_1] == maxj:
						max_j[i_ptr_1] = width
					elif max_j[i_ptr_1] > maxj:
						break
					i_ptr_1 -= 1
				else:
					break
			i_ptr_2 = i+1
			while i_ptr_2 < length:
				if map_h[i_ptr_2, maxj] <= 0:
					if max_j[i_ptr_2] == maxj:
						max_j[i_ptr_2] = width
					elif max_j[i_ptr_2] > maxj:
						break
					i_ptr_2 += 1
				else:
					break
			# i_ranges.append([i_ptr_1+1, i_ptr_2-1])
			ems_gen = (
				i_ptr_2-1, maxj, heights[height_ptr],
				i_ptr_1+1, width-1, height_limit
			)
			if ems_gen not in placement_list and \
				[*ems_gen[3:5], ems_gen[2], *ems_gen[:2], ems_gen[5]] not in placement_list:
				placement_list.append(ems_gen)
		height_ptr += 1

	# y0: max i
	height_ptr = 0
	while height_ptr < len(heights):
		map_h = height_map.copy() - heights[height_ptr]
		max_i = []
		for j in range(width):
			flag = False
			for i in range(length):
				if map_h[i, j] > 0:
					max_i.append(i-1)
					flag = True
					break
			if not flag:
				max_i.append(length - 1)
		# print(f"height = {heights[height_ptr]}, max_i = {max_i}")
		# i_ranges = []
		for j in range(len(max_i)):
			maxi = max_i[j]
			if maxi == -1:
				continue
			j_ptr_1 = j-1
			while j_ptr_1 >= 0:
				if map_h[maxi, j_ptr_1] <= 0:
					if max_i[j_ptr_1] == maxi:
						max_i[j_ptr_1] = -1
					elif max_i[j_ptr_1] < maxi:
						break
					j_ptr_1 -= 1
				else:
					break
			j_ptr_2 = j+1
			while j_ptr_2 < width:
				if map_h[maxi, j_ptr_2] <= 0:
					if max_i[j_ptr_2] == maxi:
						max_i[j_ptr_2] = -1
					elif max_i[j_ptr_2] < maxi:
						break
					j_ptr_2 += 1
				else:
					break
			# i_ranges.append([i_ptr_1+1, i_ptr_2-1])
			ems_gen = (
				maxi, j_ptr_1+1, heights[height_ptr],
				0, j_ptr_2-1, height_limit
			)
			if ems_gen not in placement_list and \
				[*ems_gen[3:5], ems_gen[2], *ems_gen[:2], ems_gen[5]] not in placement_list:
				placement_list.append(ems_gen)
		height_ptr += 1

	# y1: min i
	height_ptr = 0
	while height_ptr < len(heights):
		map_h = height_map.copy() - heights[height_ptr]
		max_i = []
		for j in range(width):
			flag = False
			for i in range(1, length+1):
				if map_h[length-i, j] > 0:
					max_i.append(length-i+1)
					flag = True
					break
			if not flag:
				max_i.append(0)
		# print(f"height = {heights[height_ptr]}, max_i = {max_i}")
		# i_ranges = []
		for j in range(len(max_i)):
			maxi = max_i[j]
			if maxi == length:
				continue
			# print(f"j={j}, maxi = {maxi}")
			j_ptr_1 = j-1
			while j_ptr_1 >= 0:
				if map_h[maxi, j_ptr_1] <= 0:
					if max_i[j_ptr_1] == maxi:
						max_i[j_ptr_1] = length
					elif max_i[j_ptr_1] > maxi:
						break
					j_ptr_1 -= 1
				else:
					break
			j_ptr_2 = j+1
			while j_ptr_2 < width:
				if map_h[maxi, j_ptr_2] <= 0:
					if max_i[j_ptr_2] == maxi:
						max_i[j_ptr_2] = length
					elif max_i[j_ptr_2] > maxi:
						break
					j_ptr_2 += 1
				else:
					break
			# i_ranges.append([i_ptr_1+1, i_ptr_2-1])
			ems_gen = (
				length-1, j_ptr_1+1, heights[height_ptr],
				maxi, j_ptr_2-1, height_limit
			)
			if ems_gen not in placement_list and \
				[*ems_gen[3:5], ems_gen[2], *ems_gen[:2], ems_gen[5]] not in placement_list:
				placement_list.append(ems_gen)
		height_ptr += 1
	placement_len = len(placement_list)
	# sort by height value
	placement_sorted = sorted(placement_list, key=lambda x: -x[2])
	# if placement_len > clip_num:
	# 	placement_sorted = placement_sorted[:clip_num]
	# else:
	# 	placement_sorted.extend([[-1 for _ in range(6)] for _ in range(clip_num - placement_len)])
	# placement_sorted = np.array(placement_sorted, dtype=np.int32)
	return placement_sorted

def generate_EMS_and_mask(height_map, coming_item, height_limit=100, clip_num=100):
	length, width = height_map.shape
	ems_placement = generate_EMS(height_map, height_limit=height_limit, clip_num=clip_num)
	mask = np.zeros((2, clip_num))
	possible_placements = []
	li, wi, hi = coming_item
	for i, placement in enumerate(ems_placement):
		if placement[0] == -1:
			continue
		# xi, yi = placement[0], placement[1]
		x_max, y_min, height, x_min, y_max = placement[0], placement[1], placement[2], placement[3], placement[4]
		if x_max - x_min + 1 >= li and y_max - y_min + 1 >= wi:
			possible_placements.append([x_min+li-1, y_min, find_min_height(height_map, (x_min+li-1, y_min), coming_item, 0), x_min, y_min+wi-1])
			possible_placements.append([x_max, y_min, find_min_height(height_map, (x_max, y_min), coming_item, 0), x_max-li+1, y_min+wi-1])
			possible_placements.append([x_min+li-1, y_max-wi+1, find_min_height(height_map, (x_min+li-1, y_max-wi+1), coming_item, 0), x_min, y_max])
			possible_placements.append([x_max, y_max-wi+1, find_min_height(height_map, (x_max, y_max-wi+1), coming_item, 0), x_max-li+1, y_max])
		if x_max - x_min + 1 >= wi and y_max - y_min + 1 >= li:
			possible_placements.append([x_min+wi-1, y_min, find_min_height(height_map, (x_min+wi-1, y_min), coming_item, 1), x_min, y_min+li-1])
			possible_placements.append([x_max, y_min, find_min_height(height_map, (x_max, y_min), coming_item, 1), x_max-wi+1, y_min+li-1])
			possible_placements.append([x_min+wi-1, y_max-li+1, find_min_height(height_map, (x_min+wi-1, y_max-li+1), coming_item, 1), x_min, y_max])
			possible_placements.append([x_max, y_max-li+1, find_min_height(height_map, (x_max, y_max-li+1), coming_item, 1), x_max-wi+1, y_max])
		# print(type(xi), type(li))
		# print(height_map.shape)
		# print(li, wi, hi)
		# print(xi-li+1, xi+1, yi, yi+wi)
	result_placements = []
	ptr = 0
	for i, placement in enumerate(possible_placements):
		li, wi, hi = coming_item
		xi, yi = placement[0], placement[1]
		mask_0 = 0
		mask_1 = 0
		if xi-li+1 >= 0 and yi+wi <= width:
			stable_cnt = positionally_stable(
				height_map[xi-li+1:xi+1, yi:yi+wi],
				li, wi, hi, height_limit
			)
			if stable_cnt > 0:
				mask_0 = 1
		
		wi, li = li, wi
		if xi-li+1 >= 0 and yi+wi <= width:
			stable_cnt = positionally_stable(
				height_map[xi-li+1:xi+1, yi:yi+wi],
				li, wi, hi, height_limit
			)
			if stable_cnt > 0:
				mask_1 = 1
		if mask_0 + mask_1 > 0:
			result_placements.append(placement)
			mask[0, ptr] = mask_0
			mask[1, ptr] = mask_1
			ptr += 1
			if ptr >= clip_num:
				break
	# if len(result_placements) < clip_num:
	# 	result_placements.extend([[-1, -1, -1, -1, -1] for _ in range(clip_num - len(result_placements))])
	result_placements.sort(key=lambda x: x[2], reverse=True)
	if len(result_placements) < clip_num:
		result_placements.extend([[0, 0, 0, 0, 0] for _ in range(clip_num - len(result_placements))])
		
	result_placements = np.array(result_placements)
	return result_placements, mask


if __name__ == '__main__':
	height_map = np.array([
		[2, 2, 1, 4, 4, 0, 0],
		[2, 2, 1, 4, 4, 0, 0],
		[0, 0, 0, 4, 4, 0, 0],
		[0, 0, 1, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 0],
	])
	coming_item = np.array([3, 1, 2])
	encoded_state = generate_state_encoding(height_map, coming_item, 10)
	print(encoded_state[:, :, -1])
	placement_list = generate_EMS(height_map)
	print(placement_list)
