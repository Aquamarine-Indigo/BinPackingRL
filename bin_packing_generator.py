import random
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

@dataclass
class Item:
    id: int
    dimensions: Tuple[int, int, int]  # (x, y, z) 维度，均为整数
    position: Tuple[int, int, int] = (0, 0, 0)  # 在容器中的位置 (x, y, z)
    rotation: Tuple[int, int, int] = (0, 0, 0)  # 绕 (x, y, z) 轴的旋转角度（度）

def split_item(item: Item, split_axis: str, split_pos: int, next_id: int) -> Tuple[Item, Item]:
    x, y, z = item.dimensions
    if split_axis == 'x':
        dim1 = (split_pos, y, z)
        dim2 = (x - split_pos, y, z)
    elif split_axis == 'y':
        dim1 = (x, split_pos, z)
        dim2 = (x, y - split_pos, z)
    else:  # 'z'
        dim1 = (x, y, split_pos)
        dim2 = (x, y, z - split_pos)
    
    # 随机旋转，限制为0°, 90°, 180°, 270°
    rot_options = [0, 90, 180, 270]
    rot1 = (random.choice(rot_options),
            random.choice(rot_options),
            random.choice(rot_options))
    rot2 = (random.choice(rot_options),
            random.choice(rot_options),
            random.choice(rot_options))
    
    # 创建新的物品实例
    new_item1 = Item(id=next_id, dimensions=dim1, rotation=rot1)
    new_item2 = Item(id=next_id+1, dimensions=dim2, rotation=rot2)
    return new_item1, new_item2

def bin_packing_generator(save_file: str = "bin_packing_dataset.json"):
    # 初始化一个大物品
    initial_item = Item(id=1, dimensions=(100, 100, 100))
    items: List[Item] = [initial_item]
    
    # 随机选择N在10到50之间
    N = random.randint(10, 50)
    next_id = 2  # 下一个物品的ID
    
    while len(items) < N:
        # 随机选择一个物品进行分割
        current_item = random.choice(items)
        items.remove(current_item)
        
        x, y, z = current_item.dimensions
        # 根据最长边选择分割轴
        if x >= y and x >= z:
            axis = 'x'
        elif y >= x and y >= z:
            axis = 'y'
        else:
            axis = 'z'
        
        # 选择分割位置，确保至少保留25%的尺寸
        if axis == 'x':
            split_min = max(1, int(x * 0.25))
            split_max = max(split_min + 1, int(x * 0.75))
            split_pos = random.randint(split_min, split_max)
        elif axis == 'y':
            split_min = max(1, int(y * 0.25))
            split_max = max(split_min + 1, int(y * 0.75))
            split_pos = random.randint(split_min, split_max)
        else:
            split_min = max(1, int(z * 0.25))
            split_max = max(split_min + 1, int(z * 0.75))
            split_pos = random.randint(split_min, split_max)
        
        # 分割物品
        item1, item2 = split_item(current_item, axis, split_pos, next_id)
        next_id += 2
        
        # 根据分割轴更新位置
        pos1 = list(current_item.position)
        pos2 = list(current_item.position)
        if axis == 'x':
            pos2[0] += item1.dimensions[0]
        elif axis == 'y':
            pos2[1] += item1.dimensions[1]
        else:
            pos2[2] += item1.dimensions[2]
        
        item1.position = tuple(pos1)
        item2.position = tuple(pos2)
        
        # 添加新物品到列表
        items.extend([item1, item2])
    
    # 如果超过N，随机移除多余的物品
    if len(items) > N:
        items = items[:N]
    
    # 计算利用率
    total_volume = sum(item.dimensions[0] * item.dimensions[1] * item.dimensions[2] for item in items)
    container_volume = 100 * 100 * 100  # 初始容器体积
    utilization_rate = total_volume / container_volume  # 利用率
    
    # 准备保存的数据
    dataset = {
        "items": [asdict(item) for item in items],
        "N": N,
        "utilization_rate": utilization_rate
    }
    
    # 保存为JSON文件
    with open(save_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"数据集已保存到 {save_file}")
    print(f"生成的利用率 (Utilization Rate): {utilization_rate:.2%}")

# 示例用法
if __name__ == "__main__":
    for i in range(10):
        bin_packing_generator(f"dataset/bin_packing_dataset_{i}.json")
        
