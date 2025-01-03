import json
import os
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class BinPackingDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Args:
            data_dir (str): Directory containing the JSON dataset files.
            split (str): 'train' or 'test'.
        """
        self.data = []
        file_names = [f for f in os.listdir(data_dir) if f.endswith('.json')]

        for file_name in file_names:
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'r') as f:
                dataset = json.load(f)
                self.data.append(dataset)

        # Split the data
        random.shuffle(self.data)
        split_idx = int(0.8 * len(self.data))
        if split == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataset = self.data[idx]
        items = dataset['items']

        # Extract features: dimensions and rotations
        dimensions = []
        rotations = []
        for item in items:
            dim = item['dimensions']  # (x, y, z)
            rot = item['rotation']    # (rot_x, rot_y, rot_z)
            dimensions.append(dim)
            rotations.append(rot)

        dimensions = torch.tensor(dimensions, dtype=torch.float)  # Shape: [N, 3]
        rotations = torch.tensor(rotations, dtype=torch.float)    # Shape: [N, 3]

        # The target can be the positions and rotations
        positions = []
        for item in items:
            pos = item['position']  # (x, y, z)
            positions.append(pos)

        positions = torch.tensor(positions, dtype=torch.float)  # Shape: [N, 3]

        return dimensions, rotations, positions


# Parameters
data_directory = 'dataset'  # Directory where datasets are stored
batch_size = 16

# Create datasets
train_dataset = BinPackingDataset(data_dir=data_directory, split='train')
test_dataset = BinPackingDataset(data_dir=data_directory, split='test')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x))

def collate_fn(batch):
    """
    Custom collate function to handle variable number of items.
    Pads sequences to the maximum length in the batch.
    """
    dimensions, rotations, positions = zip(*batch)
    lengths = [d.size(0) for d in dimensions]
    max_len = max(lengths)
    print(max_len)

    # Pad dimensions, rotations, positions
    padded_dimensions = torch.stack([torch.cat([d, torch.zeros(max_len - d.size(0), 3)]) for d in dimensions])
    padded_rotations = torch.stack([torch.cat([r, torch.zeros(max_len - r.size(0), 3)]) for r in rotations])
    padded_positions = torch.stack([torch.cat([p, torch.zeros(max_len - p.size(0), 3)]) for p in positions])

    # Create masks
    masks = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        masks[i, :length] = 1

    return padded_dimensions, padded_rotations, padded_positions, masks, lengths