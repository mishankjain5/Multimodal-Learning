import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import fiftyone as fo

class CILPDataset(Dataset):
    def __init__(self, fo_dataset, split="train", transform=None):
        self.transform = transform

        # RGB samples (one per group)
        self.rgb_samples = list(
            fo_dataset
            .select_group_slices("rgb")
            .match_tags(split)
        )

        # Build group_id -> lidar filepath map
        lidar_samples = (
            fo_dataset
            .select_group_slices("lidar")
            .match_tags(split)
        )

        self.lidar_map = {
            s.group.id: s.filepath for s in lidar_samples
        }

    def __len__(self):
        return len(self.rgb_samples)

    def __getitem__(self, idx):
        sample = self.rgb_samples[idx]

        # RGB
        rgb = Image.open(sample.filepath).convert("RGB")
        if self.transform:
            rgb = self.transform(rgb)

        # LiDAR (robust lookup)
        group_id = sample.group.id
        if group_id not in self.lidar_map:
            raise RuntimeError(f"LiDAR missing for group {group_id}")

        lidar = np.load(self.lidar_map[group_id])
        lidar = torch.tensor(lidar, dtype=torch.float32)

        # Label
        label = 0 if sample.label.label == "cubes" else 1
        label = torch.tensor(label, dtype=torch.long)

        return rgb, lidar, label
