import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, DATASET_PREFIX


class AirfoilDataset(Dataset):
    def __init__(
        self,
        coord_path=f"./dataset/{DATASET_PREFIX}_coords_array.npy",
        cl_path=f"./dataset/{DATASET_PREFIX}_cl_array.npy",
        norm_path=f"./dataset/{DATASET_PREFIX}_normalization_stats.npz",
        normalize=True,
    ):
        coords_array = np.load(coord_path).astype(np.float32)  # shape: (N, 2, 248)
        cls_array = np.load(cl_path).astype(np.float32)[:, np.newaxis]  # shape: (N, 1)

        norm = np.load(norm_path)
        self.coord_mean = norm["coord_mean"]
        self.coord_std = norm["coord_std"]
        self.cl_mean = norm["cl_mean"][0]
        self.cl_std = norm["cl_std"][0]

        if normalize:
            coords_array = (coords_array - self.coord_mean) / self.coord_std
            cls_array = (cls_array - self.cl_mean) / self.cl_std

        self.coords_tensor = torch.tensor(coords_array, dtype=torch.float32)
        self.cls_tensor = torch.tensor(cls_array, dtype=torch.float32)
        self.normalize = normalize

    def __len__(self):
        return self.coords_tensor.shape[0]

    def __getitem__(self, idx):
        return self.coords_tensor[idx], self.cls_tensor[idx]

    def denormalize_coord(self, coord_tensor):
        std = torch.tensor(self.coord_std, dtype=torch.float32, device=coord_tensor.device)
        mean = torch.tensor(self.coord_mean, dtype=torch.float32, device=coord_tensor.device)
        return coord_tensor * std + mean

    def normalize_cl(self, cl_tensor):
        std = torch.tensor(self.cl_std, dtype=torch.float32, device=cl_tensor.device)
        mean = torch.tensor(self.cl_mean, dtype=torch.float32, device=cl_tensor.device)
        return (cl_tensor - mean) / std

    def denormalize_cl(self, cl_tensor):
        std = torch.tensor(self.cl_std, dtype=torch.float32, device=cl_tensor.device)
        mean = torch.tensor(self.cl_mean, dtype=torch.float32, device=cl_tensor.device)
        return cl_tensor * std + mean


def get_dataloader(batch_size=BATCH_SIZE, shuffle=True):
    dataset = AirfoilDataset(normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset
