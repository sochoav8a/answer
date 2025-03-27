#data.dataset.py
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ConfocalDataset(Dataset):
    def __init__(self, lr_dir, hr_dir=None, transform_lr=None, transform_hr=None, paired=False, part_data=1.0):
        super(ConfocalDataset, self).__init__()

        self.npz_file = lr_dir
        data = np.load(self.npz_file)

        if 'arr_0' not in data or 'arr_1' not in data:
            raise RuntimeError(f"El archivo NPZ debe contener 'arr_0' y 'arr_1': {self.npz_file}")

        self.lr_images = data['arr_0']
        self.hr_images = data['arr_1']

        # Aplicar submuestreo según part_data
        if part_data < 1.0:
            if paired:
                min_size = min(len(self.lr_images), len(self.hr_images))
                n_samples = int(min_size * part_data)
                indices = np.random.permutation(min_size)[:n_samples]
                self.lr_images = self.lr_images[indices]
                self.hr_images = self.hr_images[indices]
            else:
                n_lr = int(len(self.lr_images) * part_data)
                n_hr = int(len(self.hr_images) * part_data)
                lr_indices = np.random.permutation(len(self.lr_images))[:n_lr]
                hr_indices = np.random.permutation(len(self.hr_images))[:n_hr]
                self.lr_images = self.lr_images[lr_indices]
                self.hr_images = self.hr_images[hr_indices]

        self.transform_lr = transform_lr
        self.transform_hr = transform_hr
        self.paired = paired

        self.lr_size = len(self.lr_images)
        self.hr_size = len(self.hr_images)

        if self.lr_size == 0 or self.hr_size == 0:
            raise RuntimeError(f"No se encontraron imágenes en el archivo: {self.npz_file}")

        print(f"Cargado dataset con {self.lr_size} imágenes LR y {self.hr_size} imágenes HR (part_data={part_data})")
    def __len__(self):
        return min(self.lr_size, self.hr_size) if self.paired else max(self.lr_size, self.hr_size)

    def __getitem__(self, index):
        lr_idx = index % self.lr_size
        hr_idx = index % self.hr_size if self.paired else random.randint(0, self.hr_size - 1)

        lr_img = self.lr_images[lr_idx]
        hr_img = self.hr_images[hr_idx]

        lr_img = torch.from_numpy(lr_img.astype(np.float32) / 255.0)
        hr_img = torch.from_numpy(hr_img.astype(np.float32) / 255.0)

        if lr_img.ndim == 2:
            lr_img = lr_img.unsqueeze(0)
        elif lr_img.ndim == 3 and lr_img.shape[2] == 3:
            lr_img = lr_img.permute(2, 0, 1)

        if hr_img.ndim == 2:
            hr_img = hr_img.unsqueeze(0)
        elif hr_img.ndim == 3 and hr_img.shape[2] == 3:
            hr_img = hr_img.permute(2, 0, 1)

        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)

        return {
            'LR': lr_img,
            'HR': hr_img,
            'LR_idx': lr_idx,
            'HR_idx': hr_idx
        }

def get_default_transforms(img_size=256):
    transform_list = [
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ]
    return transforms.Compose(transform_list)

def get_validation_transforms(img_size=256):
    transform_list = [
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ]
    return transforms.Compose(transform_list)
