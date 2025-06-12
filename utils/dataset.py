from PIL import Image
from torch.utils.data import Dataset, random_split, Subset
from typing import Type
from utils.transform import train_transform, val_test_transform
import os
import pandas as pd
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path=None, dataframe=None, img_dir=None, transform=None):
        self.data = pd.read_csv(csv_path) if csv_path else dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = self.data['id'].values
        self.labels = self.data.drop(columns=['id', 'No Finding', 'subj_id']).values.astype('int64')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


class ChestXrayDatasetWithMask(Dataset):
    def __init__(self, csv_path=None, dataframe=None, img_dir=None, mask_model=None, transform=None):
        self.data = pd.read_csv(csv_path) if csv_path else dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = self.data['id'].values
        self.labels = self.data.drop(columns=['id', 'No Finding', 'subj_id']).values.astype('int64')
        self.mask_model = mask_model

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        # Get mask (use CPU here or adapt to your GPU setup)
        with torch.no_grad():
            mask_tensor = self.mask_model(image.unsqueeze(0)).squeeze(0)
            mask_tensor = torch.sigmoid(mask_tensor)  # apply sigmoid if logits
            mask_tensor = (mask_tensor > 0.5).float()

        # Ensure mask is [1, H, W]
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        combined = torch.cat([image, mask_tensor], dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return combined, label


def chest_xray_datasplit(df: pd.DataFrame, full_dataset: Type[Dataset], dataset_dir="datasets"):
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = Subset(
        ChestXrayDataset(dataframe=df, img_dir=dataset_dir, transform=train_transform),
        train_subset.indices
    )

    val_dataset = Subset(
        ChestXrayDataset(dataframe=df, img_dir=dataset_dir, transform=val_test_transform),
        val_subset.indices
    )

    test_dataset = Subset(
        ChestXrayDataset(dataframe=df, img_dir=dataset_dir, transform=val_test_transform),
        test_subset.indices
    )

    return train_dataset, val_dataset, test_dataset