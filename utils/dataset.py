from PIL import Image
from seg_utils.data_postprocess import process_landmarks_to_mask
from seg_utils.data_preprocess import create_config
from seg_utils.hybrid import Hybrid
from torch.utils.data import Dataset, random_split, Subset
from typing import Type
#from utils.transform import train_transform, val_test_transform, mask_transform
from utils.transform_kornia import train_transform, val_test_transform
import os
import pandas as pd
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path=None, dataframe=None, img_dir=None, transform=None):
        self.data = pd.read_csv(csv_path) if csv_path else dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = self.data['Image Index'].values
        self.labels = self.data.drop(columns=['Image Index', 'No Finding']).values.astype('int64')

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
    def __init__(self, csv_path=None, dataframe=None, img_dir=None, transform=None, device='cpu'):
        self.data = pd.read_csv(csv_path) if csv_path else dataframe
        self.device = device
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = self.data['id'].values
        self.labels = self.data.drop(columns=['id', 'No Finding', 'subj_id']).values.astype('int64')
        
        config, A_t, D_t, U_t = create_config(device)
        self.mask_model = Hybrid(config, D_t, U_t, A_t).to(device)
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        original_image = Image.open(img_path).convert("RGB")
        image = original_image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        # Get mask (use CPU here or adapt to your GPU setup)
        mask_image = original_image.convert("L")
        mask_image = self.mask_transform(mask_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            landmark_output = self.mask_model(mask_image)
        
        mask_output = process_landmarks_to_mask(landmark_output, image_size=224, normalized=True).to(self.device)

        image = image.to(self.device)
        combined = torch.cat([image, mask_output], dim=0)

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


def chest_xray_with_mask_datasplit(df: pd.DataFrame, full_dataset: Type[Dataset], dataset_dir="datasets", device='cpu'):
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
        ChestXrayDatasetWithMask(dataframe=df, img_dir=dataset_dir, transform=train_transform, device=device),
        train_subset.indices
    )

    val_dataset = Subset(
        ChestXrayDatasetWithMask(dataframe=df, img_dir=dataset_dir, transform=val_test_transform, device=device),
        val_subset.indices
    )

    test_dataset = Subset(
        ChestXrayDatasetWithMask(dataframe=df, img_dir=dataset_dir, transform=val_test_transform, device=device),
        test_subset.indices
    )

    return train_dataset, val_dataset, test_dataset