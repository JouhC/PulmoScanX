from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import pandas as pd

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
