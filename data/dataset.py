import torch
from torch.utils.data import Dataset
from PIL import Image

class DiffusionDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath'])
        image = self.transform(image)
        embedding = torch.tensor(row[2:].tolist())
        return image, embedding