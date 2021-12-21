import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms







landuse = UCMercedLanduse()
landuse.get_data_extract()


class MyDataset():
    def __init__(self, img_dir, df, data_dir, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.data_dir = data_dir

    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        image = Image.open(os.path.join(self.data_dir, "images", d.image+".tif"))
        label = torch.tensor(d[1:].tolist(), dtype=torch.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        return image,label
        
    def __len__(self):
        return len(self.df)



def splitting_dataset(dataset):
    valid_no = int(len(dataset)*0.12)
    trainset ,valset  = random_split( dataset , [len(dataset) -valid_no  ,valid_no])
    return trainset, valset







