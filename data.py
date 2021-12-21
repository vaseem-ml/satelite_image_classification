import os
import torch
import zipfile
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm_notebook as tqdm

import math


stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset
    
    def __repr__(self) -> str:
        return str(self.__class__.__name__)+" obj (train & valid dataloaders)"




class UCMercedLanduse():
    def __init__(self):
        self.train_ds, self.valid_ds = [None]*2
        
    def get_data_extract(self):
        if "UCMerced" in os.listdir():
            print("Dataset already exists")
        else:
            # print("Downloading Data...")
            # download_url("https://drive.google.com/file/d/1Bm2kIUlA0Y6OpE27y-UY27VtXRoBXz8M/view?usp=sharing", root="data/")
            # print("Dataset Downloaded")
            print("Extracting data..")
            with zipfile.ZipFile("UCMerced.zip", 'r') as zip_ref:
                zip_ref.extractall("./")
            print("Extraction done!")
    def _get_tfms(self):
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)) , 
            transforms.ToTensor(),
            transforms.Normalize(*stats, inplace=True)])

        return transform
        
        


class MyDataset(Dataset):
    def __init__(self, img_dir, df, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        
    
    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, "images", d.image+".tif"))
        label = torch.tensor(d[1:].tolist(), dtype=torch.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        return image,label
        
    def __len__(self):
        return len(self.df)




def spliting_dataset(dataset):
    valid_no = int(len(dataset)*0.12) 
    trainset ,valset  = random_split( dataset , [len(dataset) -valid_no  ,valid_no])
    return trainset, valset


def get_dls(train_ds, valid_ds, bs, **kwargs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
               DataLoader(valid_ds, batch_size=bs//2, shuffle=False, **kwargs))

def cal_mean_std(train_data):
    return np.mean(train_data, axis=(0,1,2))/255, np.std(train_data, axis=(0,1,2))/255



def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
        break







def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr






def fit(epochs: int, learn) -> None:
    pass




class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.data = data
        self.base_lr=1e-5
        self.max_lr = 100
        self.bn = len(data.train_dl) - 1
        ratio = self.max_lr/self.base_lr
        self.q = ratio**(1/self.bn)
        self.best_loss = 1e9
        self.iteration = 0
        self.lrs=[]
        self.losses=[]
    
    def calc_lr(self, loss):
        self.iteration+=1
        if math.isnan(loss) or loss >  4 * self.best_loss:
            return -1
        if loss < self.best_loss and self.iteration > 1:
            self.best_loss = loss
        
        q = self.q ** self.iteration
        lr = self.base_lr * q
        self.lrs.append(lr) # append the learing rate to lrs
        self.losses.append(loss) # append the loss to losses
        return lr

    def plot(self, start=10, end=-5): # plot lrs vs losses
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log') # learning rates are in log scale
        if not os.path.exists('plots'): os.makedirs("plots")
        plt.savefig("plots/lr_plot.png")


    def find_LR(self):
        t = tqdm(self.data.train_dl, leave=False, total=len(self.data.train_dl))
        self.model.train()
        running_loss = 0
        avg_beta = 0.98
        for i, (inputs, target) in enumerate(t):
            inputs, target = inputs.cpu(), target.cpu()
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, target)
            running_loss = avg_beta * running_loss + (1-avg_beta) *loss
            smoothed_loss = running_loss / (1 - avg_beta**(i+1))
            t.set_postfix(loss=smoothed_loss)
            lr = self.calc_lr(smoothed_loss)
    #         print(lr)
            if lr == -1:
                break
            update_lr(self.opt, lr) 
            # compute gradient and do SGD step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            print('current loss', smoothed_loss)
            
            # break


    def fit():
        pass




#https://github.com/pangwong/pytorch-multi-label-classifier/blob/master/multi_label_classifier.py
#https://github.com/jarvislabsai/blog/blob/master/build_resnet34_pytorch/Building%20Resnet%20in%20PyTorch.ipynb
#https://gist.github.com/nikogamulin/7774e0e3988305a78fd73e1c4364aded