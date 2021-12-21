import os
from data import MyDataset
from models.model import load_model
import torch
import torchvision
import numpy as np
import pandas as pd
# from torch import nn
# from PIL import Image
# import tifffile as tif
# from torch import optim
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid
# from torch.autograd import Variable
# from tqdm import tqdm_notebook as tqdm
# from torchvision import datasets, models, transforms
# from torch.utils.data import Dataset, DataLoader, random_split


import argparse
import logging
import pandas as pd
# from data import MyDataset, UCMercedLanduse, spliting_dataset
from data import *
from torch import nn






def main():
    #Create the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add the arguments
    parser.add_argument("--mode", required=False, default='train', help='run mode of training or testing. [Train | Test | train | test]')
    parser.add_argument("--data_dir", required=False, default='./UCMerced', help='dataset directory')
    parser.add_argument("--model_dir", required=False, default='model', help='model directory')
    parser.add_argument("--input_channel", required=False, default=3, help='Input Channel')
    parser.add_argument("--pretrained", required=False, default=True, help='Pretrained Model')
    parser.add_argument("--input_size", required=False, default=224, help='Input size')


    # Parse the argument
    args = parser.parse_args()


    if args.mode=="train":
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        
        log_dir = args.model_dir
        log_path = log_dir+"/train.log"
    

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    log_level = logging.INFO
    logging.getLogger().setLevel(log_level)


    # Load data
    df = pd.read_csv(os.path.join(args.data_dir, 'multilabels.txt'), sep="\t")
    class_count = pd.DataFrame(df.sum(axis=0)).reset_index()
    class_count.columns = ["class", "count"]
    class_count.drop(class_count.index[0], inplace=True)
    imgIdx = df.iloc[100]
    df = df.rename(columns={'IMAGE\LABEL': 'image'})

    dataset = UCMercedLanduse()
    # to extract data
    dataset.get_data_extract()
    
    transforms = dataset._get_tfms()

    dataset = MyDataset(args.data_dir, df, transforms=transforms)
    trainset, valset = spliting_dataset(dataset)
    train_dl, valid_dl = get_dls(trainset, valset, 64, num_workers=2)
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    data = DataBunch(train_dl, valid_dl)

    model = load_model(args, len(class_count))

    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0)
    loss_func = nn.CrossEntropyLoss()
    bceLoss = nn.BCEWithLogitsLoss()
    learn = Learner(model=model, opt=opt, loss_func=bceLoss, data=data)
    learn.find_LR()
    learn.plot()






if __name__ == "__main__":
    main()