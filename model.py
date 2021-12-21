import torch
from torch import optim
from torchvision import models
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 17    



def get_model(lr=3e-3,num_classes=num_classes, opt="Adam", **kwargs):
    model_imgnet = models.resnet34(pretrained=True) # model
    num_ftrs = model_imgnet.fc.in_features
    model_imgnet.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    model = model_imgnet.to(device)
#     print(model)
    if opt=="SGD":
        return model, optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif opt=="Adam":
        return model, optim.Adam(model.parameters(), lr=lr, **kwargs)
    else:
        return model, optim.RMSprop(model.parameters(), lr=lr, **kwargs)