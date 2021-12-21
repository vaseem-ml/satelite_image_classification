import torch
from torch import nn
from .resnet import Resnet18Template
from .build_model import BuildMultiLabelModel, LoadPretrainedModel

def load_model(opt, num_classes):
    # if opt.model == "Resnet18":
    model = Resnet18Template(opt.input_channel, opt.pretrained)
    # tmp_input = torch.FloatTensor(1, opt.input_channel, opt.input_size, opt.input_size)
    # tmp_output = template(tmp_input)
    # output_dim = int(tmp_output.size()[-1])
    # print("this is output dim", output_dim)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # model = BuildMultiLabelModel(template, output_dim, num_classes)

    # return template
    return model





# model = load_model()

# y = model(torch.randn(4, 3, 224, 224))









