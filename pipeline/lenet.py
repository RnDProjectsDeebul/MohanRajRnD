import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
    

class LeNet(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.use_dropout = dropout
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(20000, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# import torch
# from torch.quantization import quantize_fx 

# model = LeNet()
# img = torch.rand(128,1,28,28)
# a = model(img)
# print("standatd model success")



# model = LeNet()
# model.eval()
# qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
# img = torch.rand(128,1,28,28)
# model_prepared = quantize_fx.prepare_fx(model, qconfig_dict, img)

# with torch.inference_mode():
#     for _ in range(10):
#         model = quantize_fx.convert_fx(model_prepared)
        
# model.eval()
# a = model(img)
# print("quantised model success")