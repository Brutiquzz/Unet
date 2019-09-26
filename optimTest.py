
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from Unet import Unet
#torch.manual_seed(20)
unet = Unet()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.1)
loss_fn = torch.nn.BCEWithLogitsLoss()
input =  torch.randn(32, 1, 64, 64, 64 , requires_grad=True)
target = torch.randint(low=0, high=2, size = (32, 1, 64, 64, 64),  requires_grad=False )
optimizer.zero_grad()
y_pred = unet(input)
print(y_pred.shape)
y = target[: , : , 20:44, 20:44, 20:44]
loss = loss_fn(y_pred, y)
print(type(loss))
print loss.item()
weight = unet.conv1.weight.data.clone()
print(unet.conv1.weight.data)
loss.backward()
optimizer.step()
print(unet.conv1.weight.data)
print(torch.equal(unet.conv1.weight.data,weight))
