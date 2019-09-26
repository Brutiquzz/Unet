import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from BatchTensorMaker import BatchTensorMaker
import sys
import os


# input er en 5 dimensionel tensor (batch size x channels x depth x height x width)
trainset = os.listdir("DataCenter/trainX")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
for batch in trainloader:
    tensor = BatchTensorMaker(batch)
    tensor.tensorMaker()
    break

input = torch.randn(32, 1, 64, 64, 64, requires_grad=True)
input2 = torch.randn(32, 1, 64, 64, 64)
#Properties som min model skal have

# Down hill1
conv1 = nn.Conv3d(1, 2, kernel_size=3,  stride=1)
conv2 = nn.Conv3d(2, 2, kernel_size=3,  stride=1)

# Down hill2
conv3 = nn.Conv3d(2, 4, kernel_size=3,  stride=1)
conv4 = nn.Conv3d(4, 4, kernel_size=3,  stride=1)

# Down hill3
conv5 = nn.Conv3d(4, 8, kernel_size=3,  stride=1)
conv6 = nn.Conv3d(8, 8, kernel_size=3,  stride=1)

# Down hill4
conv7 = nn.Conv3d(8, 16, kernel_size=3,  stride=1)
conv8 = nn.Conv3d(16, 16, kernel_size=3, stride=1)

#Down hill5
conv9 = nn.Conv3d(16, 32, kernel_size=3,  stride=1)
conv10 = nn.Conv3d(32, 32, kernel_size=3, stride=1)

#Up hill1
upConv1 = nn.Conv3d(32, 16, kernel_size=3, stride=1)
upConv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1)

#up hill2
upConv3 = nn.Conv3d(16, 8, kernel_size=3,  stride=1)
upConv4 = nn.Conv3d(8, 8, kernel_size=3,   stride=1)

#up hill3
upConv5 = nn.Conv3d(8, 4, kernel_size=3,  stride=1)
upConv6 = nn.Conv3d(4, 4, kernel_size=3,  stride=1)

#up hill4
upConv7 = nn.Conv3d(4, 2, kernel_size=3,  stride=1)
upConv8 = nn.Conv3d(2, 2, kernel_size=3,  stride=1)

#up hill5
upConv9 = nn.Conv3d(2, 1, kernel_size=3, stride=1)
upConv10 = nn.Conv3d(1, 1, kernel_size=3, stride=1)

mp = nn.MaxPool3d(kernel_size=2, stride=2)

# Et forward pass

output = torch.tensor(F.relu(conv1(tensor.batchTensorX)), requires_grad=True)
print(output.shape)
output = F.relu(conv2(output))
print(output.shape)
output = mp(output)

print(output.shape)

output = F.relu(conv3(output))
print(output.shape)
output = F.relu(conv4(output))
print(output.shape)
output = mp(output)

print(output.shape)

output = F.interpolate(output, scale_factor=2, mode='trilinear') #align_corners?

print(output.shape)
output = F.relu(upConv7(output))
print(output.shape)
output = F.relu(upConv8(output))

print(output.shape)

output = F.interpolate(output, scale_factor=2, mode='trilinear')
print(output.shape)
output = F.relu(upConv9(output))
print(output.shape)
output = Variable(F.relu(upConv10(output)), requires_grad=True)

print(output.shape)


def dice_loss(input, target):
    smooth = 1.

    iflat = input
    tflat = target


    tflat.stride = iflat.stride

    tflat = tflat.contiguous()
    iflat = iflat.contiguous()

    iflat = iflat.view(-1)
    tflat = tflat.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

loss = dice_loss(output, Variable(tensor.batchTensorY[:,:, 12:52, 12:52, 12:52], requires_grad=True))
loss.backward()
print(output.grad)
#print(conv1.weight.shape)
