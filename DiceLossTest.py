import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

input2 =  torch.randint(low=0, high=2, size = (32, 1, 24, 24, 24) , requires_grad=True)
target2 = torch.randint(low=0, high=2, size = (32, 1, 24, 24, 24),  requires_grad=False )

def dice_loss(input, target):
    smooth = 1.
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for patch in range(0,32,1):
        for width in range(0,24,1):
            for height in range(0,24,1):
                for depth in range(0,24,1):
                    inputPoint = input[patch, 0, width, height, depth] == target[patch, 0 , width, height, depth]
                    targetPoint = target[patch, 0, width, height, depth] == target[patch, 0 , width, height, depth]
                    if inputPoint == targetPoint and inputPoint == 0:
                        tp += 1
                    if inputPoint == targetPoint and inputPoint == 1:
                        tn += 1
                    if inputPoint != targetPoint and inputPoint == 0:
                        fp += 1
                    if inputPoint != targetPoint and inputPoint == 1:
                        fn += 1
    result = (float((2 * tp)) + smooth) / (float((2 * tp + fp + fn)) + smooth)
    return result - 1


def dice_loss2(input, target):
    smooth = 1.

    iflat = input
    tflat = target

    iflat = iflat.view(-1)
    tflat = tflat.view(-1)
    intersection = (iflat * tflat).sum()

    result = 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
    #result.requires_grad = True

    return result


#print (dice_loss(input2, target2))
print(dice_loss2(input2, input2))
