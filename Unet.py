import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib

class Unet(nn.Module):

    def __init__(self, weightAmount=1):
      super(Unet, self).__init__()

      # Down hill1
      self.conv1 = nn.Conv3d(1, 2*weightAmount, kernel_size=3,  stride=1)
      self.conv2 = nn.Conv3d(2*weightAmount, 2*weightAmount, kernel_size=3,  stride=1)

      # Down hill2
      self.conv3 = nn.Conv3d(2*weightAmount, 4*weightAmount, kernel_size=3,  stride=1)
      self.conv4 = nn.Conv3d(4*weightAmount, 4*weightAmount, kernel_size=3,  stride=1)

      #bottom
      self.convbottom1 = nn.Conv3d(4*weightAmount, 8*weightAmount, kernel_size=3,  stride=1)
      self.convbottom2 = nn.Conv3d(8*weightAmount, 8*weightAmount, kernel_size=3,  stride=1)

      # upsampling conv
      self.upsampconv1 = nn.Conv3d(8*weightAmount, 4*weightAmount, kernel_size=1, stride=1)

      #up hill4
      self.upConv0 = nn.Conv3d(8*weightAmount, 4*weightAmount, kernel_size=3,  stride=1)
      self.upConv1 = nn.Conv3d(4*weightAmount, 4*weightAmount, kernel_size=3,  stride=1)

      #upsampling conv
      self.upsampconv2 = nn.Conv3d(4*weightAmount, 2*weightAmount, kernel_size=1, stride=1)

      #up hill5
      self.upConv2 = nn.Conv3d(4*weightAmount, 2*weightAmount, kernel_size=3,  stride=1)
      self.upConv3 = nn.Conv3d(2*weightAmount, 2*weightAmount, kernel_size=3, stride=1)

      # down to 1 weight
      self.upConv4 = nn.Conv3d(2*weightAmount, 1, kernel_size=1, stride=1)

      self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

      #properties to be used:
      self.evaluation = []
      self.trainingEvaluation = []
      self.nextEpoch = True

      # state save
      self.bestState = 0

      # layers
      self.output = 0
      self.output2 = 0
      self.output3 = 0
      self.output4 = 0
      self.output5 = 0
      self.output6 = 0
      self.output7 = 0
      self.output8 = 0
      self.output9 = 0
      self.output10 = 0
      self.output11 = 0
      self.output12 = 0
      self.output13 = 0
      self.output14 = 0
      self.output15 = 0
      self.output16 = 0
      self.output17 = 0
      self.output18 = 0



    # Use this function to update the filters of the object
    def forward(self, input, printDimensions=False):
       self.output = F.leaky_relu(self.conv1(input))
       self.output2 = F.leaky_relu(self.conv2(self.output))

       self.output = self.mp(self.output2)

       self.output = F.leaky_relu(self.conv3(self.output))
       self.output5 = F.leaky_relu(self.conv4(self.output))

       self.output = self.mp(self.output5)

       self.output = F.leaky_relu(self.convbottom1(self.output))
       self.output = F.leaky_relu(self.convbottom2(self.output))

       self.output = F.interpolate(self.output, scale_factor=2, mode='trilinear')
       self.output = F.leaky_relu(self.upsampconv1(self.output))
       self.output = torch.cat((self.output, self.Crop(self.output5, self.output.shape) ), 1)

       self.output = F.leaky_relu(self.upConv0(self.output))
       self.output = F.leaky_relu(self.upConv1(self.output))

       self.output = F.interpolate(self.output, scale_factor=2, mode='trilinear')
       self.output = F.leaky_relu(self.upsampconv2(self.output))
       self.output = torch.cat((self.output, self.Crop(self.output2, self.output.shape) ), 1)

       self.output = F.leaky_relu(self.upConv2(self.output))
       self.output = F.leaky_relu(self.upConv3(self.output))

       self.output = torch.sigmoid(self.upConv4(self.output))

       return self.output


# this function crops an input into the target shape. It is used for skip connections
    def Crop(self, input, target):
        width = target[2]
        inputShape = input.shape
        diff = (inputShape[2] - width)/2
        start = diff
        end = inputShape[2]-diff

        return input[:, :, start:end, start:end, start:end]


    def forwardold(self, input):
        # Use U-net Theory to Update the filters.
        # Example Approach...
        self.output1 = F.leaky_relu(self.conv1(input))
        self.output2 = F.leaky_relu(self.conv2(self.output1))

        self.output3 = self.mp(self.output2)

        self.output4 = F.leaky_relu(self.conv3(self.output3))
        self.output5 = F.leaky_relu(self.conv4(self.output4))

        self.output6 = self.mp(self.output5)

        self.output7 = F.leaky_relu(self.convbottom1(self.output6))
        self.output8 = F.leaky_relu(self.convbottom2(self.output7))

        self.output9 = F.interpolate(self.output8, scale_factor=2, mode='trilinear')

        self.output10 = F.leaky_relu(self.upConv0(self.output9))
        self.output11 = F.leaky_relu(self.upConv1(self.output10))

        self.output12 = F.interpolate(self.output11, scale_factor=2, mode='trilinear')


        self.output13 = F.leaky_relu(self.upConv2(self.output12))
        self.output14 = F.leaky_relu(self.upConv3(self.output13))

        return F.leaky_relu(self.upConv4(self.output14))


# copy pasted dice loss function
    def dice_loss(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))



    def EvaluateNextEpoch(self, unet):
        best = min(self.evaluation)
        length = len(self.evaluation)
        # save the best state
        if self.evaluation[length-1] == best:
            print("We have a new best: " + str(self.evaluation[length-1]))
            torch.save(self.state_dict(), os.path.join("Unet/DataCenter/BestState", "bestState.pth"))

        # check if we should run another Epoch
        if (length > 5 and self.evaluation.index(best) == length - 6
        and self.evaluation[length-1] >= best
        and self.evaluation[length-2] >= best
        and self.evaluation[length-3] >= best
        and self.evaluation[length-4] >= best
        and self.evaluation[length-5] >= best):
            self.nextEpoch = False
            t = np.array(self.evaluation)
            t2 = np.array(self.trainingEvaluation)
            fig, ax = plt.subplots()
            ax.plot(t, label="Validation Loss")
            ax.plot(t2, label="Training Loss")
            ax.set(xlabel='Epoch', ylabel='Epoch loss',
                   title='Epoch Loss Illustration')
            ax.grid()
            fig.savefig("test.png")
            plt.show()
            print(self.evaluation)
            print(length)
        else:
            print(self.evaluation[length-1])
            print("Running Epoch number: " + str(length+1))
