# imports...
import os
import nibabel as nib
import torch
import numpy as np
class BatchTensorMaker:

    def __init__(self, batch, outputSize, patch_size, cuda):
        self.batch = batch
        self.batchTensorTrainX = 0
        self.batchTensorTrainY = 0
        self.batchTensorTestX  = 0
        self.batchTensorTestY  = 0
        self.outputSize = outputSize
        self.patch_size = patch_size
        self.cuda = cuda


# makes the batch tensor 5 diemensional
    def tensorMakerTrain(self):

        counter = 0

        tensorX = torch.empty(len(self.batch), 1, self.patch_size,self.patch_size,self.patch_size)
        tensorY = torch.empty(len(self.batch), 1, self.outputSize, self.outputSize, self.outputSize)

        for file in self.batch:
            pathX = os.path.join("Unet/DataCenter/trainX", file)
            pathY = os.path.join("Unet/DataCenter/trainY", file.replace("X", "Y"))

            niiX = nib.load(pathX)
            niiY = nib.load(pathY)

            dataX = np.asarray(niiX.dataobj)
            dataY = torch.tensor(np.asarray(niiY.dataobj))

            tensX = torch.tensor(dataX.astype(float), requires_grad=True)
            tensY = self.Crop(dataY, tensorY.shape)

            tensorX[counter][0] = tensX
            tensorY[counter][0] = tensY

            counter = counter + 1

        self.batchTensorTrainX = tensorX #+ 1000
        self.batchTensorTrainY = tensorY

        if self.cuda:
            self.batchTensorTrainX.cuda()
            self.batchTensorTrainY.cuda()





    def tensorMakerTest(self):
        counter = 0

        tensorX = torch.empty(len(self.batch), 1, self.patch_size,self.patch_size,self.patch_size)
        tensorY = torch.empty(len(self.batch), 1, self.outputSize,self.outputSize,self.outputSize)

        for file in self.batch:
            pathX = os.path.join("Unet/DataCenter/testX", file)
            pathY = os.path.join("Unet/DataCenter/testY", file.replace("X", "Y"))

            niiX = nib.load(pathX)
            niiY = nib.load(pathY)

            dataX = np.asarray(niiX.dataobj)
            dataY = torch.tensor(np.asarray(niiY.dataobj))

            tensX = torch.tensor(dataX.astype(float), requires_grad=False)
            tensY = self.Crop(dataY, tensorY.shape)

            tensorX[counter][0] = tensX
            tensorY[counter][0] = tensY

            counter = counter + 1

        self.batchTensorTestX = tensorX #+ 1000
        self.batchTensorTestY = tensorY

        if self.cuda:
            self.batchTensorTestX.cuda()
            self.batchTensorTestY.cuda()

    def Crop(self, input, target):
        width = target[2]
        inputShape = input.shape
        diff = (inputShape[2] - width)/2
        start = diff
        end = inputShape[2]-diff

        return input[start:end, start:end, start:end]
