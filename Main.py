import torch
import torch.utils.data
import nibabel as nib
import numpy as np
import sys
from nifti_data_slicer import Nifti_Slicer
from BatchTensorMaker import BatchTensorMaker
from Unet import Unet
import os
from torch.autograd import Variable
import random as ra
import re
import matplotlib.pyplot as plt
import time
import pickle


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for x in range(len(y_hat)):
        for y in range(len(y_hat[0])):
            for a in range(len(y_hat[1])):
                for z in range(len(y_hat[2])):
                    if y_actual[x,0,y,a,z]==y_hat[x,0,y,a,z] and y_hat[x,0,y,a,z]==1:
                       TP += 1
                    if y_hat[x,0,y,a,z]==1 and y_actual[x,0,y,a,z]!=y_hat[x,0,y,a,z]:
                       FP += 1
                    if y_actual[x,0,y,a,z]==y_hat[x,0,y,a,z] and y_hat[x,0,y,a,z]==0:
                       TN += 1
                    if y_hat[x,0,y,a,z]==0 and y_actual[x,0,y,a,z]!=y_hat[x,0,y,a,z]:
                       FN += 1
    return(TP, FP, TN, FN)

#region properties to be set manually

# how many .nii files are in the training set
no_trainset  = 0
no_testset = 16
patch_size = 64
# If you need to patch data set to true else false
patching = False

# Specifies if you wish to send the tensors to GPU
CUDA = False

# Specify if you want to train or not
training = False

# Specify if you want to run the test set or not
test = True

# load state
loadState = False

loadHistory = False
#endregion

#region start the slicing of data in the dataset
if patching:
    slicer = Nifti_Slicer(no_trainset, no_testset)
    #slicer.renameData()
    slicer.patch_data_factory(patch_size)
    print("Done patching data - Uncheck me!")
    sys.exit()
#endregion

#TODO data augmenting here or implement it inside the Nifti_Slicer class

#region Run the U-net

# Create the U-net
unet = Unet(weightAmount=1)
if CUDA:
    unet.cuda()
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)
weight   = 0
weight2  = 0
weight3  = 0
weight4  = 0
weight5  = 0
weight6  = 0
weight7  = 0
weight8  = 0
weight9  = 0
weight10 = 0
weight11 = 0
weight12 = 0
outputSize = len((unet.forward((torch.randint(low=0, high=2, size = (1, 1, patch_size, patch_size, patch_size) )).to(torch.float32)))[0][0][0])

if loadState:
    unet.load_state_dict(torch.load(os.path.join("Unet/DataCenter/BestState", "bestState.pth")))

if loadHistory:
    with open ('Unet/DataCenter/validationfile', 'rb') as fp:
        unet.evaluation = pickle.load(fp)
    with open ('Unet/DataCenter/trainingfile', 'rb') as tp:
        unet.trainingEvaluation = pickle.load(tp)

# Run epochs until error goes up again.
if training:
    while unet.nextEpoch:
        start = time.time()
        validationLoss = []
        trainingLoss = []
        trainset = os.listdir("Unet/DataCenter/trainX")
        #random  = ra.sample(xrange(16), 4)
        trainloader = [x for x in trainset if int(filter(str.isdigit, x[0:18])) >= 5]
        validationLoader = [x for x in trainset if int(filter(str.isdigit, x[0:18])) < 5]
        weightMov = False
        first = False
    #    for x in random:
    #         validationLoader = validationLoader + [elem for elem in trainset if x == int(re.findall(r'\d+', elem[0:22])[0])]
    #    trainloader = [x for x in trainset if x not in validationLoader]
        del(trainset)
        trainloader = torch.utils.data.DataLoader(trainloader, batch_size=16, shuffle=True, num_workers=8)
        for batch in trainloader :
            tensor = BatchTensorMaker(batch, outputSize, patch_size, CUDA)
            tensor.tensorMakerTrain()
            optimizer.zero_grad()

            y_pred = unet.forward(tensor.batchTensorTrainX)
            y = tensor.batchTensorTrainY
            loss = loss_fn(y_pred, y)
            trainingLoss.append(float(loss.item()))
            loss.backward()
            optimizer.step()
            #print(unet.conv1.weight.data)
            if first:
                if (torch.equal(unet.conv1.weight.data,        weight) != True
                and torch.equal(unet.conv2.weight.data,       weight2) != True
                and torch.equal(unet.conv3.weight.data,       weight3) != True
                and torch.equal(unet.conv4.weight.data,       weight4) != True
                and torch.equal(unet.convbottom1.weight.data, weight5) != True
                and torch.equal(unet.convbottom2.weight.data, weight6) != True
                and torch.equal(unet.upsampconv1.weight.data, weight7) != True
                and torch.equal(unet.upConv0.weight.data,     weight8) != True
                and torch.equal(unet.upConv1.weight.data,     weight9) != True
                and torch.equal(unet.upsampconv2.weight.data,weight10) != True
                and torch.equal(unet.upConv2.weight.data,    weight11) != True
                and torch.equal(unet.upConv3.weight.data,    weight12) != True
                and torch.equal(unet.upConv4.weight.data,    weight13) != True):
                    weightMov = True
                    first = False

            first = True
            weight   =       unet.conv1.weight.data.clone()
            weight2  =       unet.conv2.weight.data.clone()
            weight3  =       unet.conv3.weight.data.clone()
            weight4  =       unet.conv4.weight.data.clone()
            weight5  = unet.convbottom1.weight.data.clone()
            weight6  = unet.convbottom2.weight.data.clone()
            weight7  = unet.upsampconv1.weight.data.clone()
            weight8  =     unet.upConv0.weight.data.clone()
            weight9  =     unet.upConv1.weight.data.clone()
            weight10 = unet.upsampconv2.weight.data.clone()
            weight11 =     unet.upConv2.weight.data.clone()
            weight12 =     unet.upConv3.weight.data.clone()
            weight13 =     unet.upConv4.weight.data.clone()

        if weightMov != True:
            print("Weights are not moving!")
            sys.exit()

        # predict validate the epoch
        validationLoader = torch.utils.data.DataLoader(validationLoader, batch_size=16, shuffle=False, num_workers=8)
        for batch in validationLoader :

             tensor = BatchTensorMaker(batch, outputSize, patch_size, CUDA)
             tensor.tensorMakerTrain()

             y_pred = unet.forward(tensor.batchTensorTrainX)
             y = tensor.batchTensorTrainY
             loss = loss_fn(y_pred, y)
             validationLoss.append(float(loss.item()))

    #    with open('Unet/DataCenter/FirstEpochFileTrain', 'wb') as ap:
    #        pickle.dump(trainingLoss, ap)
    #    with open('Unet/DataCenter/FirstEpochFileVal', 'wb') as dp:
    #        pickle.dump(trainingLoss, dp)
        unet.evaluation.append(np.array(validationLoss).mean())
        with open('Unet/DataCenter/validationfile', 'wb') as fp:
            pickle.dump(unet.evaluation, fp)
        unet.trainingEvaluation.append(np.array(trainingLoss).mean())
        with open('Unet/DataCenter/trainingfile', 'wb') as tp:
            pickle.dump(unet.trainingEvaluation, tp)
        unet.EvaluateNextEpoch(unet)
        end = time.time()
        print("Epoch Elapsed in: " + str((end - start)/60) + " minutes")


#endregion
#region predict test dataset
if test:
    unet.load_state_dict(torch.load(os.path.join("Unet/DataCenter/BestState", "bestState.pth")))
    testsetOrganized = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    testsetLoss = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    perfList = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    lossResult = []
    testset = os.listdir("Unet/DataCenter/testX")
    for test in testset:
        testsetOrganized[int(filter(str.isdigit, test[0:19]))].append(test)

    for test in testsetOrganized:
        testLoader = torch.utils.data.DataLoader(test, batch_size=16, shuffle=False, num_workers=8)
        for batch in testLoader:
            tensor = BatchTensorMaker(batch, outputSize, patch_size, CUDA)
            tensor.tensorMakerTest()

            y_pred = unet.forward(tensor.batchTensorTestX)
            y = tensor.batchTensorTestY
            loss = loss_fn(y_pred, y)
            testsetLoss[int(filter(str.isdigit, batch[0][0:19]))].append(float(loss.item()))

            counter = 0
            for file in batch:
                img = nib.Nifti1Image(y_pred[counter,0, : , : , :].detach(), np.eye(4), nib.Nifti1Header())
                nib.save(img, os.path.join("Unet/DataCenter/TestResult", file))
                counter = counter + 1

            y_pred[y_pred < 0.5] = 0
            y_pred[y_pred >= 0.5] = 1
            y[y!=1] = 0

            perf = perf_measure(y, y_pred)

            #(TP, FP, TN, FN)
            a = int(filter(str.isdigit, batch[0][0:19]))
            perfList[a][0] = perfList[a][0] + perf[0]
            perfList[a][1] = perfList[a][1] + perf[1]
            perfList[a][2] = perfList[a][2] + perf[2]
            perfList[a][3] = perfList[a][3] + perf[3]


    print(perfList)
    for test in testsetLoss:
        lossResult.append(np.array(test).mean())
    #heights = [10, 20, 15]
#    bars = ['A_long', 'B_long', 'C_long']
    y_pos = range(len(lossResult))
    fig, ax = plt.subplots()
    ax.bar(y_pos, lossResult)
    ax.set_xlabel("Test Sample")
    ax.set_ylabel("Average Loss")
    ax.set_title("Test Result")

    for a,b in zip(y_pos, lossResult):
        plt.text(a, b, str(b)[0:5])
    # Rotation of the bars names
    plt.show()


#endregion
#TODO TBA
