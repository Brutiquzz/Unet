import torch
import torch.utils.data
import os
from nifti_data_slicer import Nifti_Slicer
import nibabel as nib
import nibabel.viewers as vm
import numpy as np
import csv
from Unet import Unet
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
from BatchTensorMaker import BatchTensorMaker
from PIL import Image

'''
loss_fn = torch.nn.BCEWithLogitsLoss(reduction=None)
input2 =  torch.randint(low=0, high=2, size = (32, 2, 60, 60, 60) , requires_grad=True)
target2 = torch.randint(low=0, high=2, size = (32, 2, 60, 60, 60),  requires_grad=False )

print(torch.equal(torch.sigmoid(input2),        torch.sigmoid(input2)) == True)
#loss = loss_fn(input2, target2)
#print(loss)
#print(input2[:,:, 5:23, 5:23, 5:23].shape)
#result = torch.cat((target2, input2[:,:, 17:45, 17:45, 17:45]), 1)
#print(result.shape)
#unet = Unet()

'''
'''list = []
testset = os.listdir("DataCenter/OriginalData/test")
for test in testset:
    if "Y" in test:
        pathy = os.path.join("DataCenter/OriginalData/test", test)
        niiy = nib.load(pathy)
        datay = np.asarray(niiy.dataobj)
        percent = (datay.sum()/datay.size) * 100
        print(percent)
        list.append(percent)
print(np.mean(list))
'''

'''
testset = os.listdir("DataCenter/testY")
for test in testset:
    pathy = os.path.join("DataCenter/testY", test)
    niiy = nib.load(pathy)
    datay = np.asarray(niiy.dataobj)
    shape = datay.shape
    if shape[0] != 64 or shape[1] != 64 or shape[2] != 64:
        print("Error!")
        sys.exit()
        '''
'''
testsetOrganized = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
testset = os.listdir("Unet/DataCenter/testX")
for test in testset:
    testsetOrganized[int(filter(str.isdigit, test[0:19]))].append(test)

for test in testsetOrganized:
    print("heyz")
'''
'''
heights = [10, 20, 15]
bars = ['A_long', 'B_long', 'C_long']
y_pos = range(len(bars))
fig, ax = plt.subplots()
ax.bar(y_pos, heights)
ax.set_xlabel("Test Sample")
ax.set_ylabel("Average Loss")
ax.set_title("Test Result")

for a,b in zip(y_pos, heights):
    plt.text(a, b, str(b))
# Rotation of the bars names
plt.show()
'''
'''
slicer = Nifti_Slicer(0, 1)
slicer.patchAssembler("test", "ct_scan_testX_", "testX", slicer.no_testset, 64)

pathX = os.path.join("Unet/DataCenter/TestResult", "test_0.nii.gz")

niiX = nib.load(pathX)

dataX = np.asarray(niiX.dataobj)

pathY = os.path.join("Unet/DataCenter/OriginalData/test", "ct_scan_testX_0.nii.gz")

niiY = nib.load(pathY)

dataY = np.asarray(niiY.dataobj)
#dataX[dataX > 0] = 1000
vm.OrthoSlicer3D(dataX).show()
vm.OrthoSlicer3D(dataY).show()
print(np.array_equal(dataX, dataY))
#print(np.unique(dataX))
'''


slicer = Nifti_Slicer(0, 16)
#slicer.renameData()
#sys.exit()
slicer.patchAssembler("test", "ct_scan_testY_", "testY", slicer.no_testset, 64)

sys.exit()
pathX = os.path.join("Unet/DataCenter/TestResult", "test_4.nii.gz")
#pathX = os.path.join("Unet/DataCenter/OriginalData/test", "ct_scan_testY_4.nii.gz")

niiX = nib.load(pathX)

dataX = np.asarray(niiX.dataobj)
dataX[dataX < 0.5] = 0
dataX[dataX >= 0.5] = 1
#dataX[dataX != 1] = 0

print(np.unique(dataX))
print(np.max(dataX))
#vm.OrthoSlicer3D(dataX).show()
img = nib.Nifti1Image(dataX, np.eye(4), nib.Nifti1Header())
nib.save(img, os.path.join("Unet/DataCenter/TestResult", "hallo5.nii.gz"))

#unfold_shape = 0
#patches = 0
#data = 0
def patchFunction(trainOrTest, fileNameWithoutExt, folder, no_iter, patch_size, stride_size):
    #global patches
    #global unfold_shape
    #global data
    for x in range(0,no_iter,1):
        path = os.path.join("Unet/DataCenter/OriginalData/" + trainOrTest, fileNameWithoutExt + str(x) + ".nii.gz")

        nii = nib.load(path)

        data = torch.from_numpy(np.asarray(nii.dataobj)).unsqueeze(0)

        kc, kh, kw = patch_size, patch_size, patch_size  # kernel size
        dc, dh, dw = stride_size, stride_size, stride_size  # stride
        patches = data.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
        #torch.save(self.state_dict(), os.path.join("Unet/DataCenter/BestState", "bestState.pth"))

        for patch in range(0, len(patches[0]), 1):
            clipped_img = nib.Nifti1Image(np.asarray(patches[0][patch]), affine=nii.affine) # affine=np.eye(4)
            savePath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(patch) + '.nii')
            nib.save(clipped_img, savePath)


def patchAssembler(trainOrTest, fileNameWithoutExt, folder, no_iter, output_size):
    for x in range(0,no_iter,1):
        path = os.path.join("Unet/DataCenter/OriginalData/" + trainOrTest, fileNameWithoutExt + str(x) + ".nii.gz")

        nii = nib.load(path)

        dim = torch.from_numpy(np.asarray(nii.dataobj)).shape

        # make the patches tensor
        #Make the correct ordered list of files
        testset = os.listdir("Unet/DataCenter/TestResult")
        testset = sorted(testset)
        b = [None] * len(testset)
        for x in range(0, len(testset), 1):
            b[int(filter(str.isdigit, testset[x][21:]))] = testset[x]

        #make the patches tensor from ordered list
        patches = torch.empty(1,len(b), output_size, output_size, output_size)
        for s in range(0,len(b),1):
            path = os.path.join("Unet/DataCenter/TestResult", b[s])

            nii = nib.load(path)

            data = torch.from_numpy(np.asarray(nii.dataobj))

            patches[0][s] = data

        # figure out the unfold_shape
        x = torch.randn(1, dim[0], dim[1], dim[2])  # batch, c, h, w
        kc, kh, kw = output_size, output_size, output_size  # kernel size
        dc, dh, dw = output_size, output_size, output_size  # stride
        patches2 = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = patches2.size()

        patches_orig = patches.view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        patches_orig = patches_orig.view(1, output_c, output_h, output_w)

        # show picture
        print(dim)
        print(patches_orig.shape)
        vm.OrthoSlicer3D(patches_orig[0]).show()



# find the suitable stride
patch_size = 64
output_size = 24
stride = (patch_size - output_size) / 2
#patchFunction("test", "ct_scan_testX_", "testX", 1, 64, stride)
#patchAssembler("test", "ct_scan_testX_", "testX", 1, 24)
'''
#som test
#Forward all patches without sigmoid
trainset = os.listdir("Unet/DataCenter/testX")
trainset = sorted(trainset)
a = [None] * len(trainset)
for x in range(0, len(trainset), 1):
    a[int(filter(str.isdigit, trainset[x][21:]))] = trainset[x]

unet = Unet()
for file in a:
    path = os.path.join("Unet/DataCenter/testX", file)

    nii = nib.load(path)

    data = torch.from_numpy(np.asarray(nii.dataobj)).unsqueeze(0).unsqueeze(0)
    forwarded = unet.forward(data)
    clipped_img = nib.Nifti1Image(forwarded[0].detach(),  np.eye(4), nib.Nifti1Header()) # affine=np.eye(4)
    savePath = os.path.join("Unet/DataCenter/TestResult" , file)
    nib.save(clipped_img, savePath)
'''









sys.exit()
x = torch.randn(1, 407, 301, 360)  # batch, c, h, w
kc, kh, kw = 64, 64, 64  # kernel size
dc, dh, dw = 20, 20, 20  # stride
patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
unfold_shape = patches.size()
patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
print(patches.shape)

sys.exit()

# Reshape back
patches_orig = patches.view(unfold_shape)
output_c = unfold_shape[1] * unfold_shape[4]
output_h = unfold_shape[2] * unfold_shape[5]
output_w = unfold_shape[3] * unfold_shape[6]
patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
patches_orig = patches_orig.view(1, output_c, output_h, output_w)

# Check for equality
print((patches_orig == x[:, :output_c, :output_h, :output_w]).all())
