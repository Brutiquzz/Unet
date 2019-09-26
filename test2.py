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
import torch.nn as nn
import pickle

'''
pathX = os.path.join("Unet/DataCenter/OriginalData/test", "ct_scan_testX_0.nii.gz")

niiX = nib.load(pathX)

dataX = np.asarray(niiX.dataobj)
#dataX[dataX > 0] = 1000

#print(np.unique(dataX))

vm.OrthoSlicer3D(dataX).show()
'''
'''
x = torch.randn(1, 500, 500, 500)  # batch, c, h, w
kc, kh, kw = 64, 64, 64  # kernel size
dc, dh, dw = 64, 64, 64  # stride
patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
print(patches.shape)

fold = nn.Fold(output_size=(448, 448, 448), kernel_size=(64, 64, 64))

output = fold(patches)
print(output.shape)
'''

a = np.ones((5,5,5))
a[:,3,:] = 0
a[:,:,3] = 0

'''
#a[2] = 0
a[0][3] = 0
a[1][3] = 0
a[2][3] = 0
a[3][3] = 0
a[4][3] = 0

a[0][0][3] = 0
a[0][1][3] = 0
a[0][2][3] = 0
a[0][3][3] = 0
a[0][4][3] = 0
a[1][0][3] = 0
a[1][1][3] = 0
a[1][2][3] = 0
a[1][3][3] = 0
a[1][4][3] = 0
a[2][0][3] = 0
a[2][1][3] = 0
a[2][2][3] = 0
a[2][3][3] = 0
a[2][4][3] = 0
a[3][0][3] = 0
a[3][1][3] = 0
a[3][2][3] = 0
a[3][3][3] = 0
a[3][4][3] = 0
a[4][0][3] = 0
a[4][1][3] = 0
a[4][2][3] = 0
a[4][3][3] = 0
a[4][4][3] = 0
a[4][4][2] = 0
'''
#a = a[~np.all(a == 0, axis=1)]
#a = np.delete(a,np.where(~a.any(axis=0))[0], axis=1)
#a = a[~np.all(a == 0, axis=1)]
'''
testset = os.listdir("DataCenter/OriginalData/test")
for test in testset:
    if "Y" in test:
        pathy = os.path.join("DataCenter/OriginalData/test", test)
        niiy = nib.load(pathy)
        datay = np.asarray(niiy.dataobj)
        print(datay.shape)
        '''
itemlist = [1.0131,2.234234,3.23423,4.23423,5.4565,6.546456]
with open('Unet/DataCenter/outfile', 'wb') as fp:
    pickle.dump(itemlist, fp)

itemlist2 = []
with open ('Unet/DataCenter/outfile', 'rb') as tp:
    itemlist2 = pickle.load(tp)

print(itemlist2)
