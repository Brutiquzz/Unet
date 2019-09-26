
import torch
import nibabel as nib
import numpy as np
import os
import math
import csv
import sys
from Unet import Unet

# This is not complete

class Nifti_Slicer:

    def __init__(self, no_trainset, no_testset):
        self.no_trainset = no_trainset
        self.no_testset = no_testset

    def randomPatchExtraction(self):
        datas = os.listdir("Unet/DataCenter/OriginalData/train")
        for data in datas:
            if "X" in data:
                pathx = os.path.join("Unet/DataCenter/OriginalData/train", data)
                pathy = os.path.join("Unet/DataCenter/OriginalData/train", data.replace("X","Y"))

                # Read the .nii image containing the volume with nibabel:
                niix = nib.load(pathx)
                niiy = nib.load(pathy)

                # Get data as numpy
                datax = np.asarray(niix.dataobj)
                datay = np.asarray(niiy.dataobj)

                for patchNo in range(0,300,1):
                    patchNo = 0


    def patchAssembler(self, trainOrTest, fileNameWithoutExt, folder, no_iter, patch_size):

        if patch_size % 2 != 0:
            print("Patch size must be divisible by 2")
            sys.exit()
        unet = Unet()
        outputSize = len((unet.forward((torch.randint(low=0, high=2, size = (1, 1, patch_size, patch_size, patch_size) )).to(torch.float32)))[0][0][0])
        step_size = (patch_size - outputSize)/2
        for x in range(0,no_iter,1):
            path = os.path.join("Unet/DataCenter/OriginalData/" + trainOrTest, fileNameWithoutExt + str(x) + ".nii.gz")

            # Read the .nii image containing the volume with nibabel:
            nii = nib.load(path)
            affine = nii.affine

            # Get data as numpy
            data = np.asarray(nii.dataobj)

            # split data into chunks
            # This will create new nifti files in the dimensions 64x64x64
            part        = 0
            data_width  = data.shape[0] # -1
            data_height = data.shape[1] # -1
            data_depth  = data.shape[2] # -1

            data = torch.zeros(data_width, data_height, data_depth)

            for width in range(0, data_width, step_size):
                for height in range(0, data_height, step_size):
                    for depth in range(0, data_depth, step_size):
                        # check for last iteration to perform overlapping
                        if width+patch_size > data_width and height+patch_size > data_height and depth+patch_size > data_depth:
                            loadPath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            patch = torch.from_numpy(np.asarray((nib.load(loadPath)).dataobj))
                            data[data_width-outputSize:, data_height-outputSize:, data_depth-outputSize:] = patch
                            part = part + 1
                        elif width+patch_size > data_width and depth+patch_size > data_depth:
                            loadPath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            patch = torch.from_numpy(np.asarray((nib.load(loadPath)).dataobj))
                            data[data_width-outputSize:, height:height+outputSize, data_depth-outputSize:] = patch
                            part = part + 1
                        elif width+patch_size > data_width and height+patch_size > data_height:
                            loadPath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            patch = torch.from_numpy(np.asarray((nib.load(loadPath)).dataobj))
                            data[data_width-outputSize:, data_height-outputSize:, depth:depth+outputSize] = patch
                            part = part + 1
                        elif height+patch_size > data_height and depth+patch_size > data_depth:
                            loadPath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            patch = torch.from_numpy(np.asarray((nib.load(loadPath)).dataobj))
                            data[width:width+outputSize, data_height-outputSize:, data_depth-outputSize:] = patch
                            part = part + 1
                        elif width+patch_size > data_width:
                            loadPath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            patch = torch.from_numpy(np.asarray((nib.load(loadPath)).dataobj))
                            data[data_width-outputSize:, height:height+outputSize, depth:depth+outputSize] = patch
                            part = part + 1
                        elif height+patch_size > data_height :
                            loadPath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            patch = torch.from_numpy(np.asarray((nib.load(loadPath)).dataobj))
                            data[width:width+outputSize, data_height-outputSize:, depth:depth+outputSize] = patch
                            part = part + 1
                        elif depth+patch_size > data_depth :
                            loadPath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            patch = torch.from_numpy(np.asarray((nib.load(loadPath)).dataobj))
                            data[width:width+outputSize, height:height+outputSize, data_depth-outputSize:] = patch
                            part = part + 1
                        else:
                            loadPath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            patch = torch.from_numpy(np.asarray((nib.load(loadPath)).dataobj))
                            data[width:width+outputSize, height:height+outputSize, depth:depth+outputSize] = patch
                            part = part + 1


            data[:,:,data_depth-64:data_depth-40] = data[:,:,data_depth-24:]
            data[:,:,data_depth-24:] = 0
            data[:,data_height-64:data_height-40, :] = data[:,data_height-24:,:]
            data[:,data_height-24:,:] = 0
            data[data_width-64:data_width-40,:, :] = data[data_width-24:,:,:]
            data[data_width-24:,:,:] = 0
            clipped_img = nib.Nifti1Image(data,  np.eye(4), nib.Nifti1Header()) # affine=np.eye(4)
            savePath = os.path.join("Unet/DataCenter/TestResult" , "test_" + str(x) + '.nii.gz')
            nib.save(clipped_img, savePath)




    def patchFunction(self, trainOrTest, fileNameWithoutExt, folder, no_iter, patch_size):

        if patch_size % 2 != 0:
            print("Patch size must be divisible by 2")
            sys.exit()

        unet = Unet()
        outputSize = len((unet.forward((torch.randint(low=0, high=2, size = (1, 1, patch_size, patch_size, patch_size) )).to(torch.float32)))[0][0][0])
        step_size = (patch_size - outputSize)/2
        for x in range(0,no_iter,1):
            path = os.path.join("Unet/DataCenter/OriginalData/" + trainOrTest, fileNameWithoutExt + str(x) + ".nii.gz")

            # Read the .nii image containing the volume with nibabel:
            nii = nib.load(path)

            # Get data as numpy
            data = np.asarray(nii.dataobj)

            # split data into chunks
            # This will create new nifti files in the dimensions 64x64x64
            part        = 0
            data_width  = data.shape[0] # -1
            data_height = data.shape[1] # -1
            data_depth  = data.shape[2] # -1
           # width_step = int(math.ceil(data_width/64.0))
           # height_step = int(math.ceil(data_height/64.0))
           # depth_step = int(math.ceil(data_depth/64.0))
           # print({'width' : data_width, 'height' : data_height, 'depth' : data_depth, 'wStep' : width_step, 'hStep' : height_step, 'dStep' : depth_step})
            for width in range(0, data_width, step_size):
                for height in range(0, data_height, step_size):
                    for depth in range(0, data_depth, step_size):
                        # check for last iteration to perform overlapping
                        if width+patch_size > data_width and height+patch_size > data_height and depth+patch_size > data_depth:
                            imageChunk = data[data_width-patch_size:, data_height-patch_size:, data_depth-patch_size:]
                            clipped_img = nib.Nifti1Image(imageChunk, affine=nii.affine) # affine=np.eye(4)
                            savePath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            nib.save(clipped_img, savePath)
                            part = part + 1
                        elif width+patch_size > data_width and depth+patch_size > data_depth:
                            imageChunk = data[data_width-patch_size:, height:height+patch_size, data_depth-patch_size:]
                            clipped_img = nib.Nifti1Image(imageChunk, affine=nii.affine) # affine=np.eye(4)
                            savePath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            nib.save(clipped_img, savePath)
                            part = part + 1
                        elif width+patch_size > data_width and height+patch_size > data_height:
                            imageChunk = data[data_width-patch_size:, data_height-patch_size:, depth:depth+patch_size]
                            clipped_img = nib.Nifti1Image(imageChunk, affine=nii.affine) # affine=np.eye(4)
                            savePath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            nib.save(clipped_img, savePath)
                            part = part + 1
                        elif height+patch_size > data_height and depth+patch_size > data_depth:
                            imageChunk = data[width:width+patch_size, data_height-patch_size:, data_depth-patch_size:]
                            clipped_img = nib.Nifti1Image(imageChunk, affine=nii.affine) # affine=np.eye(4)
                            savePath = os.path.join("Unet/DataCenter/" + folder , fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            nib.save(clipped_img, savePath)
                            part = part + 1
                        elif width+patch_size > data_width:
                            imageChunk = data[data_width-patch_size:, height:height+patch_size, depth:depth+patch_size]
                            clipped_img = nib.Nifti1Image(imageChunk, affine=nii.affine) # affine=np.eye(4)
                            savePath = os.path.join("Unet/DataCenter/" + folder, fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            nib.save(clipped_img, savePath)
                            part = part + 1
                        elif height+patch_size > data_height :
                            imageChunk = data[width:width+patch_size, data_height-patch_size:, depth:depth+patch_size]
                            clipped_img = nib.Nifti1Image(imageChunk, affine=nii.affine) # affine=np.eye(4)
                            savePath = os.path.join("Unet/DataCenter/" + folder, fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            nib.save(clipped_img, savePath)
                            part = part + 1
                        elif depth+patch_size > data_depth :
                            imageChunk = data[width:width+patch_size, height:height+patch_size, data_depth-patch_size:]
                            clipped_img = nib.Nifti1Image(imageChunk, affine=nii.affine) # affine=np.eye(4)
                            savePath = os.path.join("Unet/DataCenter/" + folder, fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                            nib.save(clipped_img, savePath)
                            part = part + 1
                        else:
                             imageChunk = data[width:width+patch_size, height:height+patch_size, depth:depth+patch_size]
                             clipped_img = nib.Nifti1Image(imageChunk, affine=nii.affine) # affine=np.eye(4)
                             savePath = os.path.join("Unet/DataCenter/" + folder, fileNameWithoutExt + str(x) + '_part_' + str(part) + '.nii.gz')
                             nib.save(clipped_img, savePath)
                             part = part + 1



# unpack orginal data
    def renameData(self):
        counter = 0
        files = os.listdir("Unet/DataCenter/OriginalData/train")
        for file in files:
            if "reference" in file:
                #os.rename(file, "ct_scan_trainX" + "_"+ str(counter) + ".nii.gz")
                #os.rename(file.replace("lungs", "reference"), "ct_scan_trainY" + "_" + str(counter) + ".nii.gz")

                old_file = os.path.join("Unet/DataCenter/OriginalData/train", file)
                new_file = os.path.join("Unet/DataCenter/OriginalData/train",  "ct_scan_trainY" + "_"+ str(counter) + ".nii.gz")
                os.rename(old_file, new_file)

                old_file = os.path.join("Unet/DataCenter/OriginalData/train", file.replace("-reference", ""))
                new_file = os.path.join("Unet/DataCenter/OriginalData/train",  "ct_scan_trainX" + "_" + str(counter) + ".nii.gz")
                os.rename(old_file, new_file)
                counter = counter + 1

        counter = 0
        files = os.listdir("Unet/DataCenter/OriginalData/test")
        for file in files:
            if "reference" in file:
                print(file)
                old_file = os.path.join("Unet/DataCenter/OriginalData/test", file)
                new_file = os.path.join("Unet/DataCenter/OriginalData/test",  "ct_scan_testY" + "_"+ str(counter) + ".nii.gz")
                os.rename(old_file, new_file)

                old_file = os.path.join("Unet/DataCenter/OriginalData/test", file.replace("-reference", ""))
                new_file = os.path.join("Unet/DataCenter/OriginalData/test",  "ct_scan_testX" + "_" + str(counter) + ".nii.gz")
                os.rename(old_file, new_file)
                counter = counter + 1

# This function slices all the X training data into 64x64x64 chunks
    def train_data_X_factory(self, patch_size):
        self.patchFunction("train", "ct_scan_trainX_", "trainX", self.no_trainset, patch_size)

# This function slices all the Y training data into 64x64x64 chunks
    def train_data_Y_factory(self, patch_size):
        self.patchFunction("train", "ct_scan_trainY_", "trainY", self.no_trainset, patch_size)

# This functions assembles the predictes from the test result
    def test_result_assembling(self, patch_size):
        self.patchAssembler("test", "ct_scan_testX_", "TestResult", self.no_testset, patch_size)

# This function slices all the X test data into 64x64x64 chunks
    def test_data_X_factory(self, patch_size):
        self.patchFunction("test", "ct_scan_testX_", "testX", self.no_testset, patch_size)

# This function slices all the Y test data into 64x64x64 chunks
    def test_data_Y_factory(self, patch_size):
        self.patchFunction("test", "ct_scan_testY_", "testY", self.no_testset, patch_size)

# This function produces slicing of the training data
    def training_data_factory(self, patch_size):
        self.train_data_X_factory(patch_size)
        self.train_data_Y_factory(patch_size)
        #self.merge("DataCenter/trainX", "train")

# This function produces slicing of the test data
    def test_data_factory(self, patch_size):
        self.test_data_X_factory(patch_size)
        self.test_data_Y_factory(patch_size)

# This function produces slicing of the entire dataset. Both test and training
# This also produces a merged version of X and Y in a csv file
    def patch_data_factory(self, patch_size):
        self.training_data_factory(patch_size)
        self.test_data_factory(patch_size)
