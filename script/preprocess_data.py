# create data and label for CK+
#  0=anger 1=disgust, 2=fear, 3=happy, 4=sadness, 5=surprise, 6=contempt
# contain 135,177,75,207,84,249,54 images

import csv
import os
import numpy as np
import h5py
import skimage.io

data_path = '/media/wan/36B0ED8054668DCA/dataset/train'

# anger_path = os.path.join(data_path, 'anger')
# disgust_path = os.path.join(data_path, 'disgust')
# fear_path = os.path.join(data_path, 'fear')
happy_path = os.path.join(data_path, 'happy')
sadness_path = os.path.join(data_path, 'sadness')
# surprise_path = os.path.join(data_path, 'surprise')
# contempt_path = os.path.join(data_path, 'contempt')
normal_path = os.path.join(data_path, 'normal')
# # Creat the list to store the data and label information
data_x = []
data_y = []

datapath = os.path.join('/media/wan/36B0ED8054668DCA/dataset','data48_CropSad1_CropHappy1+fer2013_sad.h5')
print(datapath)

if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# order the file, so the training set will not contain the test set (don't random)

files = os.listdir(sadness_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(sadness_path,filename))
    data_x.append(I.tolist())
    data_y.append(4)

print(np.shape(data_x))
print(np.shape(data_y))

files = os.listdir(happy_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(happy_path,filename))
    data_x.append(I.tolist())
    data_y.append(3)

print(np.shape(data_x))
print(len(data_y))

files = os.listdir(normal_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(normal_path,filename))
    data_x.append(I.tolist())
    data_y.append(6)

print(len(data_x))
print(len(data_y))
# files = os.listdir(sadness_path)
# files.sort()
# for filename in files:
#     I = skimage.io.imread(os.path.join(sadness_path,filename))
#     data_x.append(I.tolist())
#     data_y.append(4)
#
# print(np.shape(data_x))
# print(np.shape(data_y))

# files = os.listdir(surprise_path)
# files.sort()
# for filename in files:
#     I = skimage.io.imread(os.path.join(surprise_path,filename))
#     data_x.append(I.tolist())
#     data_y.append(5)
# print(np.shape(data_x))
# print(np.shape(data_y))


# files = os.listdir(anger_path)
# files.sort()
# for filename in files:
#     I = skimage.io.imread(os.path.join(anger_path,filename))
#     data_x.append(I.tolist())
#     data_y.append(0)
#
# files = os.listdir(disgust_path)
# files.sort()
# for filename in files:
#     I = skimage.io.imread(os.path.join(disgust_path,filename))
#     data_x.append(I.tolist())
#     data_y.append(1)
#
# files = os.listdir(fear_path)
# files.sort()
# for filename in files:
#     I = skimage.io.imread(os.path.join(fear_path,filename))
#     data_x.append(I.tolist())
#     data_y.append(2)
datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype = 'uint8', data=data_x)
datafile.create_dataset("data_label", dtype = 'int64', data=data_y)
datafile.close()

print("Save data finish!!!")
