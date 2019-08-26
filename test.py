from __future__ import print_function
import sys
sys.path.append('./script/MT_face_detection')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
from script import utils
from torch.autograd import Variable
from script import face_aligner
from nets import *
from skimage import io
from skimage.transform import resize
from pathlib import Path
import h5py
import skimage.io
from PIL import Image
import time
import cv2
import refine_mt
import argparse

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(current_path)

parser = argparse.ArgumentParser(description='PyTorch facial Expression Recognition CNN Test')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--align', '-a', default= 'mtcnn' , help='align and detecter face')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

cut_size = 44
path = os.path.join('./model/' + opt.model, str(opt.fold))

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

net = ResNet34()
# checkpoint = torch.load(os.path.join('model/VGG19/1', 'afe+fer2013_3.t7'))
# checkpoint = torch.load(os.path.join('model/Resnet18/5', 'afe_an_ma+fer2013_3.t7'))
checkpoint = torch.load(os.path.join('model/Resnet34/5', 'afe_an_ma+fer2013_3.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def test(image):

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Normal']

    start = time.time()
    if opt.align == 'mtcnn':
        image = cv2.imread(path)
        raw_img = refine_mt.refine_face(image)
        if raw_img:
            image = raw_img[0]
            # cv2.imshow("raw_img", image)
            # cv2.waitKey(0)
        else:
            pass
    elif opt.align == 'dlib':
        image = cv2.imread(path)
        raw_img = face_aligner.face_align(image)
        image = raw_img

    gray = rgb2gray(image)

    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    # print(gray.shape)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    # print(inputs.shape)

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs)
    # print(inputs.shape)

    outputs = net(inputs)
    end = time.time()

    print("inference time:%.3f seconds" % (end - start))
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    # score = F.softmax(outputs_avg, dim=0)
    _, predicted = torch.max(outputs_avg.data, 0)

    pre_class_names = str(class_names[int(predicted.cpu().numpy())])

    return pre_class_names

if __name__ == '__main__':

    # data_path = '/media/wan/36B0ED8054668DCA/dataset/dataset/My_data_web/baby_smile'
    # data_save_path = os.path.join(current_path, 'data_test.h5')
    # if not os.path.exists(os.path.dirname(data_save_path)):
    #     os.makedirs(os.path.dirname(data_save_path))

    correct = 0
    total = 0

    # input_folder = parent_path +'/' + 'FER/datasets/google_babysmile/Happy'
    input_folder = '/media/wan/36B0ED8054668DCA/dataset/dataset/My_data_web/baby_smile/Happy'
    input_folder = '/home/wan/lusihua/BabyEmotion/images/image/google_babycry/Sad'
    # print(input_folder)
    if input_folder:
        for p in Path(input_folder).glob('*'):
            target = input_folder.split('/')[-1]
            path = str(p)
            # print(path)
            name = os.path.basename(path)
            name = '.'.join(name.split('.')[:-1]) + '.png'
            # print(path, output_path)
            pre_class_names = test(path)
            print(pre_class_names, target)
            if target == pre_class_names:
                correct += 1
                print(correct)
            total += 1
            print(total)
        Test_acc = 100. * correct / total
        print("Test_acc: %0.3f" % Test_acc)

