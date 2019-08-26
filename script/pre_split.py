from __future__ import print_function
import os
import sys
sys.path.append('/home/lusihua/BabyExpression/BabyEmotion')
sys.path.append('./MT_face_detection')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np

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
import shutil
import refine_mt

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(current_path)

parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

cut_size = 44
path = os.path.join('./model/' + opt.model, str(opt.fold))

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

net = ResNet50()
# checkpoint = torch.load(os.path.join('model/VGG19/1', 'afe+fer2013_3.t7'))

checkpoint = torch.load(
    os.path.join('/home/wan/lusihua/BabyEmotion/model/Resnet50/5', 'afe_an_ma+fer2013_3.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def test(path_img):

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Normal']

    start = time.time()


    raw_img = refine_mt.refine_face(path_img)
    if raw_img:
        image = raw_img[0]
        # cv2.imshow("raw_img", image)
        # cv2.waitKey(0)
    else:
        return None

    # if opt.align == 'mtcnn':
    #     raw_img = refine_mt.refine_face(path)
    #     if raw_img:
    #         image = raw_img[0]
    #         # cv2.imshow("raw_img", image)
    #         # cv2.waitKey(0)
    #     else:
    #         image = cv2.imread(image)
    # elif opt.align == 'dlib':
    #     raw_img = face_aligner.face_align(image)
    #     image = raw_img


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
    input_folder = '/media/wan/36B0ED8054668DCA/My_data_web/images-babycry-b2018/images'
    sub_folders = os.listdir(input_folder)
    out_folder = '/media/wan/36B0ED8054668DCA/My_data_web/images-babycry-b2018_pre'

    if input_folder:
        for sub_folder in sub_folders:
            sub_folder = input_folder + '/' + sub_folder
            # print(sub_folder)
            for p in Path(sub_folder).glob('*'):
                # target = sub_folder.split('/')[-1]
                path = str(p)
                print(path)
                pre_classes_names = test(path)
                if not pre_classes_names:
                    shutil.copy(path, out_folder + '/' + 'origin_img')
                    print(out_folder + '/' + 'origin_img')
                else:
                    exp = str(pre_classes_names)
                    name = os.path.basename(path)
                    name = '.'.join(name.split('.')[:-1]) + '.png'
                    # print(path, output_path)
                    if os.path.exists(out_folder + '/pre_folder/' + exp):
                        pass
                    else:
                        os.mkdir(out_folder + '/pre_folder/' + exp)
                    file_path = os.path.join(out_folder + '/pre_folder/', exp)
                    print(file_path)
                    shutil.copy(path, file_path)
