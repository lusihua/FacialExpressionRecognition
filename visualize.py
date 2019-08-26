"""
visualize results for test image
"""
import sys
sys.path.append('./script/MT_face_detection')

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import cv2
import transforms as transforms
from skimage import io
from skimage.transform import resize
from nets import *
from script import face_aligner
from pathlib import Path
import refine_mt
import argparse

parser = argparse.ArgumentParser(description='visualize_demo')
parser.add_argument('--align', '-a', default= False , help='align and detecter face')
opt = parser.parse_args()

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(current_path)


cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

net = ResNet18()
checkpoint = torch.load(os.path.join('model/Resnet18/5', 'afe_an_ma+fer2013_3.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def fer(path, save_path):
        image = cv2.imread(path)
        # raw_img = face_aligner.face_align(path)
        if opt.align == 'mtcnn':
            raw_img = refine_mt.refine_face(path)
            if raw_img:
                image = raw_img[0]
                # cv2.imshow("raw_img", image)
                # cv2.waitKey(0)
            else:
                image = cv2.imread(image)
        elif opt.align == 'dlib':
            raw_img = face_aligner.face_align(image)
            image = raw_img

        gray = rgb2gray(image)

        gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

        # print(gray.shape)

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)

        inputs = transform_test(img)
        #
        # print(inputs.shape)
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Normal']

        # net = VGG('VGG19')


        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()

        start = time.time()
        inputs = Variable(inputs)
        #
        # print(inputs.shape)

        outputs = net(inputs)

        end = time.time()

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg, dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)

        plt.rcParams['figure.figsize'] = (13.5,5.5)
        axes=plt.subplot(1, 3, 1)

        #BGR2RGB
        raw_img = image[...,::-1]
        plt.imshow(raw_img)
        plt.xlabel('Input Image', fontsize=16)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.tight_layout()

        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

        plt.subplot(1, 3, 2)
        ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
        width = 0.4       # the width of the bars: can also be len(x) sequence
        color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
        for i in range(len(class_names)):
            plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
        plt.title("Classification results ",fontsize=20)
        plt.xlabel(" Expression Category ",fontsize=16)
        plt.ylabel(" Classification Score ",fontsize=16)
        plt.xticks(ind, class_names, rotation=45, fontsize=14)

        axes=plt.subplot(1, 3, 3)
        emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
        print('class_names:', class_names[int(predicted.cpu().numpy())])
        plt.imshow(emojis_img)
        plt.xlabel('Emoji Expression', fontsize=16)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.tight_layout()
        # show emojis

        #plt.show()
        plt.savefig(os.path.join(save_path))
        plt.close()

        print("inference time:%.3f seconds"% (end - start))
        print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))

if __name__ == '__main__':

    input_img = parent_path +'/' + 'FER/1.png'
    output_img = current_path + '/' + 'images/results'
    input_folder = 'images/image/google_babycry/Sad1'
    print(input_folder)
    output_folder = 'images/results'
    if  os.path.exists(input_folder):
        for p in Path(input_folder).glob('*'):
            path = str(p)
            name = os.path.basename(path)
            name = '.'.join(name.split('.')[:-1]) + '.png'
            output_path = os.path.join(output_folder, name)
            print(path, output_path)
            fer(path, output_path)
            print(path + ' -> ' + output_path)
        print('Done')
    elif input_img:
        fer(input_img, output_img)
