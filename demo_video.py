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
parser.add_argument('--video', '-v', default= False , help='detect video')
parser.add_argument('--images', '-images', default= False , help='detect folder')
opt = parser.parse_args()

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(current_path)


cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

net = ResNet34()
checkpoint = torch.load(os.path.join('model/Resnet34/10', 'afe_an_ma+fer2013_3.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

input_filepath = '/home/wan/lusihua/BabyEmotion/datasets/月子中心.mp4'


input_movie = cv2.VideoCapture(input_filepath)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# 设置output
output_movie = cv2.VideoWriter(input_filepath.replace("mp4", "avi").replace("input", "output"),
                               cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 25, (1280, 720))


if opt.video:
    # video_capture = cv2.VideoCapture(1)

    print('press Q to quit')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def fer(path, save_path):
    pass

def video():
    frame_number = 0
    while True:
        frame_number += 1
        if not input_movie.isOpened():
            print('Unable to load camera')
            pass
        # ret, frame = video_capture.read()
        ret, frame = input_movie.read()
        image = frame
        # frame = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        BGR_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('gray',gray)
        faces = refine_mt.detector.detect_face(image)
        # if faces is None:
        #     continue
        draw = image.copy()
        # print(faces)
        if faces:
            # print(faces)
            for b in faces[0]:
                x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                # print(x,y,w,h)
                cropped = BGR_frame[y:y+h, x:x+w]
                # resize_img = cv2.resize(cropped, (100, 100), interpolation=cv2.INTER_AREA)
                if len(cropped):
                    gray = rgb2gray(cropped)

                    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

                    img = gray[:, :, np.newaxis]

                    img = np.concatenate((img, img, img), axis=2)
                    img = Image.fromarray(img)

                    inputs = transform_test(img)
                    #
                    # print(inputs.shape)
                    #anger/disgust/fear/happy/sadness/surprise/contempt
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
                    expression = str(class_names[int(predicted.cpu().numpy())])
                    # socore_exp = np.random.randint(5, size=(2, 4))

                    # for i,j in enumerate(class_names):
                    #     if expression == j:
                    score_exp = score.data.cpu().numpy()[int(predicted.cpu().numpy())]
                    text = "{0:.1f}".format(score_exp * 100)
                    print(text)
                    if expression in ['Happy', 'Normal']:
                        color = (0, 255, 0)
                    elif expression in ['Sad']:
                        color = (255, 0, 0)
                    else:
                        color =(0, 0, 255)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(draw, expression, (x + w, y), font, 1, color, 2, cv2.LINE_AA )
                    #for i in range(len(class_names)):
                    #    plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
                    cv2.putText(draw, text, (x + w, y+25), font, 1, color, 2, cv2.LINE_AA)

                    # for b in faces[0]:
                    cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                    #
        print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(draw)
        # All done!
    input_movie.release()
    output_movie.release()
    cv2.destroyAllWindows()
        # cv2.imshow('Expression Detection', draw)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # input_movie.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    if opt.images:
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
    elif opt.video:
        video()