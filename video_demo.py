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
from time import sleep
import imutils
#python video_demo.py --video 1

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

net = ResNet50()
checkpoint = torch.load(os.path.join('model/Resnet50/1', '0baby_7_lu.t7'))

net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()


# output_movie = cv2.VideoWriter(input_filepath.replace("mp4", "avi").replace("input", "output"),
#                                cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 25, (1280, 720))
# sleep(0.5)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def fer(path, save_path):
    pass

def video():
    input_filepath = '/home/wan/lusihua/BabyEmotion/datasets/月子中心.mp4'
    # input_movie = cv2.VideoCapture(0)
    timeF = 1
    frame_number = 0
    start_time = time.time()
    x = 1 # displays the frame rate every 1 second
    counter = 0

    if opt.video == 0 or opt.video == 1:
        input_movie = cv2.VideoCapture(opt.video)
        print('press Q to quit')

    else:
        # input_filepath = opt.video
        if not os.path.exists(os.path.dirname(input_filepath)):
            print('video dont exist')
        else:
            input_movie = cv2.VideoCapture(input_filepath)


    width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(width, height)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    # fourcc = int(input_movie.get(cv2.CAP_PROP_FOURCC))
    fps = input_movie.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    size = (int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 设置output
    output_movie = cv2.VideoWriter('MySaveVideo_yuezi.avi', fourcc, fps, (width, height))
    BOX_COLOR = (0, 255, 0)
    last_predicted_time = 0
    last_predicted_confidence = 0
    last_predicted_emotion = ""
    time_to_wait_between_predictions = 0.5
    while True:
        frame_number += 1
        # if not input_movie.isOpened():
        #     print('Unable to load camera')
        #     pass
        # ret, frame = video_capture.read()
        ret, frame = input_movie.read()
        image = frame
        # frame = cv2.resize(frame, (320, 180))
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        BGR_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('gray',gray)

        # if opt.align == 'mtcnn':
        #     faces = refine_mt.detector.detect_face(image)
        #     face_align = refine_mt.refine_face(image)
        # elif opt.align == 'dlib':
        #     image = imutils.resize(image, width=600)
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     face_align = face_aligner.face_align(image)
        #     faces = face_aligner.detector(gray, 2)
        #faces = refine_mt.detector.detect_face(image)
        # if faces is None:
        #     continue
        draw = image.copy()
        # print(faces)

        faces = refine_mt.detector.detect_face(image)

        face_align = refine_mt.face_alin(draw,faces)

        draw = face_align[0]
        face_align_img = face_align[1]
        #print(face_align)
        if (frame_number % timeF == 0):  # 每隔timeF帧进行存储操作
            if faces:
                # print(faces)
                for b in faces[0]:
                    x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    if w < 30 and h <30:
                        continue
                    # print(x,y,w,h)
                    #cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), BOX_COLOR , 2)
                    cropped = BGR_frame[y:y+h, x:x+w]
                    if len(cropped):
                        # if time.time() - last_predicted_time < time_to_wait_between_predictions:
                        #     expression = last_predicted_emotion
                        #     confidence = last_predicted_confidence
                        # resize_img = cv2.resize(cropped, (100, 100), interpolation=cv2.INTER_AREA)

                        # else:
                        if len(face_align_img):
                            face_align_ = face_align_img[0]
                            gray = rgb2gray(face_align_)
                            cv2.imshow("face_align", face_align_)
                        else:
                            gray = rgb2gray(cropped)

                        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
                        img = gray[:, :, np.newaxis]
                        img = np.concatenate((img, img, img), axis=2)
                        img = Image.fromarray(img)
                        inputs = transform_test(img)
                        #anger/disgust/fear/happy/sadness/surprise/contempt
                        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Normal']
                        # net = VGG('VGG19')
                        ncrops, c, h, w = np.shape(inputs)
                        inputs = inputs.view(-1, c, h, w)
                        inputs = inputs.cuda()

                        inputs = Variable(inputs)
                        # print(inputs.shape)
                        outputs = net(inputs)
                        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
                        score = F.softmax(outputs_avg, dim=0)
                        _, predicted = torch.max(outputs_avg.data, 0)
                        expression = str(class_names[int(predicted.cpu().numpy())])
                        print(expression)
                        confidence = score.data.cpu().numpy()[int(predicted.cpu().numpy())]
                        confidence = "{0:.1f}".format(confidence * 100)

                        last_predicted_emotion = expression
                        last_predicted_confidence = confidence
                        last_predicted_time = time.time()
                        # print(text)

                        if expression in ['Happy', 'Normal']:
                            color = (0, 255, 0)  #Green
                        elif expression in ['Sad','Fear']:
                            color = (255, 0, 0)  #Blue
                        else:
                            color =(0, 0, 255)  #Red

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(draw, expression, (x + w , y+20), font, 0.6, color, 2, cv2.LINE_AA )
                        #for i in range(len(class_names)):
                        #    plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
                        cv2.putText(draw, confidence, (x + w + 80, y+20), font, 0.6, color, 2, cv2.LINE_AA)

                        #for b in faces[0]:
                        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                # for p in faces[1]:
                #     for i in range(5):
                #         cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

            cv2.imshow('detection', draw)
            output_movie.write(draw)	#保存帧
            print("Writing frame {} / {}".format(frame_number, length))
        # print(draw.shape)
        # output_movie.write(draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1
        if (time.time() - start_time) > x:
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
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