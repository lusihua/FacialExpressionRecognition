from __future__ import print_function
import io
import os
import sys
import random
import cv2
# import pandas as pd
from PIL import Image
import csv
current_path = os.getcwd()

root_path = '/media/wan/36B0ED8054668DCA/dataset/dataset/affectfer'

base = root_path + '/Automatically_Annotated_Images'
done = root_path + '/Automatically_train_croped/'
csv_file = root_path + '/Automatically_annotated_file_list/automatically_annotated.csv'
new_val_txt = open(root_path + '/Automatically_annotated_file_list/train.txt','w')

fname = []
face_x = []
face_y = []
face_width = []
face_height = []
expression = []
num = 0
with open(csv_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    print(reader)
    for row in reader:
        num += 1
        fname = row['subDirectory_filePath']
        print(fname)
        x = int(row['face_x'][0:])
        y = int(row['face_y'][0:])
        width = int(row['face_width'][0:])
        height = int(row['face_height'][0:])
        expression = int(row['expression'][0:])
        floder_dir = fname.split('/')[0]
        img = fname.split('/')[1]
        image = cv2.imread(os.path.join(base, fname))
        print(os.path.join(base, fname))
        # write name & expression to new txt
        if expression < 7:
            new_val_txt.write(fname)
            new_val_txt.write(' ')
            new_val_txt.write(str(expression))
            new_val_txt.write('\n')

        # process img
        try:
            imgROI = image[x:x + width, y:y + height]
        except:
            pass
        #reshape and gray
        imgROI = cv2.resize(imgROI, (112, 112), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
        if not os.path.isdir(root_path + '/Automatically_train_croped/' + floder_dir):
            os.mkdir(root_path + '/Automatically_train_croped/' + floder_dir)
        cv2.imwrite(done + floder_dir + '/' + img, gray)
        exp = str(expression)
        if os.path.exists(root_path  + '/datasets/' + exp):
            pass
        else:
            os.mkdir(root_path  + '/datasets/' + exp)
        file_path = os.path.join(root_path + '/datasets/', str(expression))
        #print(file_path + img)
        cv2.imwrite(file_path + '/' + img, gray)
        print(fname)
        cv2.waitKey(0)

    print(num)