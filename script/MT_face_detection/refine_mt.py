# coding: utf-8
import mxnet as mx
import sys
import cv2
import os
import time
import numpy as np
from mtcnn_detector import MtcnnDetector

current_path = os.path.abspath(os.path.dirname(__file__))
detector = MtcnnDetector(model_folder= current_path + '/model', ctx=mx.gpu(0), num_worker = 4 , accurate_landmark = False)

# img = cv2.imread('../data/image/010.jpg')
# img = cv2.imread('test/test1.jpg')
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
# run detector
def refine_face(img):

    results = detector.detect_face(img)

    if results is not None:

        total_boxes = results[0]
        points = results[1]
        # print(points,points.shape)
        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 144, 0.37)
        # for i, chip in enumerate(chips):
        #     cv2.imshow('chip_'+str(i), chip)
        #     cv2.imwrite('result/chip_'+str(i)+'.png', chip)
        print(np.shape(chips))
        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

        # cv2.imshow("detection result", draw)
        # cv2.waitKey(0)
        # if chips:
        return chips
        # else:
        #     return img
    else:
        return None
# --------------
# test on camera
# --------------
#
# camera = cv2.VideoCapture(1)
# while True:
#     grab, img = camera.read()
#     # img = cv2.resize(frame, (320,180))
#
#     t1 = time.time()
#     results = detector.detect_face(img)
#     print(results)
#     print ('time: ',time.time() - t1)
#
#     if results is None:
#         continue
#
#     total_boxes = results[0]
#     points = results[1]
#
#     draw = img.copy()
#     for b in total_boxes:
#         cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
#
#     for p in points:
#         for i in range(5):
#             cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
#     cv2.imshow("detection result", draw)
#     cv2.waitKey(30)

