from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import sys
import os
from pathlib import Path
#temp

def _help():
    print("Usage:")
    print("     python face_align.py <path of a picture>")
    print("For example:")
    print("     python face_align.py pic/HL.jpg")


def face_align(image):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

    # 初始化 FaceAligner 类对象
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # image = cv2.imread(img_path)

    image = imutils.resize(image, width=600)
    face_aligned = image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.imshow("Input", image)
    rects = detector(gray, 2)

    #返回检测的人脸
    print(rects)

    for rect in rects:
        # (x, y, w, h) = rect_to_bb(rect)
        # face_orig = imutils.resize(image[y:y + h, x:x + w], width=256)

        # 调用 align 函数对图像中的指定人脸进行处理
        face_aligned = fa.align(image, gray, rect)
        # cv2.imshow("Original", face_orig)
        # cv2.imshow("Aligned", face_aligned)
        # cv2.waitKey(0)
    return face_aligned

if __name__ == '__main__':
    img_path = '/media/wan/36B0ED8054668DCA/dataset/dataset/baby_smile/baby smile/56150_cdc24d5ea2_z.jpg'
    img_folder = '/media/wan/36B0ED8054668DCA/dataset/dataset/baby_smile/baby smile'
    save_path = '/media/wan/36B0ED8054668DCA/dataset/dataset/baby_smile'
    if img_folder:
        for p in Path(img_folder).glob('*'):
            target = img_folder.split('/')[-1]
            path = str(p)
            # print(path)
            name = os.path.basename(path)
            name = '.'.join(name.split('.')[:-1]) + '.png'
            # print(path, name)
            face_aligned = face_align(path)
            # save_path = save_path + '/crop/' + name
            print(save_path)
            cv2.imwrite(save_path + '/crop/' + name, face_aligned)



    # face_aligned = face_align(img_path)
    # cv2.imshow('face_align', face_aligned)
    # cv2.waitKey(0)