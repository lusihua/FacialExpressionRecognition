from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import sys
sys.path.append('/home/lusihua/BabyExpression/BabyEmotion')
sys.path.append('./MT_face_detection')
import os
from pathlib import Path
import refine_mt
import shutil
# temp

def _help():
    print("Usage:")
    print("     python face_align.py <path of a picture>")
    print("For example:")
    print("     python face_align.py pic/HL.jpg")


# def face_align(img_path):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")
#
#     # 初始化 FaceAligner 类对象
#     fa = FaceAligner(predictor, desiredFaceWidth=256)
#
#     image = cv2.imread(img_path)
#
#     image = imutils.resize(image, width=600)
#     face_aligned = image
#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("gray", gray)
#     # cv2.imshow("Input", image)
#     rects = detector(gray, 2)
#
#     # 返回检测的人脸
#     print(rects)
#
#     # face_aligner
#     if not rects:
#         return ''
#
#     for rect in rects:
#         # (x, y, w, h) = rect_to_bb(rect)
#         # face_orig = imutils.resize(image[y:y + h, x:x + w], width=256)
#
#         # 调用 align 函数对图像中的指定人脸进行处理
#         face_aligned = fa.align(image, gray, rect)
#         # cv2.imshow("Original", face_orig)
#         # cv2.imshow("Aligned", face_aligned)
#         # cv2.waitKey(0)
#         #print(face_aligned)
#         return face_aligned
def face_align(path1):
    raw_img = refine_mt.refine_face(path1)
    if raw_img:
        image = raw_img[0]

        # cv2.imshow("raw_img", image)
        # cv2.waitKey(0)
        return image
    else:
        return path1

if __name__ == '__main__':
    img_path = '/media/wan/36B0ED8054668DCA/dataset/dataset/baby_smile/baby smile/56150_cdc24d5ea2_z.jpg'
    img_folder = '/media/wan/36B0ED8054668DCA/My_data_web/images-babycry-b2018_pre/pre_folder/Sad'
    save_path = '/media/wan/36B0ED8054668DCA/My_data_web/images-babycry-b2018_pre/pre_folder/crop_sad'

    if img_folder:
        print(img_folder)
        for p in Path(img_folder).glob('*'):

            path = str(p)
            # print(path)
            name = os.path.basename(path)
            name = '.'.join(name.split('.')[:-1]) + '.png'
            # print(path, name)
            face_aligned = face_align(path)
            if len(face_aligned) > 0:
            # save_path = save_path + '/crop/' + name
                print(save_path + '/crop/' + name)
                cv2.imwrite(save_path + '/crop/' + name, face_aligned)

            else:
                shutil.copy(path, save_path + '/origin/' + name)


    # face_aligned = face_align(img_path)
    # cv2.imshow('face_align', face_aligned)
    # cv2.waitKey(0)