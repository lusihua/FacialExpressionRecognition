"author:youngkun date:20180615 function:change the size of pictures in one folder"
import cv2
import os

image_size = 48  # 设定尺寸
source_path = "/media/wan/36B0ED8054668DCA/My_data_web/baby_smile/Happy_crop/"  # 源文件路径
# target_path = "/media/wan/36B0ED8054668DCA/dataset/dataset/affectfer/datasets48an/happy/"  # 输出目标文件路径
target_path = "/media/wan/36B0ED8054668DCA/dataset/train/happy/"
others_path = "/media/wan/36B0ED8054668DCA/My_data_web/baby_smile/CK+48_CropHappy/"

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)  # 获得文件名

i = 0
for file in image_list:
    i = i + 1
    image_source = cv2.imread(source_path + file)  # 读取图片
    image_gray = cv2.cvtColor(image_source, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image_gray, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)  # 修改尺寸
    cv2.imwrite(target_path + 'sad_crap1' + str(i) + ".jpg", image)  # 重命名并且保存
    cv2.imwrite(others_path + 'sad_crap1' + str(i) + ".jpg", image)  # 重命名并且保存
print("批量处理完成")




# import cv2
# import numpy as np
#
#
# def crop_image(input_image):
#     #高水平梯度和低垂直梯度的图像区域
#     gradX = cv2.Sobel(input_image, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
#     gradY = cv2.Sobel(input_image, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)
#
#     # subtract the y-gradient from the x-gradient
#     img_gradient = cv2.subtract(gradX, gradY)
#     img_gradient = cv2.convertScaleAbs(img_gradient)
#
#     # blur and threshold the image
#     blurred = cv2.blur(img_gradient, (9, 9))
#     (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
#
#     #
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
#     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
#     # perform a series of erosions and dilations
#     closed = cv2.erode(closed, None, iterations=4)
#     closed = cv2.dilate(closed, None, iterations=4)
#
#     (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
#
#     # compute the rotated bounding box of the largest contour
#     rect = cv2.minAreaRect(c)
#     box = np.int0(cv2.cv.BoxPoints(rect))
#
#     # draw a bounding box arounded the detected barcode and display the image
#     #cv2.drawContours(input_image, [box], -1, (0, 255, 0), 3)
#
#     height,width = input_image.shape
#
#     Xs = [i[0] for i in box]
#     Ys = [i[1] for i in box]
#     x1 = min(Xs)
#     x2 = max(Xs)
#     x1_ = (x2-x1)/2-125
#     x2_ = (x2-x1)/2+125
#     if x1_< 0:
#         x1_ = 0
#     if x2_ > width:
#         x2_ = width
#     y1 = min(Ys)
#     y2 = max(Ys)
#     y1_ = (y2 - y1)/2-125
#     y2_ = (y2 - y1)/2+125
#     if y1_ < 0:
#         y1_ = 0
#     if y2_ > height:
#         y2_ = height
#     #cv2.rectangle(input_image,(x1_,y1_),(x2_,y2_),(0,255,0),2)
#     output_image = input_image[y1_:y2_, x1_:x2_]
#     return output_image
#
# j = 0
# while j < 200:
#     input_img = cv2.imread('./jpg/%d.jpg'%j,cv2.IMREAD_GRAYSCALE)
#     output_img = crop_image(input_img)
#     cv2.imwrite('cropImg/%d.jpg'%j,output_img)
#     j += 1