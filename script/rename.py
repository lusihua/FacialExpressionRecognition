# -*- coding:utf8 -*-

import sys
import shutil
import os
current_dir = os.path.abspath(os.path.dirname(__file__))
print(current_dir) #F:\project\pritice

current_dir1 = os.path.dirname(__file__)
print(current_dir1) #F:/project/pritice

parent_path = os.path.dirname(current_dir1)
print(parent_path) #F:/project

parent_path1  = os.path.dirname(parent_path)
print(parent_path1) #F:/



root_path = '/media/wan/36B0ED8054668DCA/dataset/dataset/affectfer/datasets48an'

class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''
    def __init__(self):
        self.path = root_path + '/sadness/'  #表示需要命名处理的文件夹
        self.save_path = '/media/wan/36B0ED8054668DCA/dataset/train/sadness'  # copy to path
        print(self.save_path)
    def rename(self):
        filelist = os.listdir(self.path) #获取文件路径
        print(filelist)
        total_num = len(filelist) #获取文件长度（个数）
        i = 1  #表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.jpg'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                src = os.path.join(os.path.abspath(self.path), item)
                #dst = os.path.join(os.path.abspath(self.path), 'fer' +str(i) + '.jpg')#处理后的格式也为jpg格式的，当然这里可以改成png格式

                dst = os.path.join(os.path.abspath(self.save_path), 'afe_ano' + str(i) + '.jpg')
                #dst = os.path.join(os.path.abspath(self.path), 'fer' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    # os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    shutil.copy(src, dst)
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()