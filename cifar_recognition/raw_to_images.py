# cifar-10包含5万张训练图片和1万张测试图片，每张图片为32x32的RGB图像
# 原始训练数据打包存放在data_batch_1、2、3、4、5五个文件中，测试数据打包存放在test_batch文件中
# 本代码将原始训练/测试数据转换成图片，并根据类别分别保存在0~9十个文件夹

import cv2
import numpy as np
import os

root = 'cifar_recognition/'

# 创建保存图片的文件夹，data/train/0~9, data/test/0~9
if not os.path.exists(root+'data/train'):
    os.makedirs(root+'data/train')
if not os.path.exists(root+'data/test'):
    os.makedirs(root+'data/test')

for i in range(10):
    if not os.path.exists(root+'data/train/'+str(i)):
        os.makedirs(root+'data/train/'+str(i))
    if not os.path.exists(root+'data/test/'+str(i)):
        os.makedirs(root+'data/test/'+str(i))


raw_path = 'cifar_recognition/data/cifar-10-batches-py/'

# 循环读取data_batch_1、2、3、4、5五个文件，并拆分保存成图片
for i in range(1,6):
    data = np.load(raw_path+'data_batch_'+str(i), encoding='bytes', allow_pickle=True)
    images = data[b'data']
    labels = data[b'labels']
    filenames = data[b'filenames']

    # 循环读取每个data_batch中的图片、类别标签和图片名
    for j, (image, label, filename) in enumerate(zip(images, labels, filenames)):
        image = np.reshape(image, (3, 32, 32)) # 将读取出的长度为3072的一维数据，需要转化成3x32x32的矩阵
        image = np.transpose(image, (1,2,0)) # 将3x32x32的矩阵转换成OpenCV可处理的32x32x3的图像数据
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #RGB转BGR,使OpenCV可以将其保存成正确的图像

        #cv2.imshow('1', image)
        #cv2.waitKey()
        cv2.imwrite(root+'data/train/'+str(label)+'/'+str(filename,'UTF-8'), image) #保存图像


# 读取test_batch文件，并拆分保存成图片
data = np.load(raw_path+'test_batch', encoding='bytes', allow_pickle=True)
labels = data[b'labels']
images = data[b'data']
filenames = data[b'filenames']

for j, (label, image, filename) in enumerate(zip(labels, images, filenames)):
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1,2,0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(root+'data/test/'+str(label)+'/'+str(filename, 'UTF-8'), image)