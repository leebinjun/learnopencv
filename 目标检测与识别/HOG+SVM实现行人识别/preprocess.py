import cv2 as cv
import random
import glob
import os
import numpy as np
from imutils.object_detection import non_max_suppression
from PIL import Image


# 2.2 读取训练数据
# 本程序使用的是由 CSDN 网友整理过的 INRIA 数据集，原文链接。 使用下面的代码读取样本:
#get pos
imgh = 128
imgw = 64

posfoldername = r".\INRIADATA\normalized_images\train\pos"
posimgs = []
count = 0
posfilenames = glob.glob(os.path.join(posfoldername,'*'))
print("posfilenames: ", posfilenames)
for posfilename in posfilenames:
    img = Image.open(posfilename)   
    img.save(posfilename)
    posimg = cv.imread(posfilename,1)
    posres = posimg[16:16+imgh,13:13+imgw]
    posimgs.append(posres)
    count += 1
print('pos = '+ str(count) + '\n')

negfoldername = r".\INRIADATA\normalized_images\train\neg"
#get neg
negimgs = []
count = 0
negfilenames = glob.glob(os.path.join(negfoldername,'*'))
for negfilename in negfilenames:
    img = Image.open(negfilename)   
    img.save(negfilename)
    negimg = cv.imread(negfilename,1)
    for i in range(10):
        #负样本图片过少，由于图片足够大，随机切10次很大几率得到的图片不相同，可以将一张图片当两张使用
        if((negimg.shape[1] >= imgw) & (negimg.shape[0] >= imgh)):
            y = int(random.uniform(0,negimg.shape[1] - imgw))
            x = int(random.uniform(0,negimg.shape[0] - imgh))
            negres = negimg[x:x+imgh,y:y+imgw]
            negimgs.append(negres)
            count+=1
print('neg = '+str(count)+'\n')
# 其中，对负样本随机裁剪 10 次来获得，使用负样本量更大。 接下来，为每个样本打上标签，并使用 opencv 提供的方法来计算 HOG 向量：
#get features & labels
features = []
labels = []
hog = cv.HOGDescriptor()

for i in range(len(posimgs)):
    features.append(hog.compute(posimgs[i]))
    labels.append(1)
for j in range(len(negimgs)):
    features.append(hog.compute(negimgs[j]))
    labels.append(-1)

if len(features) == len(labels):
    print('features = '+str(len(features))+'\n')



print(len(features))
print(len(labels))

# save
np.save(r'X_data.npy', features)
np.save(r'y_data.npy', labels)
b = np.load(r'X_data.npy')
a = np.load(r'y_data.npy')
print(b[:2])
print(a[:20])

