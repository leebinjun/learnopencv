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
imgh = 120
imgw = 65

posfoldername = r"C:\Users\Administrator\Desktop\people_detect_hog_svm.py\INRIADATA\normalized_images\train\pos"
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

negfoldername = r"C:\Users\Administrator\Desktop\people_detect_hog_svm.py\INRIADATA\normalized_images\train\neg"
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

# 2.3 构建并训练 SVM 模型
#function for create svm
def svm_create():
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_EPS_SVR)
    svm.setKernel(cv.ml.SVM_LINEAR)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setP(0.1)
    svm.setC(0.01)
    return svm

svm0 = svm_create()
print('Training svm0...\n')
X_train = np.array(features).reshape(-1, 1)
y_train = np.array(labels).reshape(-1,1)
print("X_train:", X_train[:2])
print("y_train:", y_train[:2])
print(X_train.shape, y_train.shape)

svm0.train(X_train, cv.ml.ROW_SAMPLE, y_train)
sv0 = svm0.getSupportVectors()
rho0, _, _ = svm0.getDecisionFunction(0)
sv0 = np.transpose(sv0)
hog.setSVMDetector(np.append(sv0,[[-rho0]],0))
hog.save('hogsvm.bin')
print('Finished!!!!\n')
# 2.4 使用 Hardexample 优化模型
# 在此，使用 5 次 Hardexample 方法，以使模型更优：

# hardexample
for k in range(5):
    #get hardexample
    hoghe = cv.HOGDescriptor()
    hoghe.load('hogsvm.bin')
    hardexamples = []
    hefilenames = glob.iglob(os.path.join(hefoldername,'*'))
    for hefilename in hefilenames:
        heimg = cv.imread(hefilename,1)
        rects,weight = hog.detectMultiScale(heimg,0,scale = 1.03)#参数可调
        for (x,y,w,h) in rects:
            heres = heimg[y : y + h, x : x + w]
            hardexamples.append(cv.resize(heres,(64,128)))

    for k in range(len(hardexamples)):
        features.append(hog.compute(hardexamples[k]))
        labels.append(-1)

    if len(features) == len(labels):
        print('allfeatures = '+str(len(features))+'\n')

    #train hardexample(allfeatures)
    svm = svm_create()
    print('Training svm...\n')
    svm.train(np.array(features),cv.ml.ROW_SAMPLE,np.array(labels))
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    hog.setSVMDetector(np.append(sv,[[-rho]],0))
    hog.save('hogsvm.bin')
    print('Finished!!!!\n')
