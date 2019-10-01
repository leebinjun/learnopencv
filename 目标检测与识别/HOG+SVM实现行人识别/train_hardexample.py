import cv2 as cv
import random
import glob
import os
import numpy as np
from imutils.object_detection import non_max_suppression
from PIL import Image

features = np.load(r'X_data.npy')
labels = np.load(r'y_data.npy')
print(features[:2])
print(labels[:20])

def svm_create():
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_EPS_SVR)
    svm.setKernel(cv.ml.SVM_LINEAR)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setP(0.1)
    svm.setC(0.01)
    return svm

# 2.4 使用 Hardexample 优化模型
# 在此，使用 5 次 Hardexample 方法，以使模型更优：

hefoldername = r".\INRIADATA\normalized_images\train\neg"
# hardexample
for k in range(5):
    #get hardexample
    hoghe = cv.HOGDescriptor()
    hoghe.load('hogsvm.bin')
    hardexamples = []
    hefilenames = glob.glob(os.path.join(hefoldername,'*'))
    for hefilename in hefilenames:
        heimg = cv.imread(hefilename, 1)
        rects, weight = hoghe.detectMultiScale(heimg, 0, scale = 1.03)#参数可调
        for (x,y,w,h) in rects:
            heres = heimg[y : y + h, x : x + w]
            hardexamples.append(cv.resize(heres,(64,128)))

    for k in range(len(hardexamples)):
        features = np.append(features, [hoghe.compute(hardexamples[k])], axis=0)
        labels = np.append(labels, [-1], axis=0)

    if len(features) == len(labels):
        print('allfeatures = '+str(len(features))+'\n')

    #train hardexample(allfeatures)
    svm = svm_create()
    print('Training svm...\n')
    X_train = features
    y_train = np.array(labels).reshape(-1,1)
    # print("y_train:", y_train[:2])

    svm.train(X_train, cv.ml.ROW_SAMPLE, y_train)
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    hoghe.setSVMDetector(np.append(sv,[[-rho]],0))
    hoghe.save('hogsvm.bin')
    print('Finished!!!!\n')
