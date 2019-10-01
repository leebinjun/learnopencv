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
X_train = features
y_train = np.array(labels).reshape(-1,1)
print("y_train:", y_train[:2])

svm0.train(X_train, cv.ml.ROW_SAMPLE, y_train)
sv0 = svm0.getSupportVectors()
rho0, _, _ = svm0.getDecisionFunction(0)
sv0 = np.transpose(sv0) 
hog = cv.HOGDescriptor()
hog.setSVMDetector(np.append(sv0,[[-rho0]],0))
hog.save('hogsvm.bin')
print('Finished!!!!\n')
