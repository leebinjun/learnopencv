import cv2 as cv
import random
import glob
import os
import numpy as np
from imutils.object_detection import non_max_suppression
from PIL import Image

# real time predict
capture = cv.VideoCapture(0)  # 视频文件路径
rate = 24  # 帧率
stop = False
delay = int(1000/rate)

while not stop:
    hogtest = cv.HOGDescriptor()
    hogtest.load('hogsvm.bin')
    rval, frame = capture.read()
    print(rval)
    rects, weights = hogtest.detectMultiScale(frame, scale = 1.03)#参数可调

    #weight
    weights = [weight[0] for weight in weights]
    weights = np.array(weights)

    #这里返回的四个值表示的是开始始位置（x,y),长宽（xx,yy)，所以做以下处理
    for i in range(len(rects)):
        r = rects[i]
        rects[i][2] = r[0] + r[2]
        rects[i][3] = r[1] + r[3]


    choose = non_max_suppression(rects, probs = weights, overlapThresh = 0.5)#参数可调（可以把overlapThresh调小一点，也不要太小）

    for (x,y,xx,yy) in choose:
        cv.rectangle(frame, (x, y), (xx, yy), (0, 0, 255), 2)
    cv.imshow('Video', frame)
    if cv.waitKey(delay) >= 0:
        stop=True
capture.release()
cv.destroyAllWindows()