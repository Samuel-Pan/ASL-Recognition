import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
# 导入OpenCV
import cv2
from matplotlib import pyplot

dict_labels = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',    
}

predModel = load_model('model_cnn.keras')

# 设定维度
dim = (28, 28) # 图像维度


x0 = 200 # 400px left of center
x1 = 400 # 400px right of center
y0 = 110 # 400px right of center
y1 = 310# 400px right of center

# 初始化视频捕获
video=cv2.VideoCapture(0, cv2.CAP_DSHOW)

cv2.namedWindow('Webcam') # 构建1个窗口
cv2.moveWindow('Webcam',40,30) # 放置窗口

while video.isOpened(): # 只要没有关掉实时摄像头
    ret,capture = video.read() # 抓取每个视频帧
    capture = cv2.flip(capture,1)
    cropped = capture[y0:y1, x0:x1] # 截取
    img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) # 转成灰度图
    img = cv2.GaussianBlur(img, (3,3), 0) # 图像平滑
    img = cv2.resize(img, dim) # 图像大小缩放
    #pyplot.imshow(img, cmap='gray') # 可视化展示图片
    #pyplot.show() # 展示
    img = np.reshape(img, (1,img.shape[0],img.shape[1],1))
    img = tf.cast(img, tf.float32)
    pred=predModel.predict(img)

    # 可视化实时效果
    cv2.rectangle(capture, (x0,y0),(x1,y1),(255,0,0),2) # 为图片添加矩形框
    cv2.putText(capture,'{} '.format(dict_labels[pred.argmax(1).item()]),(x0+25,y0+50),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),1) # 预测字母
    cv2.imshow('Webcam', capture) # 展示视频
    
    # 结果输出
    #print(pred)
    #print(dict_labels[pred.argmax(1).item()])
    
    # 退出视频输入
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()