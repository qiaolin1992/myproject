#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
size=64
x=tf.placeholder(tf.float32,[None,size,size,3])
y_=tf.placeholder(tf.float32,[None,2])
keep_prob_5=tf.placeholder(tf.float32)#定义一个占位符
keep_prob75=tf.placeholder(tf.float32)
#权重初始化
def weightVariable(shape):
    init=tf.random_normal(shape,stddev=0.01)#初始化标准差为1的一个矩阵
    return tf.Variable(init)
#偏差初始化
def biasVariable(shape):
    init=tf.random_normal(shape)
    return tf.Variable(init)
#卷积操作
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化操作
def maxPool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#防止过拟合的操作
def dropout(x,keep):
    return tf.nn.dropout(x,keep)
#CNN网络模型
def cnnLayer():
    #第一层卷积层
    w1=weightVariable([3,3,3,32])
    b1=biasVariable([32])
    conv1=tf.nn.relu(conv2d(x,w1)+b1)
    pool1=maxPool(conv1)
    drop1=dropout(pool1,keep_prob_5)
    #第二层卷积层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)
    # 第三层卷积层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)
    #全连接层
    wf=weightVariable([8*8*64,512])
    bf=biasVariable([512])
    drop3_flat=tf.reshape(drop3,[-1,8*8*64])#将最后的矩阵拉平
    dense=tf.nn.relu(tf.matmul(drop3_flat,wf)+bf)
    dropf=dropout(dense,keep_prob75)
    #输出层
    wout=weightVariable([512,2])
    bout=biasVariable([2])
    out=tf.add(tf.matmul(dropf,wout),bout)
    return out

output=cnnLayer()
predict=tf.argmax(output,1)
saver=tf.train.Saver()
sess=tf.Session()
saver.restore(sess,tf.train.latest_checkpoint('E://code/model/'))#导入模型的方案
def is_my_face(image):
    res=sess.run(predict,feed_dict={x:[image/255.0],keep_prob_5:1.0,keep_prob75:1.0})
    if res[0]==1:
        return True
    else:
        return False

detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)
#通过摄像头获取图像
while True:
    _, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1,x2:y2]
        # 调整图片的尺寸
        face = cv2.resize(face, (size,size))
        print('Is this my face? %s' % is_my_face(face))

        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)#cv2.rectangle(image, 左下角坐标, 右上角坐标, color)
        cv2.imshow('image',img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

sess.close()
