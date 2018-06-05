#!/usr/bin/env python
# -*- coding:utf-8 -*-
import dlib
import cv2
import os
import sys
import random

output_dir='E://code/my_faces'
#获得图片后存放的路径
size=64
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#改变图片的亮度与对比度
def relight(img,light=1,bias=0):
    w=img.shape[1]#图片的宽
    h=img.shape[0]#图片的高
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp=int(img[j,i,c]*light+bias)
                if tmp>255:
                    tmp=255
                elif tmp<0:
                    tmp=0
                img[j,i,c]=tmp
    return img



detector=dlib.get_frontal_face_detector()
#脸部特征提取器
camera=cv2.VideoCapture(0)
#打开摄像头，参数为输入流，可以为摄像头或视频文件
index=1
while True:
    if(index<=10000):
        print('Being processed picture %s' % index)
        success,img=camera.read()
        #从摄像头读取照片
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #转为灰度图片
        dets=detector(gray_img,1)
        #使用detector进行人脸检测
        for i,d in enumerate(dets):
            #enumerate表示枚举，i是指序号index，d指dets中的内容
            x1=d.top() if d.top()>0 else 0
            y1=d.bottom() if d.bottom()>0 else 0
            x2=d.left() if d.left()>0 else 0
            y2=d.right() if d.right()>0 else 0
            face=img[x1:y1,x2:y2]#灰度图像仅用来定位人脸的位置，后续任然用RGB3通道的图像
            #d就是识别出来的人脸的一个矩形框，d的参数就是矩形框的位置
            face=relight(face,random.uniform(0.5,1.5),random.randint(-50,50))
            #uniform用于生成指定范围内的浮点数，randint用于生成指定范围内的整数
            face=cv2.resize(face,(size,size))
            #将所有的图片定义为相同大小
            cv2.imshow('image',face)
            cv2.imwrite(output_dir+'/'+str(index)+'.jpg',face)
            index+=1
        key=cv2.waitKey(30)&0xff
        if key==27:
            break
    else:
        print('Finished!')
        break




