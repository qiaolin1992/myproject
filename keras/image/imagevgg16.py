from keras.applications.vgg16 import VGG16
from keras.layers import Input,Flatten,Dense,Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
import cv2
import h5py as h5py
import numpy as np
ishape=224
model_vgg=VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))
for layer in model_vgg.layers:
    print(len(layer.get_weights()))
    break;


'''
for layer in model_vgg.layers:
    layer.trainable=False
model=Flatten(name='flatten')(model_vgg.output)
model=Dense(4096,activation='relu',name='fc1')(model)
model=Dense(4096,activation='relu',name='fc2')(model)
model=Dropout(0.5)(model)
model=Dense(10,activation='softmax',name='prediction')(model)
model_vgg_mnist_pretrain=Model(model_vgg.input,model,name='vgg16_pretrain')
#model_vgg_mnist.summary()
sgd=SGD(lr=0.05,decay=1e-5)#学习率随着迭代次数的增加值减少
model_vgg_mnist_pretrain.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#数据处理
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=[cv2.cvtColor(cv2.resize(i,(ishape,ishape)),cv2.COLOR_GRAY2BGR)for i in x_train]#将原始图片变成224*224*3
x_train=np.concatenate([arr[np.newaxis]for arr in x_train]).astype('float32')#首先将每张图片变成一个矩阵然后进行拼接
x_test=[cv2.cvtColor(cv2.resize(i,(ishape,ishape)),cv2.COLOR_GRAY2BGR)for i in x_test]
x_test=np.concatenate([arr[np.newaxis]for arr in x_test]).astype('float32')
x_train=x_train/255
x_test=x_test/255
def tran_y(y):
    y_ohe=np.zeros(10)
    y_ohe[y]=1
    return y_ohe
y_train_ohe=np.array([tran_y(y_train[i])for i in range(len(y_train))])
y_test_ohe=np.array([tran_y(y_test[i])for i in range(len(y_test))])
model_vgg_mnist_pretrain.fit(x_train,y_train_ohe,validation_data=(x_test,y_train_ohe),epochs=200,batch_size=128)
scores=model_vgg_mnist_pretrain.evaluate(x_test,y_train_ohe,verbose=2)
'''