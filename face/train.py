import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

my_faces_path='E://code/my_faces'
other_faces_path='E://code/other_faces'
size=64
imgs=[]
labs=[]
#为了使图像呈正方形，使得长宽相等
def getPaddingSize(img):
    h,w,_=img.shape #图像是三通道的
    top,bottom,left,right=(0,0,0,0)
    longest=max(h,w)
    if w<longest:
        tmp=longest-w
        left=tmp//2 #//表示整除符号
        right=tmp-left
    elif h<longest:
        tmp=longest-h
        top=tmp//2
        bottom=tmp-top
    else:
        pass
    return top,bottom,left,right


def readData(path,h=size,w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename=path+'/'+filename #后面的filename仅仅指代文件的名字
            img=cv2.imread(filename)
            top,bottom,left,right=getPaddingSize(img)
            img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
            #cv2.copyMakeBorder给图片加边框，top和bottom扩充长度，left和right扩充高度，cv2.BORDER_CONSTANT用后面的value指填充
            #进行上一步的目的是防止输入图片不是正方形
            img=cv2.resize(img,(h,w))
            imgs.append(img)
            labs.append(path)

readData(my_faces_path)
readData(other_faces_path)
imgs=np.array(imgs)
print(imgs.shape)
labs=np.array([[0,1]if lab==my_faces_path else [1,0] for lab in labs])
#讲图片数据与标签数据转换为数组，这里构造标签的方法值得学习
#至此，数据集构造完成，下面开始进行模型构造
train_x,test_x,train_y,test_y=train_test_split(imgs,labs,test_size=0.05,random_state=random.randint(0,100))
#随机划分训练集和测试集
train_x=train_x.reshape(train_x.shape[0],size,size,3)
test_x=test_x.reshape(test_x.shape[0],size,size,3)
print(train_x.shape)
#将图片数据变为4维矩阵，分别是图片总数，图片的高、宽、通道
train_x=train_x.astype('float32')/255.0
test_x=test_x.astype('float32')/255.0
#归一化数据，使得数据的值小于1
print('train size:%s,test size:%s'%(len(train_x),len(test_x)))
batch_size=100
num_batch=len(train_x)//batch_size
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

#网络的训练
def cnnTrain():
    out=cnnLayer()
    #loss函数的定义
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y_))#softmax函数的交叉熵，softmax_cross_entropy_with_logits的输出是一个向量，如果需要求loss得求向量的平均
    #优化算法
    train_step=tf.train.AdadeltaOptimizer(0.01).minimize(cross_entropy)
    #预测准确率的计算
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out,1),tf.argmax(y_,1)),tf.float32))#argmax返回矩阵中最大值的索引，equal是对比这两个矩阵或者向量的相等的元素，
    # 如果是相等的那就返回True，不等返回False,cast是强制格式转换，分类任务的做法
    tf.summary.scalar('loss',cross_entropy)
    tf.summary.scalar('accuracy',accuracy)
    #把需要可视化的数据用summary保存起来
    merged_summary_op=tf.summary.merge_all()
    #把需要显示的数据融合起来，方便后面一起训练
    saver=tf.train.Saver()#保存模型的函数
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())#初始化全局变量
        summary_writer=tf.summary.FileWriter('E://code/tmp',graph=tf.get_default_graph())
        #tensorflow.summary.FileWriter将图运行得到的summary数据写到磁盘里
        for n in range (100):
            for i in range(num_batch):
                #每一次用给定的batch数量的数据修改参数
                batch_x=train_x[i*batch_size:(i+1)*batch_size]
                batch_y=train_y[i*batch_size:(i+1)*batch_size]
                _, loss, summary=sess.run([train_step, cross_entropy, merged_summary_op],
                                          feed_dict={x: batch_x,y_: batch_y,keep_prob_5: 0.5,keep_prob75: 0.75})
               # _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           # feed_dict={x: batch_x, y_: batch_y, keep_prob_5: 0.5, keep_prob75: 0.75})
                summary_writer.add_summary(summary,n*num_batch+i)
                #把每次训练的数据保存起来
                print(n*num_batch+i,loss)
                #每训练100次的时候验证模型——测试数据的正确率
                if(n*num_batch+i)%100==0:
                    acc=accuracy.eval({x:test_x,y_:test_y,keep_prob_5:1.0,keep_prob75:1.0})
                    print(n*num_batch+i,acc)
                    if acc>0.98 and n>2:
                        saver.save(sess,'E://code/model/model.ckpt',global_step=n*num_batch+i)
                        sys.exit(0)
        saver.save(sess, 'E://code/model/model.ckpt', global_step=n * num_batch + i)
        print('accuracy less 0.98,exited!')

#训练模型
cnnTrain()

#运行上面的程序后想要在tensorboard中看到数据需要在命令窗口输入tensorboard --logdir=E:\code\tmp\


