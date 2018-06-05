import keras
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
(x_train,y_train),(x_test,y_test)=imdb.load_data()
list1=[]
list1.extend(list(map(len,x_train)))
list1.extend(list(map(len,x_test)))
m=max(list1)
#print(m)
maxword=400
x_train=sequence.pad_sequences(x_train,maxlen=maxword)
x_test=sequence.pad_sequences(x_test,maxlen=maxword)
vocal_size=np.max([np.max(x_train[i])for i in range(x_train.shape[0])])+1#最大值就是词向量的个数加上1表示空格的值
model=Sequential()
model.add(Embedding(vocal_size,64,input_length=maxword))
model.add(Flatten())


