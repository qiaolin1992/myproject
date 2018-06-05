import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding,Dropout,Dense,Merge
k=128
ratings=pd.read_csv('E:/code/data/ml-1m/ratings.dat',sep='::',engine='python',names=['user_id','movie_id','rating','timestamp'])
n_users=np.max(ratings['user_id'])
n_movies=np.max(ratings['movie_id'])
print([n_users,n_movies,len(ratings)])
plt.hist(ratings['rating'])
plt.show()
print(np.mean(ratings['rating']))
