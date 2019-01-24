# -*-coding:utf-8-*-
'''
  @Author:Wangbei
  2018.11.27
  To preprocess the Chess (King-Rook vs. King) Data Set and the letter recognition dataset 
  from UCI data repository.
  dataset URL: http://archive.ics.uci.edu/ml/datasets.html
'''
import os
import numpy as np
import sklearn
import math
import pandas as pd
from sklearn.model_selection import train_test_split

def process_krkopt(raw_data):
    map_word = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
        ,'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'draw': 17}
    data = []
    labels = []
    with open(raw_data) as ifile:
        for line in ifile:
            tokens = line.strip().split(',')
            data.append([float(tk) if tk.isdigit() else float(ord(tk)-ord('a')) for tk in tokens[:-1]])
            labels.append(map_word.get(tokens[-1]))
    x = np.array(data)
    y = np.array(labels)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=3)
    # return x_train, x_test, y_train, y_test
    return x,y
    
def process_letter(raw_data):
  '''
    process letter recgnition dataset 
	'''
  data_set=np.loadtxt(raw_data,delimiter=",",dtype=bytes).astype(str)
  X=data_set[:,1:17]
  y=data_set[:,0]
  for i in range(X.shape[0]):
    y[i] =int(ord(y[i]) - ord('A'))
  X = X.astype(int)
  y = y.astype(int)
  # print type(X[0][0]),y.shape
  return X,y

def process_iris(raw_data):
  y_dic = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica': 2} 
  data_set=np.loadtxt(raw_data,delimiter=",",dtype=bytes).astype(str)
  X=data_set[:,:4]
  y=data_set[:,4]
  for i in range(X.shape[0]):
    y[i] = y_dic[y[i]]
  X = np.array(X,dtype = float)
  y = np.array(y,dtype = int)  
  # print X[0],y[0]
  # print type(X[0]), type(y[0])
  return X,y


# letter_path ="/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/letter-recognition"
# datafile = os.path.join(letter_path,"letter-recognition.data")
# process_letter(datafile)