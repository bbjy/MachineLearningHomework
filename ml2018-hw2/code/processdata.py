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

chess_dic = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h':8,
    "draw":17,
    "zero":0,
    "one":1,
    "two":2,
    "three":3,
    "four":4,
    "five":5,
    "six":6,
    "seven":7,
    "eight":8,
    "nine":9,
    "ten":10,
    "eleven":11,
    "twelve":12,
    "thirteen":13,
    "fourteen":14,
    "fifteen":15,
    "sixteen":16,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5,
    '6':6,
    '7':7,
    '8':8
    }

car_dic = {
    'vhigh': 4,
    'high': 3,
    'med': 2,
    'low': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5more': 5,
    'more': 5,
    'small': 3,
    'big': 1,
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3,
}

# for dota2
def feature_filter(pd_dataset):
  dataset = np.array(pd_dataset)
  new_dataset = []
  for line in dataset:
    values = line[0:4].tolist()
    for i in range(4,len(line)):
      if line[i] == 1:
        values.append(i-3) # the wining team using the (i-3)th hero
      elif line[i] == -1:
        values.append((i-3)*-1) # the hero used by the losing team
    values = values[0:4] + sorted(values[4:])
    for i in range(4,9):
      values[i] *= -1
    values = values[0:4] + values[9:] + sorted(values[4:9])
    new_dataset.append(values)
  return pd.DataFrame(new_dataset)

def process_dota2(raw_data):
  data = np.loadtxt(raw_data,delimiter=",")
  pd_traindata = pd.read_csv(raw_data,header=None)
  pd_traindata = feature_filter(pd_traindata)
 
  X = np.array(pd_traindata.iloc[:,4:])
  Y =  np.array(pd_traindata.iloc[:,0])

  # print type(X),X.shape," ",X[0]," "
  return X,Y

def process_chess(raw_data,dictionary = chess_dic):
    '''
    process Chess dataset 

  '''    
    data_set=np.loadtxt(raw_data,delimiter=",",dtype=bytes).astype(str)
    X_=data_set[:,:6]
    y_=data_set[:,6]
    X=[]
    for x in X_:
        X.append(list(map(lambda c:dictionary[c],x)))
    X=np.array(X,dtype=int)
    y=list(map(lambda c:dictionary[c], y_))
    y = np.array(y,dtype=int)
    # np.savetxt("chesswrong.txt",X)
    # np.savetxt("chesslabelwrong.txt",y)
    return X,y
    
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

def process_car(raw_data,dictionary=car_dic):
  
  data_set=np.loadtxt(raw_data,delimiter=",",dtype=bytes).astype(str)
  X_=data_set[:,:6]
  y_=data_set[:,6]
  X=[]
  for x in X_:
      X.append(list(map(lambda c:dictionary[c],x)))
  X=np.array(X)
  y=list(map(lambda c:dictionary[c], y_))
  y= np.array(y)
  return X,y

def process_abalone(raw_data):
  
  data_set=np.loadtxt(raw_data,delimiter=",",dtype=bytes).astype(str)
  X=data_set[:,:8]
  y=data_set[:,8]
  for i in range(X.shape[0]):
    X[i,0] =int(ord(X[i,0]) - ord('A'))

  X = np.array(X,dtype = float)
  y = np.array(y,dtype = int)
  for i in range(X.shape[0]):
    if y[i] == 29:
      y[i] =y[i] -2
    else:
      y[i] = y[i] -1
  
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


def process_banknote(raw_data):
  # two classes
  data_set=np.loadtxt(raw_data,delimiter=",")
  X=data_set[:,:4]
  y=data_set[:,4]
  X = np.array(X,dtype = float)
  y = np.array(y,dtype = int)  
  return X,y
# letter_path ="/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework2/ml_hw2_wb/letter-recognition"
# datafile = os.path.join(letter_path,"letter-recognition.data")
# process_letter(datafile)