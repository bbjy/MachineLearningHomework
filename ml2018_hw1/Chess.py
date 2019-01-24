import os
import csv
import numpy as np
import sklearn 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework1/Chess"

datafile = os.path.join(path,"krkopt.data")

# preprocess the dataset 
def data_process(pd_dataset):
	labelmap = {"draw":17,"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16}


	l = pd_dataset.shape[0]
	data = np.zeros((l,48))
	#label = np.zeros((l,18))
	label = np.zeros((l))
	x= pd_dataset.iloc[:,:6].as_matrix()
	y= pd_dataset.iloc[:,-1].as_matrix()
	
		
	for i in range(y.shape[0]):
		
		x[i,0] = int(ord(x[i,0]) - ord('a'))
		tmp0 = x[i,0]
		tmp1 = 8 + (x[i,1] - 1)
		data[i,tmp0] = 1
		data[i,tmp1] = 1
		x[i,2] = int(ord(x[i,2]) - ord('a'))
		tmp2 = 2*8 + x[i,2] 
		tmp3 = 3*8 + (x[i,3] - 1)
		data[i,tmp2] = 1
		data[i,tmp3] = 1
		x[i,4] = int(ord(x[i,4]) - ord('a'))
		tmp4 = 4*8 + x[i,4] 
		tmp5 = 5*8 + (x[i,5] - 1)
		data[i,tmp4] = 1
		data[i,tmp5] = 1
		# tmp6 = labelmap[y[i]]
		# label[i,tmp6] = 1
		label[i] = labelmap[y[i]]


	#np.savetxt("Chess/chessdata.txt",data)
	#np.savetxt("Chess/chesslabel.txt",label)

	print(data[0])
	print("============================")
	print(label[0])

	return x,label
#train dataset 
traindata = pd.read_csv(datafile,header=None)#.as_matrix()

x,y = data_process(traindata)
# Correlation Matrix

import seaborn as sns

def table_corr(df):
	fig2 = plt.figure(2)
	plt.title("heatmap")
	sns.heatmap(df.corr(), annot=True)
	fig2.show()
    
table_corr(traindata)
input()
# print(x.shape)
# print(y.shape)
# print(type(y))	

# with open("chess_y.txt",'a') as f:
# 	for line in y:
# 		f.write(line)
#np.savetxt('Chess_Y',y)
# print(y[0:50])

#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1)
#print("The test/train = 0.1")
# print(x_train.shape)
# print(y_train.shape)

#method1: Multi layer perceptron
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs')  #For small datasets, however, ‘lbfgs’ can converge faster and perform better.
start = time.clock()
mlp.fit(x_train,y_train)
pre_mlp = mlp.predict(x_test)
acc_mlp = accuracy_score(y_test,pre_mlp)
#f1_mlp = f1_score(y_test, pre_mlp, average='micro') 

#scores = cross_val_score(mlp,x,y,cv = 10, scoring='f1_micro')
#f1_mlp = scores.mean()
end = time.clock()
#print("The classification accuracy of Multi layer perceptronis is: %.4f\n" %acc_mlp) # 0.5933
print("The f1-score of Multi layer perceptronis is: %.4f\n" %acc_mlp) # 0.5933

print("The running time of Multi layer perceptronis is: %.2fs\n" %(end-start)) # 174.03s

#method2: SVM
from sklearn.svm import SVC
clf_svm = SVC(gamma='auto')
start = time.clock()

clf_svm.fit(x_train,y_train)
pre_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(y_test,pre_svm)
#f1_svm = f1_score(y_test, pre_svm, average='micro') 

#scores = cross_val_score(clf_svm,x,y,cv = 10,scoring='f1_micro')
#f1_svm = scores.mean()

end = time.clock()

#print("The classification accuracy of SVM is: %.4f\n" %acc_svm) # 0.5943
print("The f1-score of svm is: %.4f\n" %acc_svm) # 0.5933

print("The running time of SVM is: %.2fs\n" %(end-start)) #1644.04s

#method3: Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
start = time.clock()
'''
gnb.fit(x_train,y_train)
pre_gnb = gnb.predict(x_test)
acc_gnb = accuracy_score(y_test,pre_gnb)
'''
scores = cross_val_score(gnb,x,y, cv = 10,scoring='f1_micro')
f1_gnb = scores.mean()
end= time.clock()
#print("The classification accuracy of Naive Bayes is: %.4f\n" %acc_gnb) # 

print("The classification accuracy of Naive Bayes is: %.4f\n" %f1_gnb) # 
print("The running time of Naive Bayes is: %.2fs\n" %(end-start))

#method4: KNN
from sklearn.neighbors import KNeighborsClassifier
k_range = range(50,200)
test_acc = []
'''
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(x_train,y_train)
	pre_knn = knn.predict(x_test)

	acc_knn = accuracy_score(y_test,pre_knn)
	test_acc.append(acc_knn)
import matplotlib.pyplot as plt
plt.plot(k_range,test_acc)
plt.xlabel("Value of k for KNN")
plt.ylabel("Test accuracy")
'''
start = time.clock()
knn = KNeighborsClassifier(n_neighbors=100)
scores = cross_val_score(knn,x,y, cv = 10,scoring='f1_micro')
f1_knn = scores.mean()
end= time.clock()
print("The f1-score of  knn is: %.4f\n" %f1_knn) # 0.5933

print("The running time of knn is: %.2fs\n" %(end-start)) #1644.04s

# method5: Decision tree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
start = time.clock()
'''
dtree.fit(x_train,y_train)
pred_tree = dtree.predict(x_test)
acc_dtree = accuracy_score(y_test,pred_tree)
'''
scores = cross_val_score(dtree,x,y,cv = 10,scoring='f1_micro')
f1_dtree = scores.mean()

end= time.clock()

print("The classification accuracy of Decision Tree is: %.4f\n" %f1_dtree) # 0.5202
print("The running time of Decision tree is: %.2fs\n" %(end-start))


#method6: Logistic Regression
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
start = time.clock()
'''
lg.fit(x_train,y_train)
pre_lg = lg.predict(x_test)
acc_lg = accuracy_score(y_test,pre_lg)
'''
scores = cross_val_score(lg,x,y,cv = 10,scoring='f1_micro')
f1_lg = scores.mean()


end = time.clock()
#print("The classification accuracy of Logistic Regression is: %.4f\n" %acc_lg) # 
print("The classification accuracy of Logistic Regression is: %.4f\n" %f1_lg) # 

print("The running time of Logistic Regression is: %.2fs\n" %(end-start))





