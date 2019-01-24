'''
   Dota2_Games_results from UCI dataset ,URL:http://archive.ics.uci.edu/ml/datasets/dota2+games+results
   the following result is the obtained with the reduced and new value feaure after analyzing with PCA

'''
import os
import csv
import numpy as np
import sklearn 
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE

# preprocess the dataset 
def feature_filter(pd_dataset):
	dataset = pd_dataset.as_matrix()
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

path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework1/Dota2-Games-Dataset"

trainfile = os.path.join(path,"dota2Train.csv")
testfile = os.path.join(path,"dota2Test.csv")

pd_traindata = pd.read_csv(trainfile,header=None)
pd_testdata = pd.read_csv(testfile,header=None)
#pd_traindata = feature_filter(pd_traindata)
#pd_testdata = feature_filter(pd_testdata)
print(" # Using the orignal feature!")
'''
#plot dataset 
print("# TSNE(feature_filter) !")
dota2 = pd_traindata.iloc[:1000,1:]
mpl.rcParams['figure.figsize'] = (10, 10)
tsne = TSNE(n_components=2)
trandota2 = tsne.fit_transform(dota2)
fig1 = plt.figure(1)
plt.title(" TSNE(feature_filter)")
for i in range(len(trandota2)):
    if pd_traindata.iloc[:, 0][i] == 1:
        plt.scatter(trandota2[i][0], trandota2[i][1], c='red')
    else:
        plt.scatter(trandota2[i][0], trandota2[i][1], c='blue')
fig1.show()
#input()

print("# PCA!")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
#mpl.rcParams['figure.figsize'] = (10, 10)
#pca.fit(dota2)
dota2_pca = pca.fit_transform(dota2)

for xis in range(len(ddota2_pca)):
    if pd_traindata.iloc[:, 0][xis] == 1:
        plt.scatter(dota2_pca[xis][0], dota2_pca[xis][1], c='red')
    else:
        plt.scatter(dota2_pca[xis][0], dota2_pca[xis][1], c='blue')
plt.show()

# Correlation Matrix
print("# Correlation Matrix(feature_filter)")
import seaborn as sns

def table_corr(df):
	fig2 = plt.figure(2)
	plt.title("Correlation Matrix(feature_filter)")
	sns.heatmap(df.corr(), annot=True, cmap='summer')
	fig2.show()

    
table_corr(pd_traindata)
input()
'''
x_train = pd_traindata.iloc[:,4:]
y_train = pd_traindata.iloc[:,0]

x_test = pd_testdata.iloc[:,4:]
y_test = pd_testdata.iloc[:,0]
'''
print("# Methods!")
#method1: Multi layer perceptron
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='adam')  #For small datasets, however, ‘lbfgs’ can converge faster and perform better.
start = time.clock()
mlp.fit(x_train,y_train)
pre_mlp = mlp.predict(x_test)
acc_mlp = accuracy_score(y_test,pre_mlp)
end = time.clock()
print("The classification accuracy of Multi layer perceptronis is: %.4f\n" %acc_mlp) # 0.5933 ;after feature_filter:0.5341
print("The running time of Multi layer perceptronis is: %.2fs\n" %(end-start)) # 174.03s; after feature_filter: 108.31s

#method2: SVM
from sklearn.svm import SVC
clf_svm = SVC(gamma='auto')
start = time.clock()
clf_svm.fit(x_train,y_train)
pre_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(y_test,pre_svm)
end = time.clock()

print("The classification accuracy of SVM is: %.4f\n" %acc_svm) # 0.5943;after feature_filter: 0.5347
print("The running time of SVM is: %.2fs\n" %(end-start)) #1644.04s ;after feature_filter:2197.32s

#method3: Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
start = time.clock()
gnb.fit(x_train,y_train)
pre_gnb = gnb.predict(x_test)
acc_gnb = accuracy_score(y_test,pre_gnb)
end= time.clock()

print("The classification accuracy of Naive Bayes is: %.4f\n" %acc_gnb) #  0.5640 ;after feature_filter: 0.5256
print("The running time of Naive Bayes is: %.2fs\n" %(end-start)) #0.19s ;after feature_filter: 0.02s


#method4: KNN
from sklearn.neighbors import KNeighborsClassifier
k_range = [30,50,80,100]
train_acc = []
test_acc = []
start = time.clock()
#for k in [100]:
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(x_train,y_train)
	#train_acc.append(knn.score(x_train,y_train))
	#test_acc.append(knn.score(x_test,y_test))
pre_knn = knn.predict(x_test)

acc_knn = accuracy_score(y_test,pre_knn)
end= time.clock()
#test_acc.append(acc_knn)
'''
'''
fig3 = plt.figure(3)
plt.title('k-NN accuracy: Varying Number of Neighbors k')
plt.plot(k_range,test_acc,label = "Test accuracy")  #best:k=98 acc = 0.5654
plt.plot(k_range,train_acc,label = "Train accuracy")  #best:k=98 acc = 0.5654
plt.legend()
plt.xlabel("Value of neighbors k for KNN")
plt.ylabel("Accuracy")
fig3.show()
input()
'''
'''
print("The classification accuracy of KNN is: %.4f\n" %acc_knn) # 0.5654;after feature_filter:0.5197
print("The running time of KNN is: %.2fs\n" %(end-start)) # 5.36s ;after feature_filter:51.70s

# method5: Decision tree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth=10)
start = time.clock()
dtree.fit(x_train,y_train)
pred_tree = dtree.predict(x_test)
acc_dtree = accuracy_score(y_test,pred_tree)
end= time.clock()

print("The classification accuracy of Decision Tree is: %.4f\n" %acc_dtree) # 0.5221 ;after feature_filter:0.5110
print("The running time of Decision tree is: %.2fs\n" %(end-start)) #2.90s;after feature_filter:1.24s

#method6: Logistic Regression
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
start = time.clock()
lg.fit(x_train,y_train)
pre_lg = lg.predict(x_test)
acc_lg = accuracy_score(y_test,pre_lg)
end = time.clock()
print("The classification accuracy of Logistic Regression is: %.4f\n" %acc_lg) # 0.5977;after feature_filter:0.5318
print("The running time of Logistic Regression is: %.2fs\n" %(end-start)) # 0.56s;after feature_filter:0.80s

'''
# show the accuracy results
models = ['MLP', 'SVM','Bayes', 'KNN','DecisionTree', 'LogReg']
#scores = [acc_mlp,acc_svm,acc_gnb,acc_knn, acc_dtree,acc_lg]
scores = [0.5693,0.5985,0.5640,0.5693, 0.5505,0.5974]
fig4 = plt.figure(4)
plt.title("The clssification accuracy on Dota2-Game-results dataset(original feature)")
plt.bar(models, scores)
plt.ylim([0.5, 0.6])
fig4.show()
input()


'''
from sklearn.tree import DecisionTreeClassifier
train_acc = []
test_acc = []
d_range = range(10,50)
for d in d_range:
	dtree = DecisionTreeClassifier(max_depth=d)
	start = time.clock()
	dtree.fit(x_train,y_train)
	train_acc.append(dtree.score(x_train,y_train))
	test_acc.append(dtree.score(x_test,y_test))
	end= time.clock()

#print("The classification accuracy of Decision Tree is: %.4f\n" %acc_dtree) # 0.5221 ;after feature_filter:0.5110
#print("The running time of Decision tree is: %.2fs\n" %(end-start)) #2.90s;after feature_filter:1.24s

fig3 = plt.figure(3)
plt.title('dTree accuracy: Varying depth d')
plt.plot(d_range,test_acc,label = "Test accuracy")  #best:k=98 acc = 0.5654
plt.plot(d_range,train_acc,label = "Train accuracy")  #best:k=98 acc = 0.5654
plt.legend()
plt.xlabel("Value of d for dTree")
plt.ylabel("Accuracy")
fig3.show()
input()
'''