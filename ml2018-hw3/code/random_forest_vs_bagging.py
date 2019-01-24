# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
import processdata
import time
import os

# bagging
def bagging(x_train, y_train, x_test, y_test):
    meta_clf = tree.DecisionTreeClassifier()
    clf = BaggingClassifier(meta_clf, n_estimators=100)
    print "---start train bagging-----"
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    print "---train bagging end ------"
    print "train time: ", end_time - start_time
    print "test score: ",accuracy_score(y_test, clf.predict(x_test))


# random forest
def random_forest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    print "---start train random_forest-----"
    start_time = time.time()
    clf.fit(x_train, y_train)
    print "---train random_forest end ------"
    end_time = time.time()
    print "train time: ", end_time - start_time
    print "test score: ",accuracy_score(y_test, clf.predict(x_test))

chess_path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework3/data"
chess_data = os.path.join(chess_path,"krkopt.data")
x,y = processdata.process_krkopt(chess_data)
x_train ,x_test, y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=3)
bagging(x_train, y_train, x_test, y_test)
random_forest(x_train, y_train, x_test, y_test)
