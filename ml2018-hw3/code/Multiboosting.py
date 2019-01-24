# -*- coding: utf-8 -*-
"""
This is an implementation of <MultiBoosting: A Technique for Combining
Boosting and Wagging>, GEOFFREY I. WEBB, 2000.
"""
# @author: WangBei
# 2018-12-09
import math
from math import e, log,exp
import numpy as np
from numpy import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import train_test_split
import os
import processdata
from sklearn.model_selection import cross_val_score

class Adaboost(object):
    """
    Adaboost(X, y, estimator = DecisionTreeClassifier, itern = 20, mode = "sign")   
    Basic Adaboost to solve two-class problem
    Parameters
    ----------
    X: numpy 2d array (m samples * n features)
    y: numpy 1d array (m samples' label) 
    estimator: base_estimator of boosting
    itern: number of iterations
    mode: sign mode output label directly, while num mode output a confidence 
    rate x. The more positive x is ,the more likely the label is Adaboost.cls0;
    the more negative x is, the more likely the label is not Adaboost.cls0
    """
    def __init__(self, X, y, estimator = DecisionTreeClassifier, itern = 20):
        self.X = X
        self.y = y.copy()
        self.estimator = estimator
        self.itern = itern
        self.estimators = [] # estimators produced by boosting algorithm
        self.betas = np.array([])  # weights of each boost estimator
        self.m = self.X.shape[0] # number of samples
        self.w = np.ones((self.m )) # weights of samples
        # self.max_depth = max_depth
        self.bootstrap = range(0,self.m)
    @staticmethod
    def bootstrap_sample(data_num):
        idx = np.random.randint(0,data_num,size=(data_num))
        return idx
        
    def train(self):
        m = self.m
        # print "round number: ",self.itern
        for k in range(self.itern):
            clf = self.estimator()
            clf.fit(self.X[self.bootstrap], self.y[self.bootstrap])            
            y_predict = clf.predict(self.X[self.bootstrap]) 
            error = 0  # number of wrong prediction
            for i in range(m):
                if y_predict[i] != self.y[i]:
                    error += self.w[i]               
            error = 1.0* error/m

            if error > 0.5:
                self.bootstrap = self.bootstrap_sample(m)
                self.w =  np.ones((m))
                continue

            elif error == 0:
                self.betas = np.append(self.betas, 1e-10)
                self.bootstrap = self.bootstrap_sample(m)
                self.w =  np.ones((m))
            else:
                beta = float(log((1.0 - error) / error)) # estimator weight
                self.betas = np.append(self.betas, beta)
                for i in range(m): # update sample weights
                    if y_predict[i] != self.y[i]:
                        self.w[i] /= (2.0*error) 
                    else:
                        self.w[i] /= (2.0*(1-error))
                    if self.w[i] < 1e-8:
                        self.w[i] = 1e-8
            self.estimators.append(clf)
    
    def test(self, X_test, y_test):
        """return precision of trained estimator on x_test and y_test"""  
        result = []
        # y_test = list(y_test)
        for i in range(X_test.shape[0]):
            result.append([])
       
        for index, estimator in enumerate(self.estimators):
            y_test_result = estimator.predict(X_test)
            for index2, res in enumerate(result):
                res.append([y_test_result[index2], np.log(1/self.betas[index])])
        final_result = []
        # vote
        for res in result:
            dic = {}
            for r in res:
                dic[r[0]] = r[1] if not dic.has_key(r[0]) else dic.get(r[0]) + r[1]
            final_result.append(sorted(dic, key=lambda x:dic[x])[-1])

        # print (float(np.sum(final_result == y_test)) / len(y_test))
        test_score = float(np.sum(final_result == y_test)) / len(y_test)
        return final_result,test_score

            
class MultiBoosting(Adaboost):
    # def __init__(self, X, y,estimator = DecisionTreeClassifier, itern = 100):
    #     super(MultiBoosting, self).__init__(X,y,estimator,itern)
    def __init__(self, X, y,estimator = DecisionTreeClassifier, itern = 100):
        super(MultiBoosting, self).__init__(X,y,estimator,itern)
        self.iterations = []
        self.current_iteration = 0
        self.set_iterations()

    # sample from poisson
    @staticmethod
    def poisson_sample(data_num):
        bootstrap = []
        for i in range(data_num):
            tmp = data_num+1
            while tmp >= data_num:
                tmp = np.random.poisson(i, 1)
            bootstrap.append(tmp[0])
        return bootstrap

   # Table 3. Determining sub-committee termination indexes.
    def set_iterations(self):
        n = int(float(self.itern)**0.5)
        for i in range(n):
            self.iterations.append(math.ceil((i+1)*self.itern *1.0/n))
        for i in range(self.itern):
            self.iterations.append(self.itern)

    def train(self):
        m = self.m
        w = self.w
        # print "round number: ",self.itern
        for t in range(self.itern):
            if self.iterations[self.current_iteration] == t:
                self.bootstrap = self.poisson_sample(m)
                self.w = np.ones((m))
                self.current_iteration+=1

            clf = self.estimator()
            clf.fit(self.X[self.bootstrap], self.y[self.bootstrap])            
            y_predict = clf.predict(self.X[self.bootstrap]) 
            error = 0  # number of wrong prediction
            for i in range(m):
                if y_predict[i] != self.y[i]:
                    error += self.w[i]               
            error = error/m
            if error > 0.5:
                self.bootstrap = self.poisson_sample(m)
                self.w =  np.ones((m))
                self.current_iteration+=1
                continue

            elif error == 0:
                self.betas = np.append(self.betas, 1e-10)
                self.bootstrap = self.poisson_sample(m)
                self.w =  np.ones((m))
                self.current_iteration+=1
            else:
                beta = float(log((1.0 - error) / error)) # estimator weight
                self.betas = np.append(self.betas, beta)
                for i in range(m): # update sample weights
                    if y_predict[i] != self.y[i]:
                        self.w[i] /= (2*error) 
                    else:
                        self.w[i] /= (2*(1-error))
                for i in range(m):
                    if self.w[i] < 1e-8:
                        self.w[i] = 1e-8
            self.estimators.append(clf)
         
        
        
