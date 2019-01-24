# -*- coding: utf-8 -*-
"""
This is an implementation of <Improved boosting algorithms using 
confidence-rated predictions>, Schapire, 1999.
"""

from math import e, log,exp
import numpy as np
from numpy import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import train_test_split
import os
import processdata

class Adaboost():
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
    def __init__(self, X, y, estimator = DecisionTreeClassifier, itern = 20,max_depth = 5, mode = "sign"):
        self.X = X
        self.y = y.copy()
        self.estimator = estimator
        self.mode = mode
        self.itern = itern
        self.estimators = [] # estimators produced by boosting algorithm
        self.alphas = np.array([])  # weights of each boost estimator
        self.m = self.X.shape[0] # number of samples
        self.w = np.array([1.0/self.m] * self.m) # weights of samples
        self.max_depth = max_depth
        
    def train(self):
        m = self.m
        print "round number: ",self.itern
        for k in range(self.itern):
            clf = self.estimator(max_depth=self.max_depth, presort = True)
            clf.fit(self.X, self.y, sample_weight = self.w)
            self.estimators.append(clf)
            y_predict = clf.predict(self.X) 
            error = 0  # number of wrong prediction
            for i in range(m):
                if y_predict[i] != self.y[i]:
                    error += self.w[i]
            if error == 0:
                error += 0.01 # smoothness
            alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16))) # estimator weight
            self.alphas = np.append(self.alphas, alpha)
            for i in range(m): # update sample weights
                if y_predict[i] != self.y[i]:
                    self.w[i] *= e**alpha
                else:
                    self.w[i] /= e**alpha
            self.w /= sum(self.w)

        # train_score = score(self.X,self.y)   
        # print "train score: %.4f" %train_score

    def predict(self, X):
        y_predict = np.array([])
        if self.mode == "sign":
            for i in range(X.shape[0]):
                predict_i = (sum(self.alphas * 
                                 np.array([int(self.estimators[k].predict(X[i].reshape(1,-1))) for k in range(len(self.alphas))])))
                y_predict = np.append(y_predict, np.sign(predict_i))
        else:
            for i in range(X.shape[0]):
                predict_i = (sum(self.alphas * 
                                 np.array([int(self.estimators[k].predict(X[i].reshape(1,-1))) for k in range(len(self.alphas))])))
                y_predict = np.append(y_predict, predict_i)
            
        return y_predict
    
    def score(self, X_test, y_test):
        """return precision of trained estimator on x_test and y_test"""  
        y_predict = self.predict(X_test)

        sumy=0
        error = 0 # error
        for i in range(X_test.shape[0]):
            if y_predict[i] == 1:
                sumy +=1

            if y_predict[i] != y_test[i]:
                error += 1
        error /= 1.0* X_test.shape[0]
        # print "sumy", sumy
        return 1 - error
            

class AdaboostMR():
    def __init__(self, X, y, class_num, estimator = DecisionTreeClassifier, itern = 20,max_depth = 5,mode='sign'):
        self.X = X
        self.y = y.copy()
        self.class_num = class_num
        self.estimator = estimator
        self.mode = mode
        self.itern = itern
        self.estimators = [] # estimators produced by boosting algorithm
        self.alphas = np.array([])  # weights of each boost estimator
        self.m = self.X.shape[0] # number of samples
        self.D = np.zeros((self.m,self.class_num,self.class_num)) # weights of samples
        self.label_list = []
        self.max_depth =max_depth


    def train(self):
       
        m = self.m
        D = self.D
        print "round number: ",self.itern

        # init D
        for i in range(m):
            l1 = self.y[i]
            # print l1
            l0s = [l for l in range(0,self.class_num)]
            l0s.remove(l1)
            D[i,l0s,l1] = 1.0/((self.class_num-1)*1*m/self.class_num)

        # print "D.sum(): ",D.sum()
        # D = D/ D.sum()

        weightArrays = np.zeros(m)

        for k in range(self.itern):
            preds = np.zeros((m))
            weightArrays = [sum(sum(D[i])) for i in range(m)]
            clf = self.estimator(
                criterion = 'entropy',
                splitter = 'best',
                max_depth = self.max_depth)

            #training 
            clf = clf.fit(self.X, self.y, sample_weight = weightArrays)
            self.estimators.append(clf)
            preds = clf.predict(self.X) 
            pbs = clf.predict_proba(self.X)

            error = 0.0  # number of wrong prediction
            error = 1 - clf.score(self.X,self.y,sample_weight = weightArrays)

            # if k==self.itern -1:
            #     print "train score: %.4f" %(1 - error)

            # alpha = 0.5 * log((1 - error) / max(error, 1e-16))
            alpha = 0.5 * log((1 + error) / max(1-error, 1e-16))
            self.alphas = np.append(self.alphas, alpha)

            #update D

            for i in range(m):
                for l0 in range(self.class_num):
                    for l1 in range(self.class_num):
                        D[i,l0,l1]  = D[i,l0,l1] * exp(0.5 * alpha * (pbs[i][l0] - pbs[i][l1]))
            Z = sum(sum(sum(D)))
            D /= Z

    def test(self,data,target):
        test_pred_prob = np.zeros((data.shape[0],self.class_num))
        for i in range(self.itern):
            test_pred_prob += self.estimators[i].predict_proba(data) * self.alphas[i]
        test_pred_prob = test_pred_prob.tolist()
        test_pre = []
        for i in range(len(test_pred_prob)):
            maxidx = test_pred_prob[i].index(max(test_pred_prob[i]))
            test_pre.append(maxidx)

        correct = 0
        for index in range(data.shape[0]):
            if test_pre[index] == target[index]:
                correct +=1
            # else:
            #     print "index: ", index, "test_pre: ",test_pre[index], "target: ",target[index]
        score = correct * 1.0 /data.shape[0]
        # print "correct data: " ,correct, "total data: ",data.shape[0]
        # print "score: %.4f" %(score) 
        return correct,data.shape[0],score


        
        
        
        
        
        
        
        
        

            
                
                
