# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:06:50 2020

@author: Module implementing the CART algorithm to grow a Tree
"""
import os
import csv
import numpy as np
from numpy import genfromtxt

directory = r'C:\Users\Spare 3\Desktop\kaggle\CART'
directory=r'/Users/pablo/Desktop/python_scripts_random/CART'

os.chdir(directory)

f               = open('spambase.names')
labels          = f.readlines()[33:]
labels          = [i.split(':')[0] for i in labels]
my_data_raw     = genfromtxt('spambase.data', delimiter=',')
subset          = np.random.default_rng().choice(my_data_raw.shape[0], size = 3000, replace = False)
my_data         = my_data_raw[subset,:]
Y               = my_data[:,my_data.shape[1]-1]#.reshape((my_data.shape[0],1))
X               = my_data[:,0:my_data.shape[1]-1]


class DecisionTree:
    
    def __init__(self, feature = 0,  treshold = None, right = None, left= None, x = None, y = None, classed = None, score = None):
        self.feature        = feature
        self.treshold       = treshold
        self.right          = right
        self.left           = left
        self.x              = x 
        self.y              = y
        self.m              = len(y)
        self.classed        = classed
        self.score          = score
    
    
def BuildTree(X, Y):
    '''
    Go through every feature to determine its best partition value
    Decide which is the purest partition
    Return the feature and the threshold split
    '''
    n_features      = X.shape[1]
    m               = Y.shape[0]
    smallest_score  = 100
    classes         = np.unique(Y)
    feature_idx     = 0
    
    for feat in range(n_features):
        print('feature #' + str(feat) )
        x = X[:, feat]
        current_score = Gini(tuple(Y), classes, m)
        a, b = zip(*sorted(zip(x, Y)))
        for obs in range(1, m):
            n_left  = obs
            n_right = m - obs
            
            
            if a[n_left-1] == a[n_left]:
                continue
            # get right and left datasets
            left_y, right_y = b[:n_left], b[n_left:]
            score_left = Gini(left_y, classes, n_left)
            score_right = Gini(right_y, classes, n_right)
            score = (n_left*score_left + n_right*score_right)/m
            
            # check whether the 
            if score <smallest_score:
                
                smallest_score   = score
                treshold        = (a[n_left-1]+a[n_left])/2
                feature_idx     = feat
        print('chosen feature is #'+str(feature_idx))
        print('with score: '+str(smallest_score))
        
    if current_score<=smallest_score:
        #select the majority class in leave as the classified value
        values, count = np.unique(Y, return_counts=True)
        ind = np.argmax(count)
        return DecisionTree(x=X, y = Y, classed = values[ind], score = current_score)
    else:
        print('Iteration Done')
        filt = X[:,feature_idx]<treshold
        left_keep_x, left_keep_y = X[filt,:], Y[filt]
        right_keep_x, right_keep_y = X[filt==False,:], Y[filt==False]
        
        left_branch = BuildTree(left_keep_x, left_keep_y)
        right_branch = BuildTree(right_keep_x, right_keep_y)
        return DecisionTree(feature = feature_idx, treshold = treshold, right = right_branch, left= left_branch, x = X, y = Y, score = smallest_score)
    
                
            
def Gini(y,classes, N):
    gini = 0
    for clas1 in classes:
        p1 = y.count(clas1)/N
        for clas2 in classes:
            if clas1!=clas2:
                p2 = y.count(clas2)/N 
                gini+=p1*p2
    return gini


def Predict(tree, x):
    '''Predict class of the example x'''
    
    
    if tree.classed != None:
        return tree.classed
    else:
        col = x[tree.feature]
        if col < tree.treshold:
            branch = tree.left
        else:
            branch = tree.right
    return Predict(branch,x)

        
        
     

tree = BuildTree(X, Y) 
i=1
Predict(tree, X[i,:])
subset
y=[]
y_hat = []
correct=[]
for i in range(len(Y)):
    if i not in subset:
        real = my_data_raw[i, -1]
        estimate = Predict(tree, my_data_raw[i,:-1])
        y.append(real)
        y_hat.append(estimate)
        correct.append(real==estimate)
sum(correct)/len(correct)
    
    
        
        
        