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
subset          = np.random.default_rng().choice(my_data_raw.shape[0], size = 2500, replace = False)
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
    
    
def BuildTree(X, Y, feat_keep = 5):
    '''
    Go through every feature to determine its best partition value
    Decide which is the purest partition
    Return the feature and the threshold split
    Arguments:
        X -- np.ndarray: features of the data
        Y -- np.ndarray: labels of the data
        feat_keep -- number of features to randomly select on which to grow each node
    '''
    n_features      = X.shape[1]
    m               = Y.shape[0]
    smallest_score  = 100
    classes         = np.unique(Y)
    feature_idx     = 0
    
    # select all features to operate the node partition
    if feat_keep == n_features:
        selected_features = list(range(n_features))
    # select only a fraction equal to share of the features (for random forest)
    else:
        selected_features=np.random.default_rng().choice(n_features, size=feat_keep, replace = False)
    # go through each feature
    for feat in selected_features:
        print('feature #' + str(feat) )
        x = X[:, feat]
        current_score = Gini(tuple(Y), classes, m)
        # order dataset by value of selected feature
        a, b = zip(*sorted(zip(x, Y)))
        # calculate evaluation function for each treshold
        for obs in range(1, m):
            n_left  = obs
            n_right = m - obs
            
            
            if a[n_left-1] == a[n_left]:
                continue
            # get right and left datasets
            left_y, right_y = b[:n_left], b[n_left:]
            # score on the first split of the data
            score_left = Gini(left_y, classes, n_left)
            # score on the second split of the data
            score_right = Gini(right_y, classes, n_right)
            # final score
            score = (n_left*score_left + n_right*score_right)/m
            
            # if score is smaller than the previous smaller score, update the feature and treshold
            if score <smallest_score:
                
                smallest_score   = score
                treshold        = (a[n_left-1]+a[n_left])/2
                feature_idx     = feat
        print('chosen feature is #'+str(feature_idx))
        print('with score: '+str(smallest_score))
    # when no partition is better than any partition, we have reached a leave    
    if current_score<=smallest_score:
        #select the majority class in leave as the classified value
        values, count = np.unique(Y, return_counts=True)
        ind = np.argmax(count)
        return DecisionTree(x=X, y = Y, classed = values[ind], score = current_score)
    # if not we need to keep growing the tree, both on the right and on the left
    else:
        print('Iteration Done')
        filt = X[:,feature_idx]<treshold
        left_keep_x, left_keep_y = X[filt,:], Y[filt]
        right_keep_x, right_keep_y = X[filt==False,:], Y[filt==False]
        
        left_branch = BuildTree(left_keep_x, left_keep_y, feat_keep)
        right_branch = BuildTree(right_keep_x, right_keep_y, feat_keep)
        return DecisionTree(feature = feature_idx, treshold = treshold, right = right_branch, left= left_branch, x = X, y = Y, score = smallest_score)
    
                
            
def Gini(y,classes, N):
    '''
    Calculate the Gini Index on labels y (np.ndarray),given the different 
    classes (np.ndarray) and number of observations N (int)'''
    gini = 0
    for clas1 in classes:
        p1 = y.count(clas1)/N
        for clas2 in classes:
            if clas1!=clas2:
                p2 = y.count(clas2)/N 
                gini+=p1*p2
    return gini


def Predict(tree, x):
    '''Predict class of the example x, using tree'''
    if tree.classed != None:
        return tree.classed
    else:
        col = x[tree.feature]
        if col < tree.treshold:
            branch = tree.left
        else:
            branch = tree.right
    return Predict(branch,x)


def RandomForest(X, Y, total_trees = 100, bootstrap_size = 500, m = None):
    '''
    Fits a Random Forest classifier to dataset X with labels y
    Returns:
        The forest of trees: a list where each element is one of the grown trees
    Arguments:
        X -- np.ndarray: features of the data
        Y -- np.ndarray: labels of the data
        total_trees -- number of trees to grow
        boostrap_size -- sample size of the bootstrap
        m -- number of randomly selected features on which to grow every node
    '''
    n_features = X.shape[1]
    forest = []
    for tree in range(total_trees):
        # get a bootstrap of the sample
        bootstrap = np.random.default_rng().choice(X.shape[0], size = bootstrap_size, replace = False)
        x = X[bootstrap,:]
        y = Y[bootstrap]
        if m == None:
            m= int(round(np.sqrt(n_features)))
        
        forest.append(BuildTree(X, Y, m))
    return forest
 
    
        
     

tree = BuildTree(X, Y, 57) 
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
    

forest = RandomForest(X, Y, total_trees = 100, bootstrap_size = 1500, m = 30)
# predict using the forest
y=[]
y_hat=[]
all_trees = []
correct=[]
for i in range(len(Y)):
    if i not in subset:
        real = my_data_raw[i, -1]
        all_predictions = []
        for classifier in range(len(forest)):
            tree = forest[classifier]
            estimate = Predict(tree, my_data_raw[i,:-1])
            all_predictions.append(estimate)
        values, count = np.unique(all_predictions, return_counts=True)    
        all_trees.append(all_predictions)
        estimate=values[np.argmax(count)]
        y.append(real)
        y_hat.append(estimate)
        correct.append(real==estimate)
sum(correct)/len(correct)

        