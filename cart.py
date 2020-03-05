# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:06:50 2020

@author: Module implementing the CART algorithm to grow a Tree
"""
import os
import csv
from numpy import genfromtxt

directory = r'C:\Users\Spare 3\Desktop\kaggle\CART'
directory=r'/Users/pablo/Desktop/python_scripts_random/CART'

os.chdir(directory)

f           = open('spambase.names')
labels      = f.readlines()[33:]
labels      = [i.split(':')[0] for i in labels]
my_data     = genfromtxt('spambase.data', delimiter=',')
Y           = my_data[:,my_data.shape[1]-1].reshape((my_data.shape[0],1))
X           = my_data[:,0:my_data.shape[1]-1]


class CART:
    
    def __init__(self):
        self.X = X
        self.Y = Y
        
    def best_split(self, X, Y):
        '''
        Go through every feature to determine its best partition value
        Decide which is the purest partition
        Return the feature and the threshold split
        '''
        n_features      = X.shape[1]
        m               = Y.shape[0]
        largest_score   = 0
        
        for feat in range(n_features):
            x = X[feat,:]
            a, b = zip(*sorted(zip(x, Y)))
            for obs in range(m):
                n_left  = obs + 1
                n_right = m + 1 - obs
                
                score_left = 
                score_right = 
                score = (n_left*score_left + n_right*score_right)/(n_left+n_rigth)
                
                
                # check whether the 
                if score > largest_score:
                    largest_score   = score
                    threshold       = x
                    feature_idx     = feat
                
        
        return feature_idx, threshold
                
            
        
        
        
    def fit(self, loss_func = 'entropy'):
        
        
        
        
        
        
        