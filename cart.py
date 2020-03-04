# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:06:50 2020

@author: Module implementing the CART algorithm to grow a Tree on dataset
"""
import os
import csv
from numpy import genfromtxt


directory = r'C:\Users\Spare 3\Desktop\kaggle\CART'
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
        
    def train(self, loss_func = 'entropy'):
        