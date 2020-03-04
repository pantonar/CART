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

labels      =
my_data     = genfromtxt('spambase.data', delimiter=',')
X           = my_data[-1]
Y           = genfromtxt('spambase.data', delimiter=',')

with open('spambase.data') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
