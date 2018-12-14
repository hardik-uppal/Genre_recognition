#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 23:18:51 2018

@author: hardikuppal
"""

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
import sklearn
from sklearn import decomposition
from cnf_mat import perf_score

#import matplotlib.pyplot as plt



data_set=pandas.read_csv('data_set.csv', delimiter=',',header=0)
data_set=data_set.replace(['pop','metal','disco','blues','reggae','classical','rock','hiphop','country','jazz'],[1,2,3,4,5,6,7,8,9,10])
GENRES=['pop','metal','disco','blues','reggae','classical','rock','hiphop','country','jazz']
#GENRES=['pop','metal','classical','country']
#data_set=data_set.loc[data_set['genre'].isin([1,2,6,9])]
####   PCA
data_set_y=data_set.loc[:,'genre']
print((data_set.shape))
result_svm=[]
for i in range(1,10):
    pca = decomposition.PCA(n_components=i)
    pca.fit(data_set)
    data_set = pca.transform(data_set)
    #
    data_set=pandas.DataFrame(data=data_set[0:,0:])  # 1st row as the column names
    
    
    data_set = pandas.concat([data_set.reset_index(drop=True), data_set_y.reset_index(drop=True)], axis=1)
#    print((data_set.shape))
    print(sum(pca.explained_variance_ratio_))
    ########## PCA end
    #print(data_set.genre)
    train, test = train_test_split(data_set, test_size = 0.25,random_state=2,stratify=data_set.genre)
    
    train_x=train.loc[:,train.columns != 'genre']
    train_y=train.loc[:,'genre']
    #
    test_x=test.loc[:,train.columns != 'genre']
    test_y=test.loc[:,'genre']
    
#    print("Training data size: {}".format(train.shape))
#    print("Test data size: {}".format(test.shape))
    
    
#    print("-------------------SVM-------------------")
    
    ###  save the model
    #joblib.dump(model, 'model.pkl')
    #print("Trained and saved the model to project folder successfully.")
    
    #load the model
    #svm = joblib.load('model.pkl')
    
    
    svm=SVC(C=1000,gamma='auto',kernel='poly',degree=2)
    svm.fit(train_x,train_y)
#    print("Training Score: {:.3f}".format(svm.score(train_x,train_y)))
    print("Test score: {:.3f} for features:{}".format(svm.score(test_x,test_y),i))
    result_svm.append(svm.score(test_x,test_y))
    
print(result_svm)
#plt.plot(numpy.arange(1,49),result_svm)