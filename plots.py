#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:14:25 2018

@author: hardikuppal
"""

import pandas

from matplotlib import cm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

#import matplotlib.pyplot as plt



data_set=pandas.read_csv('data_set.csv', delimiter=',',header=0)
data_set=data_set.replace(['pop','metal','disco','blues','reggae','classical','rock','hiphop','country','jazz'],[1,2,3,4,5,6,7,8,9,10])

#data_set=data_set.loc[data_set['genre'].isin([1,2,6,9])]

data_set_y=data_set.loc[:,'genre']
print((data_set.shape))
#print(data_set.genre)
pca = decomposition.PCA(n_components=3)
pca.fit(data_set)
data_set = pca.transform(data_set)

data_set=pandas.DataFrame(data=data_set[0:,0:]) 
print((data_set.shape))

data_set = pandas.concat([data_set.reset_index(drop=True), data_set_y.reset_index(drop=True)], axis=1)
print((data_set.shape))
X=data_set.iloc[:,0]
Y=data_set.iloc[:,1]
Z=data_set.iloc[:,2]
colors=['b', 'c', 'y']
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(X, Y, Z)
plt.show()