#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:37:55 2018

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

pca = decomposition.PCA(n_components=10)
pca.fit(data_set)
data_set = pca.transform(data_set)
#
data_set=pandas.DataFrame(data=data_set[0:,0:])  


data_set = pandas.concat([data_set.reset_index(drop=True), data_set_y.reset_index(drop=True)], axis=1)
print((data_set.shape))
print(sum(pca.explained_variance_ratio_))
########## PCA end
#print(data_set.genre)
train, test = train_test_split(data_set, test_size = 0.25,random_state=2,stratify=data_set.genre)

train_x=train.loc[:,train.columns != 'genre']
train_y=train.loc[:,'genre']
#
test_x=test.loc[:,train.columns != 'genre']
test_y=test.loc[:,'genre']

print("Training data size: {}".format(train.shape))
print("Test data size: {}".format(test.shape))


print("-------------------SVM-------------------")

###  save the model
#joblib.dump(model, 'model.pkl')
#print("Trained and saved the model to project folder successfully.")

#load the model
#svm = joblib.load('model.pkl')


svm=SVC(C=1000,gamma='auto',kernel='poly',degree=2)
svm.fit(train_x,train_y)
print("Training Score: {:.3f}".format(svm.score(train_x,train_y)))
print("Test score: {:.3f}".format(svm.score(test_x,test_y)))

#predicted_y=svm.predict(test_x)
#cnf_matrix=sklearn.metrics.confusion_matrix(test_y,predicted_y)
#print(cnf_matrix)
#perf_score(svm,test_x,test_y,GENRES)

print("-------------------KNN-------------------")

results_knn=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    results_knn.append(knn.score(test_x,test_y))
    
    
max_accuracy_knn=max(results_knn)
best_k=1+results_knn.index(max(results_knn))
print("Max Accuracy is {:.3f} on test dataset with {} neighbors.\n".format(max_accuracy_knn,best_k))

plt.plot(numpy.arange(1,21),results_knn)
plt.xlabel("n Neighbors")
plt.ylabel("Accuracy")


knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(train_x,train_y)
print("Training Score: {:.3f}".format(knn.score(train_x,train_y)))
print("Test score: {:.3f}".format(knn.score(test_x,test_y)))
perf_score(knn,test_x,test_y,GENRES)

print("-------------------Random forest-------------------")
results_forest=[]
for i in range(2,20):
    forest=RandomForestClassifier(random_state=1,n_estimators=i)
    forest.fit(train_x,train_y)
    results_forest.append(forest.score(test_x,test_y))
    
max_accuracy_forest=max(results_forest)
best_n_est=2+results_forest.index(max(results_forest))
print("Max Accuracy is {:.3f} on test dataset with {} estimators.\n".format(max_accuracy_forest,best_n_est))

forest=RandomForestClassifier(random_state=1,n_estimators=best_n_est)
forest.fit(train_x,train_y)
print("Training Score: {:.3f}".format(forest.score(train_x,train_y)))
print("Test score: {:.3f}".format(forest.score(test_x,test_y)))

plt.plot(numpy.arange(2,20),results_forest)
plt.xlabel("n Estimators")
plt.ylabel("Accuracy")
perf_score(forest,test_x,test_y,GENRES)

print("------------------- Neural Net-------------------")

neural=MLPClassifier(max_iter=1000,random_state=1,hidden_layer_sizes=[10,10])
neural.fit(train_x,train_y)
print("Training Score: {:.3f}".format(neural.score(train_x,train_y)))
print("Test score: {:.3f}".format(neural.score(test_x,test_y)))
perf_score(neural,test_x,test_y,GENRES)
