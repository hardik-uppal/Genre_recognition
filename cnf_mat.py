#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 05:42:59 2018

@author: hardikuppal
"""
import  matplotlib.pyplot as plt
import numpy
import itertools
import sklearn

def confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def perf_score(model,dataset_x,dataset_y,GENRES):
    true_y=dataset_y
    true_x=dataset_x
    pred=model.predict(true_x)

    print("---------------PERFORMANCE ANALYSIS FOR THE MODEL----------------\n")

    print("Accuracy score: {}%".format(sklearn.metrics.accuracy_score(true_y,pred)*100))
    print("Classification score: \n{}".format(sklearn.metrics.classification_report(true_y,pred)))
    
    cnf_matrix=sklearn.metrics.confusion_matrix(true_y,pred)
    plt.figure()
    confusion_matrix(cnf_matrix,classes=GENRES,title='Confusion matrix')