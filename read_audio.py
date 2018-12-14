#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:00:19 2018

@author: hardikuppal
"""
import pandas as pd
import librosa 
import numpy
import sklearn
from extract_features import extract_features



dataset_path = "../Genre_recognition/genres"
project_path="../Genre_recognition"
metadata = pd.read_csv("%s/meta_data.csv" % project_path, sep=",", header=0)
metadata.columns = ["filename", "label"]
audio_sample=[]
# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_size = 512
frame_size = 1028

is_created = False
labels = []
print("Working for extarction")
for i in range(0,len(metadata.index)):
    
    y, sr = librosa.load("%s/%s/%s" % (dataset_path,metadata.iloc[i].label,metadata.iloc[i].filename ), mono=True)
    #audio_sample.append((y,sr))
    row = extract_features(y, sr, frame_size, hop_size)
    if not is_created:
        dataset_numpy = numpy.array(row)
        is_created = True
    elif is_created:
        dataset_numpy = numpy.vstack((dataset_numpy, row))

    labels.append(metadata.iloc[i].label)
    
    print("#", end ="")
print("100%")
print("Normalizing the data...")

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
dataset_numpy = scaler.fit_transform(dataset_numpy)

Feature_Names = ['meanZCR', 'stdZCR', 'meanSpecCentroid', 'stdSpecCentroid', 'meanSpecContrast', 'stdSpecContrast',
'meanSpecBandwidth', 'stdSpecBandwidth', 'meanSpecRollof', 'stdSpecRollof',
'meanMFCC_1', 'stdMFCC_1', 'meanMFCC_2', 'stdMFCC_2', 'meanMFCC_3', 'stdMFCC_3',
'meanMFCC_4', 'stdMFCC_4', 'meanMFCC_5', 'stdMFCC_5', 'meanMFCC_6', 'stdMFCC_6',
'meanMFCC_7', 'stdMFCC_7', 'meanMFCC_8', 'stdMFCC_8', 'meanMFCC_9', 'stdMFCC_9',
'meanMFCC_10', 'stdMFCC_10', 'meanMFCC_11', 'stdMFCC_11', 'meanMFCC_12', 'stdMFCC_12','meanMFCC_13', 'stdMFCC_13','meanMFCC_14', 'stdMFCC_14'
,'meanMFCC_15', 'stdMFCC_15','meanMFCC_16', 'stdMFCC_16','meanMFCC_17', 'stdMFCC_17','meanMFCC_18', 'stdMFCC_18','meanMFCC_19', 'stdMFCC_19']

dataset_pandas = pd.DataFrame(dataset_numpy, columns=Feature_Names)
dataset_pandas["genre"] = labels
dataset_pandas.to_csv("data_set.csv", index=False)
print("Data set has been created")

