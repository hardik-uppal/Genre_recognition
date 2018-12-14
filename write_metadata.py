#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:02:06 2018
Writing the metadata
@author: hardikuppal
"""
import csv
import os
import numpy as np



label=[]
filename=[]
for root, dirs, files in os.walk("../Genre_recognition/genres"):  
    
#extarct filename and class/genre 
#for files in os.walk("../Genre_recognition/genres/\%s",label_tmp):- walk through the directory
    for filename_tmp in files:
        filename.append((filename_tmp,filename_tmp[0:filename_tmp.find('.')]))
        




myFile = open('meta_data.csv', 'w')
with myFile:  
   writer = csv.writer(myFile)
   writer.writerows(filename)

