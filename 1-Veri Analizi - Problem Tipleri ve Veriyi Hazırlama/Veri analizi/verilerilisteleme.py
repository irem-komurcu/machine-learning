#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 18:57:33 2018

@author: irem
"""
#library
#this is best machine learning librarys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#codes

#data uploading
veriler=pd.read_csv('/home/irem/Makine_öğrenmesi/Bölüm1/veriler.csv')

#Data preprocessing 
boy=veriler[['boy']]
print(boy)


#eksik veriler

#eksik veriler için ortalama alarak, ortalama değeri girebiiriz. bunun için direkt eksik veriye değer atamaktansa bunu yapabiliriz.

#sklearn  sci-kit learn   ---> bilimsel olarak makine öğrenmesi için bir kütüphane
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)