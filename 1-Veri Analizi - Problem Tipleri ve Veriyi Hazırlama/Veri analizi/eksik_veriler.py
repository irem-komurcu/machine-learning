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
veriler=pd.read_csv('/home/irem/Makine_öğrenmesi/veriler.csv')


#Data preprocessing 
boy=veriler[['boy']]
print(boy)


#eksik veriler

#eksik veriler için ortalama alarak, ortalama değeri girebiiriz. bunun için direkt eksik veriye değer atamaktansa bunu yapabiliriz.

#sklearn  sci-kit learn   ---> bilimsel olarak makine öğrenmesi için bir kütüphane
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)

#burada belirttiğimiz imputer ile sözel ve boş kalan alanlarda sıkıntı çıkacaktır. yas ve diğer sayısal aralıklardaki boş veriler içi aralık belirterek aşağıdaki gibi kullanabiliriz.
Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)


ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)





























