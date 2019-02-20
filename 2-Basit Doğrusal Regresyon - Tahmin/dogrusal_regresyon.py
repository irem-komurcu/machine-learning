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


#VERİ ÖN İŞLEME
#codes

#data uploading
veriler=pd.read_csv('/home/irem/Makine_öğrenmesi/Bölüm2/satislar.csv')


#Data preprocessing 
aylar=veriler[['Aylar']]
print(aylar)

satislar=veriler[['Satislar']]
print(satislar)

satislar2=veriler.iloc[:,:1].values
print(satislar2)


#veri kümesini test ve eğitim olarak bölme
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test= train_test_split(aylar,satislar, test_size=0.33, random_state=0 )

"""
#ÖZNİTELİK ÖLÇEKLEME
#STANDARDSCALER İLE STANDARTLAŞTIRMA
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
"""

#model inşası (linear regression)
#ASLINDA BURDA X_TRAINDEN Y_TRAIN İ ÖĞRENMESİNİ İSTEMİŞTİK.
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin=lr.predict(x_test)
#tahmini yaptıktan sonra, hiç bir yerde girdi larak vermediğim Y_traine bakıp karşılaştırıyorum


x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("aylara göre satış")
plt.xlabel("aylar")
plt.ylabel("satışlar")
















