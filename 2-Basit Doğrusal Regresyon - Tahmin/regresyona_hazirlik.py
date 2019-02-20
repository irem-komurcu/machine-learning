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


#ÖZNİTELİK ÖLÇEKLEME
#STANDARDSCALER İLE STANDARTLAŞTIRMA
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


#VERİ ŞABLONU OLUŞTURMA


















