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
veriler=pd.read_csv('/home/irem/Makine_öğrenmesi/Bölüm1//veriler.csv')


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


#burada stiring verilerimi sayısal verilere dönüştürüyoruz.
#yaptığımız şey 0,1,2 şeklinde id lendirmek.
#bunun zararlı bir yönü var ama.1-2-3 gibi sayılar birbiri ile işleme girebilir. çarpma bölme toplama gibi...

#encoder: Kategorik -> Numeric
ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)
#burada da string verileri id lendiriorum ama bu sefer tr:001, fr:100, us: 010 şeklinde. Bu yöntem daha işe yarar
#Her bir değeri kolon başlığına çeviriyor. varsa 1 yoksa şeklinde 001,100,010 şekline dönüşüyor. Mantık bu
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#Cinsiyet için ENCODER
c=veriler.iloc[:,-1:].values
print(c)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
c[:,0]=le.fit_transform(c[:,0])
print(c)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)



#obje oluşturdum
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas, index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1:].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])
print(sonuc3)


#axis diyerek kolon kolon değil satır satır birleştir dedim. Ortak olan satırları da atla dedim.
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)


#veri kümesini test ve eğitim olarak bölme
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test= train_test_split(s,sonuc3, test_size=0.33, random_state=0 )


#ÖZNİTELİK ÖLÇEKLEME
#STANDARDSCALER İLE STANDARTLAŞTIRMA
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


#VERİ ŞABLONU OLUŞTURMA


















