# -*- coding: utf-8 -*-
# 2. Ödev 1. soru 2. kısım IRIS DATASET

#Ali İbrahim Yılmaz-040170010
#Ahmet Özler Aktaş -040170104


import numpy as np                 #numpy kütüphanesi çağırıldı
import math                        #math kütüphanesi euler sayısını çağırmak için çağırıldı
from scipy.misc import derivative  #sigmoid fonksiyonunun türevini almak için derivative kütüphanesi çağırıldı

iteration = 200;                    # 200 iterasyon yapılacak
learning_rate = 0.49;               # öğrenme oranı [0,1) aralığında bir sayı olmalı
error_sum = 0                       # her bir iterasyondaki hataların toplamını tutmak için değişken

dataset_matrix = [] # iris.data dosyasının içindeki bilgilerin atanacağı boş matrisimiz.


with open('iris.data') as f:  # Data dosyası open fonksiyonu vasıtasıyla açıldı.
    for line in f:   # for döngüsü sayesinde her satırda gezildi ve ona göre işlem yapıldı.
        row = line.split(',') # satırdaki bilgiler virgülle ayrılmasına göre ayrıldı ve ayrı eleman olarak alındı.
        
        for a in range(4):  # iris.data dosyasından okunan her satırın 1,2,3 ve 4. sütunlarını stringden float sayı yapısına çevirmek gerekmekte.
            row[a] = float(row[a])
            row[a] = row[a]/8  # datasetimizdeki max değer 7.9 olduğundan normalizasyon değerini 8 olarak seçtik ve değerleri (0,1) aralığına çektik.
            
        if (row[4] == 'Iris-setosa') or (row[4] == 'Iris-setosa\n'):  # 3 çeşit çiçek bulunmakta. Oluşturulan matirisin son satırına [0,2] arasındaki tam sayı değerlerini atayarak hangi türden çiçek olduğu anlaşılacak. Mesela 0 için Iris-setosa.
           row[4] = 0
           
        if (row[4] == 'Iris-versicolor') or (row[4] == 'Iris-versicolor\n'):
            row[4] = 1
            
        if (row[4] == 'Iris-virginica') or (row[4] == 'Iris-virginica\n'):
           row[4] = 2
        
           
        dataset_matrix.append(row) # çiçek türünü string versiyondan tam sayıya ve sayı değerlerini stringden ondalıklı sayıya çevirdikten sonra,
                                   # son olarak başta oluşturduğumuz boş matrise elde ettiğimiz satırımızı ekliyoruz ve dataset matrisimiz hazır. 
np.random.seed(50)                   
w = np.random.random((5,1)) #ilk ağırlık matrisi oluşturuldu.

train_matrix = []

train_data1 = dataset_matrix[0:28:1] # datasetteki 3 çeşit bitki için ilk 28 veri eğitim kümesi olarak ayrıldı.
train_data2 = dataset_matrix[50:78:1]
train_data3 = dataset_matrix[100:128:1]

for v in train_data3:  # 28'er veriden oluşan eğitim kümesi tek bir matriste birleştirildi
    train_matrix.append(v)
for v in train_data1:
    train_matrix.append(v)
for v in train_data2:
    train_matrix.append(v)
    
np.random.shuffle(train_matrix) # türlerin kendi aralarında matriste sıralı olmasından dolayı eğitim kümesindeki tüm satırlar rastgele karıştırıldı.

def sigmoid(x):                     
  return 1 / (1 + math.e**(-x))     # lojistik fonksiyon

