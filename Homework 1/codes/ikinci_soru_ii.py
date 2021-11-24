
# 2. Soru ii şıkkı

#Ali İbrahim Yılmaz-040170010
#Ahmet Özler Aktaş -040170104


import numpy as np                # numpy kütüphanesi import edildi
import matplotlib.pyplot as plt   # matplotlib kütüphanesi import edildi



counter=0;                        # Eğitim esnasında doğru çıktıları saymak için counter değiskeni olusturuldu.
                                  # İterasyon sınırına gelmeden tüm noktalar doğru sınıflandırılırsa döngüden çıkılacak    
test_counter=0;                   # Test sonuçlarının doğruluğunu saymak için değişken
iteration=50;                     # 50 iterasyon yapılacak
learning_rate = 1.0;              # learning rate 1 olarak belirlenmiştir.

cluster= [[0,-1,1], [0,0,1], [0,1,1], [1,-1,1], [1,0,1], [1,1,1], [-1,-1,1], [-1,0,1], [-1,1,1],
          [-3,3,-1], [-3,1,-1], [-3,0,-1], [-3,-1,-1], [-3,-3,-1], [-1,3,-1], [-1,-3,-1],
          [0,3,-1], [0,-3,-1], [1,3,-1], [1,-3,-1], [3,3,-1], [3,1,-1], [3,0,-1], [3,-1,-1],
          [3,-3,-1], [-2,3,-1], [-3,2,-1], [-3,-2,-1], [-2,-3,-1], [2,3,-1], [3,2,-1], [3,-2,-1], [2,-3,-1]]

# İlk şıkta önerdiğimiz fonksiyonları, ara katman birimleri olarak kullanırsak:
#fi(x1, x2)= [x1, x2, |x1|+|x2|] fonksiyonu

length=len(cluster)                   # kümedeki eleman sayısı
cluster_3D = np.zeros((length,4))     # kümenin 3 boyuta çıkarılmış hali için (33,4) boyutunda 0 matrisi. 4. boyut etiketler için ayrılmıştır.
cluster_a= np.array(cluster)          # cluster liste halinde. Onu arraye çevirdik.


def fi_function (x1, x2, i, label):   # fi fonksiyonu parametre olarak 4 değişken alır. 
                                      # i eleman sırasını, label de etiketleri belirtmek için
    cluster_3D[i] =  np.array([[x1], [x2], [abs(x1)+ abs(x2)], [label]]).reshape(4,)  # kümenin fi fonksiyonu ile 4 boyutlu hale getirilmesi
                                                                                      # reshape(4,) matrislerin uyumlu olması için kullanıldı.
    
def step_function(x):                       # Rosenblatt'ın genlikte ayrık algılayıcısında basamak fonksiyonu yok
                                            # Sınıflandırma işlemi (wi)*(fii)+w(m+1) sonucunun 0 dan büyük olup olmamasına göre sınıflandırma yapıyor.
                                            # Buna göre basamak fonksiyonu kullanarak etiket belirtme işlemi yapılabilir. 
    if(x<0): a=-1                           # x 0 dan küçük ise sonuç=-1
    else:  a=1                              # x 0 dan büyük ya da eşit ise sonuç=1
    return a                                # sonucu döndür    
    
    
    
for i in range(length):                     # kümedeki eleman sayısı kadar döngü devam edecek      
    fi_function( cluster_a[i][0], cluster_a[i][1],  i, cluster_a[i][2]) # Parametreler fi fonksiyonuna veriliyor ve 4 boyutlu yeni bir küme oluşturuluyor.
   
    
np.random.seed(40)                            # kod her çalıştırıldığında farklı değer vermemesi için seed fonksiyonu kullanıldı.
np.random.shuffle(cluster_3D)                 # cluster matrisindeki elemanların sırası rastgele değiştirildi
cluster_3D_t = np.transpose(cluster_3D)       # kümenin transpozu alındı
cluster_3D_wl = np.delete(cluster_3D_t, 3, 0) # cluster_3D_t etiketleri de içeriyor. Bu yüzden etiket sütunu çıkartılmış hali cluster_3D_wl değişkenine atandı.

bias = np.ones((1,33))                        # (1,33) boyutunda bias matrisi oluşturuldu
cluster_vector = np.append(cluster_3D_wl, bias, axis=0)   #cluster_3D_wl matrisine bias satırı eklendi.

np.random.seed(40)                   
w = np.random.randint(-2,3,(4,1))    # (4,1) boyutunda ilk ağırlık matrisi: [-2,2] aralığında

length=len(cluster_vector[0])        # cluster_vector matrisi, cluster_3D_wl kümesinin transpozu olduğu için sütun sayısı eleman sayısını verecek.
                                     # length değişkeni cluster_vector sayısının sütunu sayısı değerine eşitlendi
                                  
 
for i in range(iteration):                # iterasyon sayısı kadar döngü devam edecek
    counter=0                             # counter değişkeni her iterasyon başlangıcında sıfırlanmalı
    for j in range(length):               # cluster_vector deki eleman sayısı kadar döngü devam edecek
        
        wt = np.transpose(w)                 # her adımda ağırlık matrisi olan w matrisinin transpozu alınır
        xn=cluster_vector[:,j].reshape(4,1)  # cluster_vector matrisinde sıradaki eleman wt ile çarpılabilecek hale getirilir.
        v = np.dot(wt,xn)                    # matrix çarpımı v değişkenine atanır. 
        y = step_function(v)                 # s çıktısı basamak fonksiyonuna giriş olarak verildi       
        yd = cluster_3D_t[3][j]              # cluster_3D_t deki küme etiketini yd ye eşitliyoruz
        if(y==yd): counter+=1                # y ve yd birbirine eşit ise doğru sınıflandırma yapıldı. counter 1 arttırıldı.
        w = w + learning_rate*0.5*(yd-y)*xn  # weight in güncellenmesi
        
    if(counter==length):break        # bütün elemanlar doğru sınıflandırıldıysa döngüden çık 
   

     
wt = np.transpose(w)                 # en son güncellenmiş weight in transpozu wt ye eşitlendi

test_label = np.zeros((1,33))        # tabloda gerçek etiketler ve tahmin edilen etiketleri görsel olarak kıyaslamak için oluşturuldu.
 
for k in range(length):              # test kümesindeki bütün elemanlar için döngü
    
    xn=cluster_vector[:,k].reshape(4,1)   # cluster_vector matrisinde sıradaki eleman wt ile çarpılabilecek hale getirilir.
    v = np.dot(wt,xn)                     # matrix çarpımı v değişkenine atanır.
    y = step_function(v)
    yd = cluster_3D_t[3][k]            # cluster_3D_t deki küme etiketini yd ye eşitliyoruz
    if(y==yd) : test_counter+=1        # y ve yd birbirine eşit ise doğru sınıflandırma yapıldı. test_counter 1 arttırıldı.
    
    test_label[0][k]=y                 # y değerini test_label matrisindeki sıradaki elemana eşitliyoruz    
    
    
result = (test_counter/length)*100     # sonuç % değerinin hesaplanması
print("%{:.2f}".format(result))        # sonucun yazdırılması


cluster_labeled = np.append(cluster_3D_t, test_label, axis=0)   # gerçek etiketler ile modelimizin çıktıları tabloda alt alta gelecek şekilde birleştirildi


# kümedeki elemanların 3 boyutlu düzlemde çizdirilmesi

cluster_f = np.zeros((9,4))     # 1'ler kümesinin 9 adet elemanı için (9,4) boyutunda 0 matrisi oluşturuldu
cluster_s = np.zeros((24,4))    # -1'ler kümesinin 24 adet elemanı için (24,4) boyutunda 0 matrisi oluşturuldu
f,s=0,0                         # 1'ler ve -1'ler kümelerinde sıradaki elemanı tutmak için


for n in range(length):            # cluster kümesindeki elemanlar için döngü
    if(cluster_3D[n][3]==1):       # cluster kümesindeki sıradaki elemanın etiketi 1 ise 
        cluster_f[f]=cluster_3D[n] # sıradaki elemanı cluster_f matrisine ekle 
        f+=1                       # cluster_f matrisindeki sırayı 1 arttır.
   
    else:                          # cluster kümesindeki sıradaki elemanın etiketi -1 ise
       cluster_s[s]=cluster_3D[n]  # sıradaki elemanı cluster_s matrisine ekle 
       s+=1                        # cluster_s matrisindeki sırayı 1 arttır.
    
fig = plt.figure()                  # matplotlip kütüphanesi ile çizim yapmak için fonksiyonlar
ax = fig.add_subplot(111,projection='3d')   
     
ax.scatter(cluster_f[:,0], cluster_f[:,1], cluster_f[:,2], s = 10 , color = 'blue', label = "cluster 1",alpha=1  )     # 1'ler kümesini mavi renkte çizdir
ax.scatter(cluster_s[:,0], cluster_s[:,1], cluster_s[:,2], s = 10 , color = 'red', label = "cluster -1",alpha=1  )     # -1'ler kümesini kırmızı renkte çizdir

ax.set_xlabel('X Axis')   # eksenleri isimlendirme işlemi
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

ax.legend()               # hangi regin angi kümeyi belirttiğini göstermek için etiket bilgilendirmesi






