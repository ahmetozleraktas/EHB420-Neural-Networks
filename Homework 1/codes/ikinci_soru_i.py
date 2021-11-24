
# 2. Soru i şıkkı

#Ali İbrahim Yılmaz-040170010
#Ahmet Özler Aktaş -040170104


import numpy as np                #numpy kütüphanesi import edildi
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


def step_function(x):
    if(x<0): a=-1                           # x 0 dan küçük ise sonuç=-1
    else:  a=1                              # x 0 dan büyük ya da eşit ise sonuç=1
    return a                                # sonucu döndür


np.random.seed(40)                      # kod her çalıştırıldığında farklı değer vermemesi için seed fonksiyonu kullanıldı.
np.random.shuffle(cluster)              # cluster matrisindeki elemanların sırası rastgele değiştirildi
cluster_t = np.transpose(cluster)       # kümenin transpozu alındı
cluster_wl = np.delete(cluster_t, 2, 0) # cluster_t etiketleri de içeriyor. Bu yüzden etiket sütunu çıkartılmış hali cluster_wl değişkenine atandı.

np.random.seed(40)                  
w = np.random.randint(-2,3,(3,1))    # (3,1) boyutunda ilk ağırlık matrisi: [-2,2] aralığında
bias = np.ones((1,33))               # (1,33) boyutunda bias matrisi oluşturuldu

cluster_vector = np.append(cluster_wl, bias, axis=0)   #cluster_wl matrisine bias satırı eklendi.

length=len(cluster_vector[0])        # cluster_vector matrisi, cluster kümesinin transpozu olduğu için sütun sayısı eleman sayısını verecek.
                                     # length değişkeni cluster_vector sayısının sütun sayısı değerine eşitlendi
                            
 
for i in range(iteration):                # iterasyon sayısı kadar döngü devam edecek
    counter=0                             # counter değişkeni her iterasyon başlangıcında sıfırlanmalı
    for j in range(length):               # cluster_vector deki eleman sayısı kadar döngü devam edecek
        
        wt = np.transpose(w)                # her adımda ağırlık matrisi olan w matrisinin transpozu alınır
        xn=cluster_vector[:,j].reshape(3,1) # cluster_vector matrisinde sıradaki eleman wt ile çarpılabilecek hale getirilir.
        v = np.dot(wt,xn)                   # matrix çarpımı v değişkenine atanır. 
        y = step_function(v)                # v çıktısı basamak fonksiyonuna giriş olarak verildi     
        yd = cluster_t[2][j]                # cluster_t deki küme etiketini yd ye eşitliyoruz
        if(y==yd): counter+=1               # y ve yd birbirine eşit ise doğru sınıflandırma yapıldı. counter 1 arttırıldı.
            
        w = w + learning_rate*0.5*(yd-y)*xn # weight in güncellenmesi
        
    if(counter==length):break               # bütün elemanlar doğru sınıflandırıldıysa döngüden çık 
        
        
        
wt = np.transpose(w)           # en son güncellenmiş weight in transpozu wt ye eşitlendi

test_label = np.zeros((1,33))  # tabloda gerçek etiketler ve tahmin edilen etiketleri görsel olarak kıyaslamak için oluşturuldu.
 

for k in range(length):        # test kümesindeki bütün elemanlar için döngü
    
    xn=cluster_vector[:,k].reshape(3,1)   # cluster_vector matrisinde sıradaki eleman wt ile çarpılabilecek hale getirilir.
    v = np.dot(wt,xn)                     # matrix çarpımı v değişkenine atanır.
    y = step_function(v)                  # v çıktısı basamak fonksiyonuna giriş olarak verildi
    yd = cluster_t[2][k]               # cluster_t deki küme etiketini yd ye eşitliyoruz
    if(y==yd) : test_counter+=1        # y ve yd birbirine eşit ise doğru sınıflandırma yapıldı. test_counter 1 arttırıldı.
    
    test_label[0][k]=y                 # y değerini test_label matrisindeki sıradaki elemana eşitliyoruz    
    
result = (test_counter/length)*100     # sonuç % değerinin hesaplanması
print("%{:.2f}".format(result))        # sonucun yazdırılması

cluster_labeled = np.append(cluster_t, test_label, axis=0)   # gerçek etiketler ile modelimizin çıktıları tabloda alt alta gelecek şekilde birleştirildi
 

# kümedeki elemanların 2 boyutlu düzlemde çizdirilmesi

cluster_f = np.zeros((9,3))   # 1'ler kümesinin 9 adet elemanı için (9,3) boyutunda 0 matrisi oluşturuldu
cluster_s = np.zeros((24,3))  # -1'ler kümesinin 24 adet elemanı için (24,3) boyutunda 0 matrisi oluşturuldu
f,s = 0,0                     # 1'ler ve -1'ler kümelerinde sıradaki elemanı tutmak için


for n in range(length):           # cluster kümesindeki elemanlar için döngü

    if(cluster[n][2]==1):         # cluster kümesindeki sıradaki elemanın etiketi 1 ise 
        cluster_f[f]=cluster[n]   # sıradaki elemanı cluster_f matrisine ekle 
        f+=1                      # cluster_f matrisindeki sırayı 1 arttır.
   
    else:                         # cluster kümesindeki sıradaki elemanın etiketi -1 ise
       cluster_s[s]=cluster[n]    # sıradaki elemanı cluster_s matrisine ekle 
       s+=1                       # cluster_s matrisindeki sırayı 1 arttır.
      
        
fig = plt.figure()                # matplotlip kütüphanesi ile çizim yapmak için fonksiyonlar
ax = fig.add_subplot()            
      
plt.scatter(cluster_f[:,0], cluster_f[:,1], color='blue', label = "cluster 1"   ) # 1'ler kümesini mavi renkte çizdir
plt.scatter(cluster_s[:,0], cluster_s[:,1], color='red', label = "cluster -1"   ) # -1'ler kümesini kırmızı renkte çizdir

ax.set_xlabel('X Axis')    # eksenleri isimlendirme işlemi
ax.set_ylabel('Y Axis')

ax.legend(loc=(1,1))       # hangi regin angi kümeyi belirttiğini göstermek için sağ üst kısımda etiket bilgilendirmesi


 
