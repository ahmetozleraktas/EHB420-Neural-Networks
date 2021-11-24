
# 1. Soru b şıkkı

#Ali İbrahim Yılmaz-040170010
#Ahmet Özler Aktaş -040170104



import numpy as np

counter=0;                        # Eğitim esnasında doğru çıktıları saymak için counter değiskeni olusturuldu.
test_counter=0;                   # Test sonuçlarının doğruluğunu saymak için değişken
iteration=50                      # 50 iterasyon yapılacak
learning_rate = 1.0;              # ogrenme hizi 1 olarak belirlendi



def step_function(x):                       # basamak fonksiyonunu
    if(x<0): a=-1                           # x 0 dan küçük ise sonuç=-1
    else:  a=1                              # x 0 dan büyük ya da eşit ise sonuç=1
    return a                                # sonucu döndür



np.random.seed(3)                               # kod her çalıştırıldığında farklı değer vermemesi için seed fonksiyonu kullanıldı.
w = np.random.randint(-2,3,(7,1))                # (7,1) boyutunda ilk ağırlık matrisi: [-2,2] aralığında

np.random.seed(4)
fp_train_data=np.random.randint([-3,-3,-3,-3,-3,-3,1,1],[4,4,4,4,4,4,2,2],(15,8))  # 1. eğitim kümesi için 15 adet 8 boyutlu [-3,3] değer aralığında noktalar üretildi.
                                                                                   # normalde (15,6) boyutunda olması gerekirdi
                                                                                   # Ancak 1 boyut bias için 1 boyut da etiket için eklendi
                                                                                   # Matris çarpımlarında etiket uzayı kullanılmayacak.
                                                                                   # Oluşturulan kümeler çok büyük olasılıkla lineer ayrıştırılabilir olmayacaktır.
                                                                                   # Bu sebeple kümelerin lineer ayrıştırılamaz olduğu varsayılacaktır.
np.random.seed(5)                                                                                   
sp_train_data=np.random.randint([-3,-3,-3,-3,-3,-3,1,-1],[4,4,4,4,4,4,2,0],(15,8)) # 2. eğitim kümesi için 15 adet 8 boyutlu [-3,3] değer aralığında noktalar üretildi.

train_data = np.concatenate((fp_train_data, sp_train_data)) # 1. ve 2. kümenin elemanları train_data değişkeninde birleştirildi.


np.random.seed(4)       
np.random.shuffle(train_data)                 # train_data matrisindeki elemanların sırası rastgele değiştirildi


length=len(train_data)                        # train_data daki eleman sayısı length değişkenine atandı
train_data_t = np.transpose(train_data)       # train_data nın transpozu alındı
train_data_wl = np.delete(train_data_t, 7, 0) # train_data_t etiketleri de içeriyor. Bu yüzden etiket sütunu çıkartılmış hali train_data_wl değişkenine atandı.


for i in range(iteration):                # iterasyon sayısı kadar döngü devam edecek
    counter=0                             # counter değişkeni her iterasyon başlangıcında sıfırlanmalı
    for j in range(length):               # train_data daki eleman sayısı kadar döngü devam edecek
        
        wt = np.transpose(w)                # her adımda ağırlık matrisi olan w matrisinin transpozu alınır
        xn=train_data_wl[:,j].reshape(7,1)  # train_data_wl matrisinde sıradaki eleman wt ile çarpılabilecek hale getirilir.
        v = np.dot(wt,xn)                   # matrix çarpımı v değişkenine atanır. 
        y = step_function(v)                # v çıktısı basamak fonksiyonuna giriş olarak verildi
            
        yd = train_data_t[7][j]             # train_data_t deki küme etiketini yd ye eşitliyoruz
        
        if(y==yd): counter+=1               # y ve yd birbirine eşit ise doğru sınıflandırma yapıldı. counter 1 arttırıldı.
  
        w = w + learning_rate*0.5*(yd-y)*xn # weight in güncellenmesi
          
    if(counter==30):break                   # bütün elemanlar doğru sınıflandırıldıysa iterasyon sayısını tamamlamadan döngüden çık 
            
        
           
np.random.seed(60)                          # np.random.seed(40) yaparsak test kümesini, eğitim kümesindeki değerlerin aynısını ile oluşturmuş oluruz.
fp_test_data=np.random.randint([-3,-3,-3,-3,-3,-3,1,1],[4,4,4,4,4,4,2,2],(10,8))   #1. test kümesini oluşturma

np.random.seed(80) 
sp_test_data=np.random.randint([-3,-3,-3,-3,-3,-3,1,-1],[4,4,4,4,4,4,2,0],(10,8))  # 2. test kümesini oluşturma 


test_data = np.concatenate((fp_test_data, sp_test_data))  #test kümesinin birleştirilmesi

np.random.seed(100)       
np.random.shuffle(test_data)                # test kümesindeki elemanlar rastgele sıralandı 
        
length=len(test_data)                       # test kümesindeki eleman sayısı length değişkenine atandı
test_data_t = np.transpose(test_data)       # test kümesinin transpozu alındı
test_data_wl = np.delete(test_data_t, 7, 0) # test_data_t etiketleri de içeriyor. Bu yüzden etiket sütunu çıkartılmış hali test_data_wl değişkenine atandı.

wt = np.transpose(w)          # en son güncellenmiş weight in transpozu wt ye eşitlendi

test_label = np.zeros((1,20))  # tablo üzerinde görsel kıyaslama yapmak için üretildi
                               # gerçek etiketler ile modelimizin test çıktılarını kıyaslamak için
 
for k in range(length):       # test kümesindeki bütün elemanlar için döngü
    
    xn=test_data_wl[:,k].reshape(7,1)  # test_data_wl değişkeni wt ile çarpılabilecek hale getirilir.
    v = np.dot(wt,xn)                  # matrix çarpımı v değişkenine atanır.
    y = step_function(v)               # v çıktısı basamak fonksiyonuna giriş olarak verildi
    yd = test_data_t[7][k]             # test_data_t deki küme etiketini yd ye eşitliyoruz
    if(y==yd) : test_counter+=1        # y ve yd birbirine eşit ise doğru sınıflandırma yapıldı. test_counter 1 arttırıldı.
   
    test_label[0][k]=y                 # y değerini test_label matrisindeki sıradaki elemana eşitliyoruz
    
result = (test_counter/length)*100     # sonuç % değerinin hesaplanması
print("%{:.2f}".format(result))        # sonucun yazdırılması
        
     
cluster_labeled = np.append(test_data_t, test_label, axis=0)    # gerçek etiketler ile modelimizin çıktıları tabloda alt alta gelecek şekilde birleştirildi        
        
        
        
        
        
        
        
        
        
        
