
# 4. Soru 

#Ali İbrahim Yılmaz-040170010
#Ahmet Özler Aktaş -040170104



import numpy as np                 #numpy kütüphanesi çağırıldı
import math                        #math kütüphanesi euler sayısını çağırmak için çağırıldı
from scipy.misc import derivative  #sigmoid fonksiyonunun türevini almak için derivative kütüphanesi çağırıldı

error_sum = 0                       # her bir iterasyondaki hataların toplamını tutmak için değişken 
iteration = 200;                    # 200 iterasyon yapılacak
step_size = 0.49;                   # adım genişliği [0,1] aralığında bir sayı olmalı                 
test_counter = 0;                   # Test sonuçlarının doğruluğunu saymak için değişken


def f_function(x1,x2,x3):                       # Yaklaşık olarak ifade edilmek istenilen fonksiyon tanımlanıyor.     
    return (0.5*x1*x2+x2**2*math.e**(-x3))/1.5  # fonksiyonun normalize edilmiş değerleri döndürülecek.


def sigmoid(x):                     # 0.5*x1*x2+x2**2*math.e**(-x3) fonksiyonu bazı istisnalar dışında [0,1] aralığında gelir.
                                    # Bu sebeple tanh yerine lojistik fonksiyon kullanılmıştır.
  return 1 / (1 + math.e**(-x))     # lojistik fonksiyon


np.random.seed(50)                   
w = (np.random.random((4,1)))*2-1     # (4,1) boyutunda ilk ağırlık matrisi: [-1,1] aralığında
                                      # sigmoid(v) fonksiyonu ile iyi sonuç alabilmek için v değerlerinin [-1,1] aralığında olması gerekir.
                                      # v lerin [-1,1] aralığında olması için x ler [0,1] aralığında iken w matrisinde negatif eleman bulunmalı.
                                      
np.random.seed(90)                 
data=np.random.random((100,5))      # Eğitim ve test kümesi için 100 adet 5 boyutlu (0 ve 1 değer aralığında) noktalar üretildi.
                                    # Normalde (100,3) boyutunda olması gerekirdi
                                    # Ancak 1 boyut bias için 1 boyut da fonksiyonun gerçek sonucunu listelemek için eklendi
                                    # Matris çarpımlarında gerçek sonuç uzayı kullanılmayacak.  
                                    
bias = np.ones((100,1))             # (100,1) boyutunda bias matrisi oluşturuldu    

data[:,3]=bias[:,0]                 # data matrisinin 3. sütunu [0,1] aralığındaki değerlere sahip.
                                    # 3. sütunun tamamını 1 yapmak için bu eşitlik kullanıldı. 
                                          
length=len(data)                    # data matrisinin eleman sayısı length değişkenine atandı.

for i in range(length):             # data matrisindeki her elamanın ilk 3 sütun değeri, fonksiyonda yerine yazılmak için ayrıldı
    x1=data[i][0]                         # i ninci elemanın 0. sütunundaki değer x1 değişkenine atandı
    x2=data[i][1]                         # i ninci elemanın 1. sütunundaki değer x2 değişkenine atandı
    x3=data[i][2]                         # i ninci elemanın 2. sütunundaki değer x3 değişkenine atandı
    data[i][4]=f_function(x1,x2,x3)       # 4. sütun, fonksiyonun gerçek değerlerini belirtmek içindi.  # fonksiyonun sonucu data matrisinin her elemanının 4. sütununa yazıldı                                           
    
    #data[i][4]=(data[i][4]-np.min(data[:,4]))/ (np.max(data[:,4])-np.min(data[:,4])) # alternatif normalizasyon işlemi

test_data=np.zeros((40,5))                   # test kümesi 40 elemandan oluşacak 
train_data=np.zeros((60,5))                  # eğitim kümesi 60 elemandan oluşacak 

test_data=data[60:100,:]                     # data matrisinin son 40 elemanı test kümesi için ayrıldı
train_data= np.delete(data,slice(60,100), 0) # data matrisinin son 40 elemanının çıkartılmış hali ile eğitim kümesi oluşturuldu.

train_data_t = np.transpose(train_data)       # train_data nın transpozu alındı
train_data_wl = np.delete(train_data_t, 4, 0) # train_data_t gerçek sonuç sütununu da içeriyor. Bu yüzden gerçek sonuç sütununun çıkartılmış hali train_data_wl değişkenine atandı.
length=len(train_data)                        # train_data kümesinin eleman sayısı length değişkenine atandı.

e = np.zeros((length,1))                      # hata değerlerini tutmak için e matrisi oluşturuldu.
min_error_mean = 1000                         # minimum hata ortlamasına ilk değer olarak 1000 atandı.
best_w = w                                    # best_w, w'nin ilk değerine eşitlenir
statement_counter = 0;                        # döngü içerisinde best_w 'nin kaç kere değiştiğini saymak için


for i in range(iteration):                    # iterasyon sayısı kadar döngü devam edecek
    error_sum = 0                             # Hataların toplamı her iterasyonda sıfırlanıyor.
    
    for j in range(length):                   # train_data daki eleman sayısı kadar döngü devam edecek
        
        wt = np.transpose(w)                  # her adımda ağırlık matrisi olan w matrisinin transpozu alınır
        xn=train_data_wl[:,j].reshape(4,1)    # train_data_wl matrisindeki sıradaki eleman, wt ile çarpılabilecek hale getirilir.
        v = np.dot(wt,xn)                     # matrix çarpımı v değişkenine atanır. 
        y = sigmoid(v)                        # v değeri sigmoid fonksiyonuna parametre olarak verilir. Sonuç y'ye eşitlenir.
        yd = train_data[j][4]                 # train_data daki gerçek sonucu yd ye eşitliyoruz
        e[j]=yd-y                             # yd-y hatası e[j]'ye atandı.
        E=0.5*(e[j]**2)                       # epsilon (e[j]), hata fonksiyonunda yerine yazıldı
        error_sum += E                        # hataların toplanması
    
        w = w + step_size*e[j]*derivative(sigmoid, v)*xn     # weight in güncellenmesi
    
    mean = error_sum/length                   # hataların ortalaması alınıyor.
    
    if(mean < min_error_mean):                # i ninci iterasyondaki hataların ortalaması min_error_mean den daha küçük olup olmadığı test edilir.
        min_error_mean = mean                 # i ninci iterasyondaki hataların ortalaması min_error_mean den daha küçükse, min_error_mean güncellenir.
        best_w = w                            # daha az hata ortalamasına sahip olan ağırlık best_w ye eşitlenir.
        statement_counter+=1                  # bu ifadeye kaç kez girildiğini saymak için
        i_holder = i                          # en son kaçıncı iterasyonda bu ifadeye girildiğine bakmak için
    
    if(mean<0.00025): break;                  # hataların ortalamaso 0.0003 den küçükse eğitimi bitir.


test_data_t = np.transpose(test_data)         # test_data nın transpozu alındı
test_data_wl = np.delete(test_data_t, 4, 0)   # test_data_t gerçek sonuç sütununu da içeriyor. Bu yüzden gerçek sonuç sütununun çıkartılmış hali test_data_wl değişkenine atandı.


length=len(test_data)                         # test_data kümesinin eleman sayısı length değişkenine atandı.
wt = np.transpose(best_w)                     # en son güncellenen weightin transpozu alındı
cluster_result=np.zeros((length,1))           # tabloda görsel olarak kıyaslama yapmak için bir vektör oluşturuldu. Bu vektör adaline çıkışındaki y değerlerini tutacak. Daha sonra yd ile kıyas yapmamızı sağlayacak.
cluster_check=np.zeros((length,1))            # tabloya, istenilen değere yakın değerler elde edildiğinde 1 değeri, elde edilemediğinde 0 değeri yazan vektör eklenir.

error_sum_t = 0                               # test kümesinin hatalar toplamı için değişken


for j in range(length):                       # test kümesindeki elemanlar için döngü
    
    xn=test_data_wl[:,j].reshape(4,1)         # xn test kümesindeki sıradaki elemanı tutuyor.
    v = np.dot(wt,xn)                         # matrix çarpımı v değişkenine atanır.
    y = sigmoid(v)                            # v değeri sigmoid fonksiyonuna parametre olarak verilir. Sonuç y'ye eşitlenir.
    yd = test_data[j][4]                      # test_data daki gerçek sonucunu yd ye eşitliyoruz
    e[j]=yd-y                                 # yd-y farkı e[j]'ye atandı.
    E_t=0.5*(e[j]**2)                         # epsilon (e[j]), hata fonksiyonunda yerine yazıldı. E_t test kümesi hataları için değişken
    error_sum_t += E_t                        # hatalar toplanıyor
    
    tolerance = yd*0.1                        # gerçek sonucun +-%10 toleranslı hali doğru kabul edilecek
                                              
    
    if((yd<0 and (yd+tolerance<=y<=yd-tolerance)) or (yd>=0 and (yd-tolerance<=y<=yd+tolerance))):    # negatif ve pozifir sayılar için farklı tolerans aralıkları  
        test_counter+=1                   # fonksiyon sonuçlarına istenilen yakınlıkta değer üretildi ise test_counter 1 artar
        cluster_check[j]=1                # istenilen aralıkta çıktı alındı ise cluster_check[j]=1 olur.
   
    cluster_result[j]=y                   # y çıktısı, yd gerçek fonksiyon sonucunun yanına eklenmek için kaydedilir.

mean_t = error_sum_t/length            # test kümesindeki hataların ortalaması                    
    
result = (test_counter/length)*100     # sonuç % değerinin hesaplanması
print("%{:.2f}".format(result))        # sonucun yazdırılması


cluster_labeled = np.append(test_data, cluster_result, axis=1)          # cluster_labeled matrisine cluster_result sütunu eklendi
cluster_labeled = np.append(cluster_labeled, cluster_check, axis=1)     # cluster_labeled matrisine cluster_check sütunu eklendi

# cluster_labeled matrisindeki 4, 5 ve 6. sütunlar görsel olarak kontrol için
 


















