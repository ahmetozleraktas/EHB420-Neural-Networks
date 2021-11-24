
# 1. Soru a şıkkı

#Ali İbrahim Yılmaz-040170010
#Ahmet Özler Aktaş -040170104


import numpy as np                 #numpy kütüphanesi import edildi
import matplotlib.pyplot as plt    #matplotlib kütüphanesi import edildi
from scipy.misc import derivative  #sigmoid fonksiyonunun türevini almak için derivative kütüphanesi çağırıldı

 
iteration = 200;                    # 200 iterasyon yapılacak
learning_rate = 0.49;               # öğrenme oranı [0,1) aralığında bir sayı olmalı                 
class_num = 4;                      # 4 adet sınıf var
momentum=0.1;                       # momentum katsayısı 0.1 olarak belirlendi


def sigmoid(x):                     # 5x10 boyutundaki örüntüler [0,1] aralığındaki değerlerden oluşmaktadır.
                                    # Bu sebeple sigmoid fonksiyonu olarak lojistik fonksiyon kullanılmıştır.    
    return 1/(1 + 2.718**(-x))      # lojistik fonksiyon


image = np.ones((5,10,4))           # 4 sınıfa ait 5x10 boyutundaki örüntü matrisi

# 4 farklı sınıf bulunmaktadır. Bu sınıflar 4 farklı Arapça harften oluşmaktadır. Raporda harflerin şekilleri gösterilmektedir. 

# 5x10 boyutundaki 4 sınıf aşağıdaki gibidir.

image[:,:,0]=[[1,1,1,1,1,1,1,1,0,1],[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0,0],[0,1,1,1,1,1,1,0,1,0],[0,0,0,0,0,0,0,0,0,0]]
image[:,:,1]=[[1,1,1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,0,1],[0,0,0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,0,1,1,1,1,1]]
image[:,:,2]=[[1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,0,1,1,0],[0,1,1,1,0,0,0,0,0,0],[0,1,1,1,0,1,1,1,1,1],[0,0,0,0,0,1,1,1,1,1]]
image[:,:,3]=[[1,1,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1,0],[1,1,1,1,1,0,1,1,1,0],[1,1,1,1,1,0,0,0,0,0]]


data_gen1=18     # Her sınıf için 18 adet örüntü oluşturulacak. Oluşturulan bu örüntüler gürültü işleminden geçirilecek. 
                 # Bunlardan 3er tanesi validation, 6şar tanesi test, 9ar tanesi train setine ayrıştırılacak

noisy = np.zeros((5,10,class_num,data_gen1)) # (5,10,4,18) boyutunda gürültü işleminden geçirilmiş elemanları tutacak tensör

rn=1  # random.seed() fonksiyonunun içine koymak için değişken

for i in range(class_num):
    for j in range(data_gen1):
        np.random.seed(rn)           # noise işareti rastgele oluşturulacak. Kod her çalıştırıldığında farklı değerler üretmemesi için
                                     # np.random.seed() fonksiyonu kullanılmıştır. seed() fonksiyonunun içerisine sabit bir değer yazarsak
                                     # for döngüsü boyunca hep aynı rastgele değeri üretir. Bunun için seed() fonksiyonun içerisine düzenli 
                                     # değişen rn değişkeni parametre olarak verilmiştir.
        noise=np.random.rand(5,10)   # 5x10 boyutunda rastgele gürültü üretiliyor. 
        noisy[:,:,i,j] = image[:,:,i] + 0.2 * noise          # gürültülü örüntüler noisy tensöründe depolanıyor
        noisy[:,:,i,j] = noisy[:,:,i,j]/noisy[:,:,i,j].max() # normalizasyon işlemi
        rn+=1                                                # rn 1 arttırıldı
 
        
# gürültülü tensöründeki örüntülerin değerlerini [0.1,0.9] aralığına getirmek için işlemler
noisy=noisy*0.8    
noisy=noisy+0.1


data_gen2=18   # Her sınıf için 18 adet örüntü oluşturulacak. Oluşturulan bu örüntüler bozma işleminden geçirilecek. 
               # Bunlardan 3er tanesi validation, 6şar tanesi test, 9ar tanesi train setine ayrıştırılacak
               
rand_image = np.zeros((5,10,class_num,data_gen2))   # (5,10,4,18) boyutunda bozma işleminden geçirilmiş elemanları tutacak tensör


# bazı örüntülerde 1 değişiklik , bazılarında 2 değişiklik yapılacak
np.random.seed(40)     
rand_index1 = np.random.randint(0,50,(data_gen2,class_num))  # 0 ile 50 arasında (18,4) boyutunda rastgele sayılardan olumuş matris üretildi

np.random.seed(50)
rand_index2 = np.random.randint(0,50,(data_gen2,class_num))  # 2 değişiklik yapılacak örüntüler için


for i in range(class_num):
    for j in range(data_gen2):
        rand_image[:,:,i,j] = np.array(image[:,:,i])   
        
        if (j%2==0): # örünütlerin yarısında 2 değişiklik yapılacak
            
            rand_image[int(rand_index2[j,i] % 5), int(rand_index2[j,i]/5), i, j]=np.where(rand_image[int(rand_index2[j,i]%5), int(rand_index2[j,i]/5), i, j]==1, 0.0, 1.0)   
            # bozulmuş örüntüler rand_image tensöründe depolanıyor
       
        rand_image[int(rand_index1[j,i] % 5), int(rand_index1[j,i]/5), i, j]=np.where(rand_image[int(rand_index1[j,i]%5), int(rand_index1[j,i]/5), i ,j]==1, 0.0, 1.0)
 
    
# rand_image tensöründeki örüntülerin değerlerini [0.1,0.9] aralığına getirmek için işlemler
rand_image=rand_image*0.8
rand_image=rand_image+0.1

length = data_gen1+data_gen2  # toplam örüntü sayısı

data = np.zeros((5,10,class_num,length))      # gürültülü ve bozulmuş tensörlerin tamamını tutacak tensör
data_last = np.zeros((5,10,class_num,length)) # data matrisindeki örüntülerin sırasının değiştirilmiş halini tutacak tensör

data = np.concatenate((noisy, rand_image),axis=3)  # örüntüleri birleştiriliyor


sn=np.zeros((length,1)) # np.random.shuffle() fonksiyonunun içerisine verilecek parametre

for i in range(length): # sn vektörü 0 dan length e kadar değerleri içerecek
    sn[i,0]=i
 
np.random.seed(45)    
np.random.shuffle(sn)   # np.random.shuffle() fonksiyonu ilk sütunun sırasını karıştırıyor. 
                        # ancak biz data tensöründeki 4. boyutun sırasının karıştırılmasını istiyoruz.
                        # bu yüzden sn vektörünün sırasını karıştırıp bunu data tensörüne sıra numarası olarak vereceğiz
                         


for j in range(length):
    data_last[:,:,:,j]=data[:,:,:,int(sn[j,0])]  # data_last tensörü data tensörünün 4. boyutunun sırası karıştırılmış haline eşitleniyor.
     


te_num=12  # test kümesinde her sınıfta kaç örüntünün bulunduğunun sayısı
tr_num=18  # train kümesinde her sınıfta kaç örüntünün bulunduğunun sayısı
va_num=6   # validation kümesinde her sınıfta kaç örüntünün bulunduğunun sayısı

# test, train ve validation kümeleri oluşturuluyor
test_data = np.zeros((5,10,class_num,te_num))
train_data= np.zeros((5,10,class_num,tr_num))
val_data=np.zeros((5,10,class_num,va_num))
  
# data_last matrisindeki örüntüler paylaştırılıyor
test_data = data_last[:,:,:,tr_num:length-va_num]
train_data= data_last[:,:,:,0:tr_num]
val_data=data_last[:,:,:,length-va_num:length]


# 5x10 boyutundaki örüntülerin ağırlık matrisiyle çarpılabilmesi için (50,1) boyutuna getirilmesi
test_data=test_data.reshape(50,1,class_num,te_num)
train_data=train_data.reshape(50,1,class_num,tr_num)
val_data=val_data.reshape(50,1,class_num,va_num)

# örüntüdeki elemanların sonuna bias eklenecek. 
test_data_b= np.zeros((51,1,class_num,te_num))
train_data_b= np.zeros((51,1,class_num,tr_num))
val_data_b= np.zeros((51,1,class_num,va_num))

train_data_wl= np.zeros((52,1,class_num,tr_num)) # hangi örüntünün hangi sınıfta olduğunu anlamak için 
                                                 # örüntü vektörünün en alt kısmına etiket değeri yazılacak

# kümelerdeki her örüntünün sonuna bias ekleme işleme 
for i in range(class_num):                   
    for j in range(te_num): 
        test_data_b[:,0,i,j]=np.append(test_data[:,0,i,j],[1], axis=0)
       
for i in range(class_num):                   
    for j in range(tr_num):         
        train_data_b[:,0,i,j]=np.append(train_data[:,0,i,j],[1], axis=0)
        train_data_wl[:,0,i,j]=np.append(train_data_b[:,0,i,j],[i], axis=0) # train_data_wl tensöründeki her örüntüye etiket bilgisi ekleniyor

for i in range(class_num):                   
    for j in range(va_num):         
        val_data_b[:,0,i,j]=np.append(val_data[:,0,i,j],[1], axis=0)


length_train=tr_num*class_num   # eğitim kümesindeki toplam örüntü sayısı



train_data_wl_2=train_data_wl.reshape(52,1,length_train,1)  # train_data_wl tensörünün 3. ve 4. boyutu değiştirildi. (sıra karıştırma işlemi için)
train_data_wl_1=np.zeros((52,1,length_train,1))             # train_data_wl_2 tensöründeki 3. boyutun sırasının karıştırılmasını istiyoruz. 

sn=np.zeros((length_train,1)) # np.random.shuffle() fonksiyonunun içerisine verilecek parametre

for i in range(length_train): # sn vektörü 0 dan length_train değerine kadar değerleri içerecek
    sn[i,0]=i


np.random.seed(88)    
np.random.shuffle(sn)   # np.random.shuffle() fonksiyonu ilk sütunun sırasını karıştırıyor. 
                        # ancak biz train_data_wl_1 tensöründeki 3. boyutun sırasının karıştırılmasını istiyoruz.
                        # bu yüzden sn vektörünün sırasını karıştırıp bunu train_data_wl_1 tensörüne sıra numarası olarak vereceğiz
                         

for i in range(length_train):
    train_data_wl_1[:,0,i,0]=train_data_wl_2[:,0,int(sn[i,0]),0] # train_data_wl_1 tensörü train_data_wl_2 tensörünün 3. boyutunun sırası karıştırılmış haline eşitleniyor.
   
    
train_data_b=np.zeros((51,1,length_train,1))

for i in range(length_train):    
    train_data_b[:,0,i,0]=np.delete(train_data_wl_1[:,0,i,0], 51, 0) # train_data_b tensöründeki örüntüler, train_data_wl_1 tensöründeki örüntülerin etiketsiz hali


# 2 adet gizli katman olacak
# 1. gizli katmanda 6 adet nöron, 2. gizli katmanda 5 adet nöron, çıkış katmanında 4 adet nöron olacak


fl_num=6   # ilk katmandaki nöron sayısı
sl_num=5   # ikinci katmandaki nöron sayısı

# ağırlık matrisleri [-0.15,0.15] aralığında olacak
np.random.seed(80)
w1=(np.random.rand(fl_num,51)*2-1)*0.15

np.random.seed(90)
w2=(np.random.rand(sl_num,fl_num+1)*2-1)*0.15

np.random.seed(100)
wo=(np.random.rand(class_num,sl_num+1)*2-1)*0.15

# katman çıkışlarındaki v ve y lerin boyutları belirtiliyor.
v1=np.zeros((fl_num,1))  
y1=np.zeros((fl_num,1))
y1_b=np.zeros((fl_num+1,1))  # y1 vektörünün bias eklenmiş halini belirtmek için

v2=np.zeros((sl_num,1))
y2=np.zeros((sl_num,1))
y2_b=np.zeros((sl_num+1,1))  # y2 vektörünün bias eklenmiş halini belirtmek için

vo=np.zeros((class_num,1))
yo=np.zeros((class_num,1))

# yd matrisi. Hamming mesafesi dikkate alınarak oluşturuldu.
yd=np.array([[[1],[0],[0],[0]],[[0],[1],[0],[0]],[[0],[0],[1],[0]],[[0],[0],[0],[1]]])

e=np.zeros((class_num,1)) # çıkıştaki yd-yo farklarını tutmak için. 

temp_wo=wo  # temp_wo bir süreliğine bir önceki wo değerini tutar. (for döngüsü içerisinde bu şekilde olacak)
temp_w2=w2
temp_w1=w1

wo_l=temp_wo # temp_wo'nun bir önceki wo değerini tuttuğu aralıkta, wo_l değeri temp_wo ya eşitlenir.
             # wo_l bir önceki wo yu tutmuş olur. (for döngüsü içerisinde bu şekilde olacak)               
w2_l=temp_w2
w1_l=temp_w1

temp_error=10000   # durdurma kriteri olarak cross validation yöntemi kullanılacaktır.
                   # yeni hataların ortalaması, eski hataların ortalamasından daha büyük olursa eğitim sonlandırılır.
                   # eski hataların ortalamasını tutmak için temp_error değişkeni oluşturuldu.
                   # eski hatanın ilk değeri olarak büyük değerli bir sayı değeri atandı.
                   # küçük atanırsa eğitim daha birinci iterasyondayken durabilir
                   # (oluşacak ilk hatanın büyüklüğünü bilmediğimiz için)



for d in range(iteration):                    # iterasyon sayısı kadar döngü devam edecek
    error_sum = 0                             # Hataların toplamı her iterasyonda sıfırlanıyor.
    index_c1=0                                # train_data_b tensörünün 3. boyutunun indeksini belirtmek için. 3 boyutun büyüklüğü tr_num*class_num kadar
    for j in range(class_num):                # sınıf sayısı kadar döngü devam edecek
        for k in range(tr_num):               # train_data daki örüntü sayısı kadar döngü devam edecek
            
            x_b=train_data_b[:,:,index_c1,0]       # (51,1) boyutundaki örüntü değerleri x_b ye eşitlendi
            
            v1 = np.dot(w1, x_b)                   # x_b, w1 ağırlık matrisi ile çarpılıyor
            y1 = sigmoid(v1)                       # v1 sonucu sigmoid fonksiyonundan geçiriliyor
            y1_b = np.append(y1, [[1]], axis=0 )   # y1 vektörüne bias ekleniyor
            
            v2 = np.dot(w2,y1_b)                   # y1_b, w2 ağırlık matrisi ile çarpılıyor
            y2 = sigmoid(v2)                       # v2 sonucu sigmoid fonksiyonundan geçiriliyor
            y2_b = np.append(y2, [[1]], axis=0 )   # y2 vektörüne bias ekleniyor
            
            vo = np.dot(wo,y2_b)                   # y2_b, wo ağırlık matrisi ile çarpılıyor
            yo = sigmoid(vo)                       # vo sonucu sigmoid fonksiyonundan geçiriliyor
            
            ind=int(train_data_wl_1[51,0,index_c1,0]) # train_data_wl_1 tensöründeki örüntüler etiket değerine de sahip
                                                      # ind değişkenine, train_data_wl_1[51,0,index_c1,0]'daki etiket değeri atanıyor
            e=yd[ind]-yo                              # yd-yo, epsilona eşitleniyor.
            
            # ağırlık matrislerinin bias terimi ile çarpılan sütunları çıkartılıyor
            wo_wb=np.delete(wo,5,axis=1)    
            w2_wb=np.delete(w2,6,axis=1)
            w1_wb=np.delete(w1,50,axis=1)
            
            # bias terimi ile çarpılan sütunu olmayan ağırlık matrislerinin transpozu alınıyor
            wo_t=np.transpose(wo_wb)
            w2_t=np.transpose(w2_wb)
            w1_t=np.transpose(w1_wb)
            
            # delta değerleri hesaplanıyor
            deltao= e*derivative(sigmoid, vo)
            delta2=np.dot(wo_t,deltao)*derivative(sigmoid, v2)
            delta1=np.dot(w2_t,delta2)*derivative(sigmoid, v1)
            
            # ağırlıkların güncellenmesi
            w1=w1+learning_rate*np.dot(delta1,np.transpose(x_b))+momentum*(w1-w1_l)
            w2=w2+learning_rate*np.dot(delta2,np.transpose(y1_b))+momentum*(w2-w2_l)
            wo=wo+learning_rate*np.dot(deltao,np.transpose(y2_b))+momentum*(wo-wo_l)  
            
            
            w1_l=temp_w1 # eski ağırlıklar temp_w1 de tutuluyor. w1_l eski ağırlıklara eşitleniyor
            w2_l=temp_w2
            wo_l=temp_wo
            
            temp_w1=w1   # temp_w1 yeni ağırlıklara eşitlendi.
            temp_w2=w2
            temp_wo=wo
            
            index_c1+=1 # index 1 arttırılıyor
            
            
    error_sum_ve=0     # validation kümesinin testi sırasındaki hataların toplamı her iterasyonda sıfırlanıyor
    
    for a in range(class_num): 
        for b in range(va_num): 
            
            # ağırlık güncellemesi olmadan test işlemi yapılıyor.
            # bu kısım durdurma kriteri olarak kullanılacak
            
            x_b=val_data_b[:,:,a,b]
            
            v1 = np.dot(w1, x_b) 
            y1 = sigmoid(v1)             
            y1_b = np.append(y1, [[1]], axis=0 )
            
            v2 = np.dot(w2,y1_b)   
            y2 = sigmoid(v2)
            y2_b = np.append(y2, [[1]], axis=0 )
            
            vo = np.dot(wo,y2_b)
            yo = sigmoid(vo)
            
             
            e=yd[a]-yo
            E=0.5*np.dot(np.transpose(e),e) # hata fonksiyonu
            error_sum_ve += E               # hataların toplanması
            
            
    mean_ve=error_sum_ve/(va_num*class_num)    # hataların ortalaması alınıyor 
    mean_vel=temp_error                        # bir önceki hataların ortalama değeri, geçici hata ortalama değerine eşitleniyor
    temp_error=mean_ve                         # geçici hata ortalaması şuanki hata ortalamasına eşitleniyor 
   
    if(mean_vel<mean_ve): break                # Durdurma kriteri: Eski hataların ortalama değeri yeni hataların ortalama değerinden küçükse eğitimi durdur.
    
length_test=class_num*te_num

r_holder=np.zeros((4,1,length_test))   # test sonuçlarının görselleştirilmesi için tensör

test_counter = 0;                   # Test sonuçlarının doğruluğunu saymak için değişken
error_sum_t=0                       # test kümesindeki hataların toplamını tutacak değişken
index_c2=0                          # 0 dan te_num*class_num a kadar değerler alacak
 
for i in range(class_num):                        # her bir sınıf için döngü
    for j in range(te_num):                       # test kümesindeki elemanlar için döngü
    
        x_b=test_data_b[:,:,i,j]
       
        v1 = np.dot(w1, x_b) 
        y1 = sigmoid(v1)             
        y1_b = np.append(y1, [[1]], axis=0 )
       
        v2 = np.dot(w2,y1_b)   
        y2 = sigmoid(v2)
        y2_b = np.append(y2, [[1]], axis=0 )
       
        vo = np.dot(wo,y2_b)
        yo = sigmoid(vo)
        
        r_holder[:,:,index_c2]=yo    # ÇKA'nın çıkışı r_holder a atanıyor
        
        e=yd[i]-yo    
        
        # sınıflandırmanın doğru olup olmadığının test edilmesi
        if (i==0 and (e[0]<=0.3 and -0.4<=e[1]<=0 and -0.4<=e[2]<=0 and -0.4<=e[3]<=0)): 
            test_counter+=1 
        elif (i==1 and (-0.4<=e[0]<=0 and e[1]<=0.3 and -0.4<=e[2]<=0 and -0.4<=e[3]<=0)): 
            test_counter+=1
        elif (i==2 and (-0.4<=e[0]<=0 and -0.4<=e[1]<=0 and e[2]<=0.3 and -0.4<=e[3]<=0)): 
            test_counter+=1
        elif (i==3 and (-0.4<=e[0]<=0 and -0.4<=e[1]<=0 and -0.4<=e[2]<=0 and e[3]<=0.3)):
            test_counter+=1
            
        
        E=0.5*np.dot(np.transpose(e),e)
        error_sum_t += E        # hatalar toplanıyor
    
        index_c2+=1             # index 1 arttırılıyor
    
    
r_holder_last=r_holder.reshape(4,length_test)  
r_holder_last_t=np.transpose(r_holder_last)    # Sonuçları tablo halinde görmek için


mean_t = error_sum_t/(length_test)            # test kümesindeki hataların ortalaması
    
result = (test_counter/(te_num*class_num))*100     # sonuç % değerinin hesaplanması
print("%{:.2f}".format(result))                    # sonucun yazdırılması




