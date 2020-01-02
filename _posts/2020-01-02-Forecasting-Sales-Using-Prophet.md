---
layout: post
title: "Forecasting Sales (Time-series) Using Prophet Algorithms."
categories:
  - Post Formats
tags:
  - Data Analyst
  - Machine Learning
---

# Introduction

Sebelumnya saya ingin menyampaikan bahwa dataset yang saya peroleh berasal dari test Data Engineer di Bareksa. Dengan dokumentasi ini saya berhasil lolos tahap test awal namun gagal saat interview (saya akui saat itu masih banyak kekurangan dalam diri saya, hehe). Berikut ini adalah soal test nya:
![test](https://i.imgur.com/NKoqCGA.png)

Disini saya tidak akan menjelaskan math behind Prophet algorithm nya ya hehe.

# 1. Understanding the dataset

Dataset nya sendiri terdiri dari 2 csv yaitu test_data_de.csv dan train_data_de.csv, kemudian karena ini forecasting saya akan gabungkan kedua nya menjadi data_combine.csv

Features yang ada di dalam dataframe adalah: date, sku, Price, promo_item, promo_card, dan qty. Saya
tidak mengerti dari feature sku sehingga akan saya abaikan.


# 2. Viewing the data

Mari kita lihat sekilas seperti apa data nya.
```python
print("This is Head")
print(df.head())


print('This is Tail')

print(df.tail())

```
![image](https://i.imgur.com/d11pqGM.png)
Totalnya, ada sekitar 2396 datapoints dalam dataframe. Selanjutnya kita akan melihat outlook dari dataset.

```python
df.describe().T
```
![image](https://i.imgur.com/g5MMt9w.png)
Semua numerical features terlihat baik: full count values dan no null. Ada tiga tipe store, yaitu: ta,tb, dan tc. Semua data terlihat bersih, saatnya lanjut ketahap berikutnya.


# 3. Profile promo_item

Saya akan membagi kedalam 3 tahap data eksplorasi yaitu: Profiling promo_item, profiling promo_card, dan profiling store. Saya menggunakan Seaborn dan Matplotlib untuk Data Visualisasi dan Storry Telling.
## Data Exploration
```python
# Import seaborn and matplotlib with matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

matplotlib.rcParams['font.size'] = 3
matplotlib.rcParams['figure.dpi'] = 200
```
Selanjutnya kita akan memisahkan terhadap features yang melakukan promo_item dan tidak melakukan promo_item

```python
# Getting data of promo and not promo
promodf = df[df['promo_item']==1]
notpromodf =df[df['promo_item']==0]
print("promodf:\n{}".format(promodf))

# Getting the shapes and numbers of these people
print("shape of promodf:{} ".format(promodf.shape))
print("shape of notpromodf:{} ".format(notpromodf.shape))
```
![promodf](https://i.imgur.com/tXBottP.png)
Jumlah promo_item yang dilakukan adalah 135 dan tidak melaukan promo_item adalah 2262. Selanjutnya kita akan melihat distribusi dari promo_item dan non-promo_item


Lalu akan kita visualisasikan
```python
#make seperated list for price
x1 = list(promodf['Price'])
x2 = list(notpromodf['Price'])

#make seperated list for store
x3 = list(promodf['Store'])
x4 = list(notpromodf['Store'])

#make seperated list for promo_card
x5 = list(promodf['promo_card'])
x6 = list(notpromodf['promo_card'])

#make seperated list for quantity
x7 = list(promodf['qty'])
x8 = list(notpromodf['qty'])

#assign color and names
colors = ['#E69F00', '#56B4E9']
names = ['promo', 'not promo']

#Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(411)
plt.hist([x1,x2], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('Price')
plt.ylabel('normalized value')

# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(412)
plt.hist([x3,x4], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('Store')
plt.ylabel('normalized value')

# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(413)
plt.hist([x5,x6], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('promo_card')
plt.ylabel('normalized value')

# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(414)
plt.hist([x7,x8], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('quantity')
plt.ylabel('normalized value')


plt.subplots_adjust(top=11 ,bottom=10)
plt.tight_layout()
plt.show()
```
![image](https://i.imgur.com/WcC0sXS.png)

Sekarang, apa insight yang bisa kita dapatkan?

### Insights:

•	Price: Promo_item dilakukan lebih sering terhadap harga menengah antara 2000 sampai 2500.
•	Store: tb yang paling sering melakukan promo_item disusul oleh tc dan ta
•	promo_card: Perusahaan cukup sering melakukan promo_item dan promo_card secara bersamaan.
•	qty: Penjualan jumlah besar ternyata sering ditemui terhadap yang melakukan promo_item

Seberapa sering Store melakukan promo_item dapat kita lihat dengan menggunakan pie chart.
```python
# Do the value counts of store
promostorecounts = promodf['Store'].value_counts()
notpromostorecounts = notpromodf['Store'].value_counts()

# plot each pie chart in a separate subplot
plt.subplot(221)
plt.pie(promostorecounts, labels=promostorecounts.index, autopct='%1.1f%%')
plt.title('Promo_item')
plt.axis('equal')

# plot each pie chart in a separate subplot
plt.subplot(222)
plt.pie(notpromostorecounts, labels=notpromostorecounts.index, autopct='%1.1f%%')
plt.title('No Promo_item')
plt.axis('equal')
```
![pie1](https://i.imgur.com/8PDH8oB.png)
Terbukti bahwa tb sering melakukan promo_item dengan jumlah 42%, disusul tc dan ta yang masing-masing 34.1% dan 23%.

## Correlation Analysis

Sekarang mari kita analisa korelasi fitur-fitur yang ada terhadap promo_item. Ini dilakukan dengan menggunakan metode .corr(). Lalu kita aplikasikan kedalam Seaborn Heatmap.

```python
corr_promo_item = promodf.drop('promo_item',axis=1).corr()
sns.heatmap(corr_promo_item)
```
![correlation1](https://i.imgur.com/3TVLleT.png)

Kita dapat mencari element mana yang kemungkinan saling bergantungan:
1.	promo_card dan qty: nilai ketergantungannya tidak terlalu besar namun kita dapat mengetahui ada korelasi antara promo_card dan penjualan terhadap promo_item.

# 4. Profile promo_card

Kode yang kita gunakan akan sama dengan profile sebelumnya jadi saya akan menampilkan hasil analisanya secara langsung.

![image](https://i.imgur.com/zZGPyiF.png)

Sekarang, apa insight yang bisa kita dapatkan? 
•	Price: Promo_item dilakukan lebih sering terhadap harga menengah antara 2000 sampai 2500.
•	Store: tb yang paling sering melakukan promo_card disusul oleh tc dan ta
•	qty: Penjualan volume besar ternyata sering ditemui terhadap yang melakukan promo_card
Sebenarnya tidak terlalu berbeda jauh dengan promo_item jadi analisa saya cukupkan hanya sampai sini.

# 5. Profile Store
Kita akan melihat toko mana yang melakukan penjualan dalam volume yang tinggi:
```python
# Getting data of each store ta, tb, tc
tadf = df[df['Store']=='ta']
tbdf =df[df['Store']=='tb']
tcdf =df[df['Store']=='tc']
print("tadf:\n{}".format(tadf))
print("tbdf:\n{}".format(tbdf))
print("tcdf:\n{}".format(tcdf))

# Getting the shapes and numbers of these people
print("shape of tadf:{} ".format(tadf.shape))
print("shape of tbdf:{} ".format(tbdf.shape))
print("shape of tcdf:{} ".format(tcdf.shape))

#make seperated list for price
x1 = list(tadf['Price'])
x2 = list(tbdf['Price'])
x3 = list(tcdf['Price'])

#make seperated list for qty
x4 = list(tadf['qty'])
x5 = list(tbdf['qty'])
x6 = list(tcdf['qty'])

#assign color and names
colors = ['#E69F00', '#56B4E9', '#17E449']
names = ['ta', 'tb', 'tc']

#Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(221)
plt.hist([x1,x2,x3], bins=12, density=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('Price')
plt.ylabel('density')

# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(222)
plt.hist([x4,x5,x6], bins=12, density=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('qty')
plt.ylabel('density')


plt.subplots_adjust(top=11 ,bottom=10)
plt.tight_layout()
plt.show()
```

![store](https://i.imgur.com/oIxBssT.png)

Sekarang, apa insight yang bisa kita dapatkan? 
•	Transaksi dengan nominal harga tertinggi ada toko ta yaitu diatas 3500. Hal yang paling mencolok disini adalah volume transaksi terbesar ada di nominal 2300-an dengan toko tc memiliki performa terbaik disusul toko tc dan ta
•	Untuk kuantitas terbanyak dilakukan oleh toko tb, yaitu diatas 60 kemungkinan hal ini disebabkan oleh rutinnya toko tb melakukan promo_card maupun promo_item. Penjualan volume tertinggi ada di kuantitas 0 sampai 20.
 



Thank you so much for reading. For your information, I am right now on the path to became Junior Data Scientist and this work can't happen without help of many people from stackoverflow, reddit.com/r/MLQuestion, towardsdatascientist, etc.

