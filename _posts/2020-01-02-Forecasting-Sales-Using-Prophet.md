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

Prediksi penjualan atau sales forecasting adalah sebuah tolak ukur apakah bisnis yang kita jalankan apakah sudah sesuai dengan arah yang kita inginkan, atau belum. Dengan melaukan prediksi penjualan masa depan kita dapat menyusun strategi untuk mempertahankan nilai dari bisnis kita.

Tim Inti Data Science dari Facebook telah mengembangkan dan membuat sebuah open-source tool untuk business forecasting, yaitu Prophet. Mengutip dari kata-kata mereka, tujuan pengembangan Prophet adalah untuk “make it easier for experts and non-experts to make high quality forecasts that keep up with demand”. Aplikasi dari Prophet dimulai dengan modelling sebuah time-series menggunakan parameter yang ditentukan analis, produksi forecast dan kemudian mengevaluasinya.

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
 
# Prophet

Pada intinya, Prophet adalah model tambahan dengan komponen berikut:
1[](http://latex.codecogs.com/gif.latex?y%28t%29%3Dg%28t%29&plus;s%28t%29&plus;h%28t%29&plus;%5Cepsilon_%7B%28t%29%7D)

* g(t) models trend, yang menggambarkan peningkatan atau penurunan data dalam jangka panjang. Prophet menggabungkan dua model tren, saturating growth model dan piecewise linear model, tergantung pada jenis masalah dari forecasting.
*	s(t) models seasonality, yang menggambarkan bagaimana data dipengaruhi factor musiman seperti waktu tahunan (contoh: musim hujan, musim liburan dll.)
*	h(t) memodelkan efek dari liburan atau even besar yang mempengaruhi bisnis (contoh: hari raya idul fitri, hari raya natal, hari libur nasional, dll)
*	ϵ_t adalah sebuah istilah eror yang tidak dapat direduksi.

## Set up

Dimulai dengan melakukan import terhadap library yang akan digunakan. Install Prophet terlebih dahulu menggunakan pip. [Panduan bagaimana menginstall Prophet.](https://facebook.github.io/prophet/docs/installation.html#python)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
%matplotlib inline
```
## Preparing the data
1.	Untuk forecasting kita akan berfokus pada date dan qty kolom, sehingga kita akan drop kolom lainnya.
2.	Konversi date ke dalam datetime format menggunakan fungsi to_datetime().
3.	Kita akan melakukan resampling data kedalam format monthly.

```python
df = df[['date', 'qty']].dropna()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

daily_df = df.resample('D').mean()
d_df = daily_df.reset_index().dropna()
d_df.sort_values(by=['date'])

sns.set_style('whitegrid')
plt.figure(figsize = (12, 5))
ax = plt.axes()

sns.lineplot(x='date', y='qty', data=d_df, color='#76b900')
ax.xaxis.set_major_locator(plt.MaxNLocator('auto'))
plt.title('Quantity Sold', fontsize=16)
plt.xlabel('date', fontsize=16)
plt.ylabel('qty', fontsize=16)
```
Selanjutnya, perubahan quantity dalam suatu waktu akan divisualisasikan. 
![](https://i.imgur.com/Yap1HAl.png)

Terlihat pada gambar bahwa data non-stationary sehingga penggunaka Prophet merupakan langkah yang tepat.

## Forecast

Prophet mengikuti model API dari sklearn, sehingga memudahkan pengguna yang terbiasa menggunakannya. Sebelum melakukan forecasting saya melakukan [tuning terhadap parameter Prophet.](https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3)

![](https://i.imgur.com/IjFtNmv.png)

```python
df.columns = ['ds', 'y']

prophet = Prophet(
    growth ="linear",
    seasonality_mode="multiplicative",
    changepoint_prior_scale=30,
    seasonality_prior_scale=35,
    holidays_prior_scale=20,
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
).add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=55
).add_seasonality(
    name='daily',
    period=1,
    fourier_order=15
).add_seasonality(
    name='weekly',
    period=7,
    fourier_order=20
).add_seasonality(
    name='yearly',
    period=365.25,
    fourier_order=20
).add_seasonality(
    name='quarterly',
    period=365.25/4,
    fourier_order=5,
    prior_scale=15)

m = Prophet()
m.fit(df)
```

## Plot
Selanjutnya plotting terhadap forecast dataframe.

```python
future = m.make_future_dataframe(periods=90)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
```

![](https://i.imgur.com/D9zo5xR.png)

Kita dapat memecah ini sedikit lebih jauh dengan memanggil metode plot_components() untuk memereksa komponen dari forecast.
```python
fig2 = m.plot_components(forecast)
```
![](https://i.imgur.com/BVR3uZY.png)

Terlihat bahwa trend dari penjualan menunjukan fall pattern. Dibutuhkan sebuah strategi marketing baru untuk mempertahankan kuantitas penjualan. Hal ini kemungkinan disebabkan karena ekonomi global sedang mengalami uncertainty.
Jika melihat dari plot per-tahun dapat diketahui penjualan tertinggi ada pada pertengahan bulan September  disusul pertengahan Maret dan April. 

## Evaluasi

1.	Fungsi cross_validation() akan digunakan pada model dan spesifikasi forecast horizon dengan parameter horizon.
2.	Selanjutnya menggunakan performance_metrics() untuk mengetahui performance metrics.

```python
from fbprophet.diagnostics import cross_validation, performance_metrics
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
df_p = performance_metrics(df_cv)
df_p
```
![](https://i.imgur.com/gs533Dw.png)

Kemudian lakukan plot terhadap mean absolute percent error (MAPE) terhadap forecast horizon untuk menentukan apakah forecasting yang kita lakukan dapat dipercaya.
```python
from fbprophet.plot import plot_cross_validation_metric
fig3 = plot_cross_validation_metric(df_cv, metric='mape')
```
![](https://i.imgur.com/9nyFkTV.png)

Dari table performance metrics dan Gambar 5. bahwa akurasi menurun saat horizon diperluas. Secara keseluruhan, error menurun dari 27% pada bulan pertama dan menjadi sekitar 1% pada bulan keempat. Setelah bulan keempat error cenderung mengalami kenaikan sampai setelah bulan ke-12 error meningkat menjadi sekitar 650%.

# Kesimpulan

1.	Model ini sangat baik untuk melakukan prediksi/forecasting terhadap 4 bulan pertama dan cenderung naik terhadap bulan berikutnya. Error menjadi sangat tinggi setelah forecasting horizon 275 hari.
2.	Perlunya dilakukan perubahan strategi bisnis agar penjualan tetap sustainable di masa yang akan dating, karena trendline menunjukan penurunan.

Thank you so much for reading. For your information, I am right now on the path to became Junior Data Scientist and this work can't happen without help of many people from stackoverflow, reddit.com/r/MLQuestion, towardsdatascientist, etc.

