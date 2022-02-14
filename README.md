# Laporan Proyek Machine Learning - Sulaiman Muharik

## Domain Proyek

Situs penyedia film menyediakan berbagai jenis film yang sangat beragam untuk ditonton oleh para pengguna. Namun, setiap pengguna memiliki preferensi film masing-masing untuk ditonton. Dalam mencari film yang pengguna sukai memakan waktu yang cukup lama. Oleh karena itu, diperlukan sistem rekomendasi yang dapat merekomendasikan film lain yang mungkin disukai dan belum pernah ditonton oleh pengguna. Teknik collaborative filtering dapat digunakan menggunakan dataset yang ada. Ini akan memungkinkan pengguna tidak perlu repot-repot untuk mencari film yang ingin ditonton dan peningkatan jumlah penonton pada film tersebut.

## Business Understanding

### Problem Statements

- Dengan data rating yang Anda miliki, bagaimana perusahaan dapat merekomendasikan film lain yang mungkin disukai dan belum pernah ditonton oleh pengguna? 

### Goals

- Menghasilkan sejumlah rekomendasi restoran yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik collaborative filtering.


## Data Understanding
Pada kasus ini, terdapat 2 file, ratings dan movies. Dataset ratings berisi semua rating film yang diberikan rating oleh user dan dataset movies berisi data semua film yang ada. Dataset rating memiliki banyak rating sebanyak 100836. Dataset movies memiliki banyak film sebanyak 9742. File dataset: [Movie Recommender System Dataset](https://www.kaggle.com/gargmanas/movierecommenderdataset).

### Variabel-variabel pada Movie Recommender System Dataset adalah sebagai berikut:
Untuk memahami data dilakukan tahap univariate exploratory data analysis.

- rating variable
  File ratings.csv memiliki 100836 entri dan tidak terdapat missing value.
  
  [ratings_df.head()]
  Data ratings terdiri dari 4 kolom. Kolom-kolom tersebut antara lain: 
  - userId : merupakan ID yang dimiliki oleh user.
  - movieId: merupakan ID yang dimiliki oleh film.
  - rating : merupakan rating film yang diberikan rating oleh user.
  - timestamp: merupakan cap waktu ketika rating diberikan.
  
  Terdapat 100836 rating yang diberikan oleh 610 user untuk 9724 film.
  
- movies variable
  File movies.csv memiliki 9742 entri dan tidak terdapat missing value.

  [movies_df.head()]
  - movieId: merupakan ID yang dimiliki oleh film.
  - title  : merupakan judul film yang dimiliki.
  - genres : merupakan genre yang dimiliki oleh film tertentu.

  Terdapat 9737 film yang memiliki rating dan terdapat 5 judul memiliki genre yang berbeda.


## Data Preparation
Pada file movie.csv terdapat 5 judul film yang memiliki variable genre lebih dari 1. Oleh karena itu, perlu dilakukan data preparation pada dataset.
 - Pembersihan data.
 - Menyandikan fitur
 - Pembagian dataset.


## Modeling
Pada kasus ini diterapkan teknik collaborative filterin untuk membuat sistem rekomendasi karena dataset yang dimiliki sangat cocok untuk teknik ini, yaitu data rating dari user. Dari data rating pengguna akan diidentifikasi film -film yang mirip dan belum pernah dikunjungi oleh pengguna untuk direkomendasikan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada sistem rekomentasi ini, metrik yang digunakan adalah precision. Precision merupakan ... dengan rumus: 
[gambar]

Dari hasil prediksi, didapatkan 

**---Ini adalah bagian akhir laporan---**

