# Laporan Proyek Machine Learning - Sulaiman Muharik

## Domain Proyek

Situs penyedia film menyediakan berbagai jenis film yang sangat beragam untuk ditonton oleh para pengguna. Namun, setiap pengguna memiliki preferensi film masing-masing untuk ditonton. Dalam mencari film yang pengguna sukai memakan waktu yang cukup lama. Oleh karena itu, diperlukan sistem rekomendasi yang dapat merekomendasikan film lain yang mungkin disukai dan belum pernah ditonton oleh pengguna. Teknik collaborative filtering dapat digunakan menggunakan dataset yang ada. Ini akan memungkinkan pengguna tidak perlu repot-repot untuk mencari film yang ingin ditonton dan peningkatan jumlah penonton pada film tersebut.

## Business Understanding

### Problem Statements

- Dengan data rating yang Anda miliki, bagaimana perusahaan dapat merekomendasikan film lain yang mungkin disukai dan belum pernah ditonton oleh pengguna? 

### Goals

- Menghasilkan sejumlah rekomendasi restoran yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik collaborative filtering.


## Data Understanding
Pada kasus ini, terdapat 2 data, ratings dan movies. Data ratings berisi semua rating film yang diberikan rating oleh user dan dataset movies berisi data semua film yang ada. Dataset rating memiliki banyak rating sebanyak 100836. Data movies memiliki banyak film sebanyak 9742. File dataset: [Movie Recommender System Dataset](https://www.kaggle.com/gargmanas/movierecommenderdataset).

### Variabel-variabel pada Movie Recommender System Dataset adalah sebagai berikut:
Untuk memahami data dilakukan tahap univariate exploratory data analysis.

- rating variable
  
  File ratings.csv memiliki 100836 entri dan tidak terdapat missing value.
  
  |  | userId |	movieId |	rating |	timestamp |
  | --- | --- | --- | --- | --- |
  | 0 |	1 |	1 |	4.0 |	964982703 |
  | 1 |	1 |	3 |	4.0 |	964981247 |
  | 2 |	1 |	6 |	4.0 |	964982224 |
  | 3 |	1 |	47 |	5.0 |	964983815 |
  | 4 |	1 |	50 |	5.0 |	964982931 |
  
  Data ratings terdiri dari 4 kolom. Kolom-kolom tersebut antara lain: 
  - userId : merupakan ID yang dimiliki oleh user.
  - movieId: merupakan ID yang dimiliki oleh film.
  - rating : merupakan rating film yang diberikan rating oleh user.
  - timestamp: merupakan cap waktu ketika rating diberikan.
 
  Terdapat 100836 rating yang diberikan oleh 610 user untuk 9724 film.    
  ```
  Jumlah userId:  610
  Jumlah movieId:  9724
  Jumlah data rating:  100836
  ```  
  
  
- movies variable
  
  File movies.csv memiliki 9742 entri dan tidak terdapat missing value.

  | |	movieId |	title |	genres |
  | --- | --- | --- | --- |
  | 0 |	1 |	Toy Story (1995) |	Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 1 |	2 |	Jumanji (1995) |	Adventure\|Children\|Fantasy |
  | 2 |	3 |	Grumpier Old Men (1995) |	Comedy\|Romance |
  | 3 |	4 |	Waiting to Exhale (1995) |	Comedy\|Drama\|Romance |
  | 4 |	5 |	Father of the Bride Part II (1995) |	Comedy |

  Data ratings terdiri dari 4 kolom. Kolom-kolom tersebut antara lain: 
  - movieId: merupakan ID yang dimiliki oleh film.
  - title  : merupakan judul film yang dimiliki.
  - genres : merupakan genre yang dimiliki oleh film tertentu.
  
  Terdapat 9737 film yang memiliki rating dan terdapat 5 judul memiliki genre yang berbeda.
  ```
  Jumlah movieId:  9742
  Jumlah film:  9737
  ```


## Data Preparation
Pada file movie.csv terdapat 5 judul film yang memiliki variable genres lebih dari 1. Oleh karena itu, perlu dilakukan data preparation pada dataset.
 - Pembersihan data.
 
   Pada proses pembersihan data, dilakukan penyamaan genres dan penanganan duplikat. Hal ini dilakukan untuk memberikan rekomendasi yang lebih baik.
   - Penyamaan genres.
   
     Pada data movies terdapat 5 film yang memiliki deskripsi genres yang berbeda. Film yang memiliki genres lebih banyak akan dipilih untuk mewakilkan film tersebut.
   
     |  | movieId |	title |	genres |
     | --- | --- | --- | --- |
     | 4169 |	6003 |	Confessions of a Dangerous Mind (2002) |	Comedy\|Crime\|Drama\|Thriller |
     | 9106 |	144606 |	Confessions of a Dangerous Mind (2002) |	Comedy\|Crime\|Drama\|Romance\|Thriller |
     | 650 |	838 |	Emma (1996) |	Comedy\|Drama\|Romance |
     | 5601 |	26958 |	Emma (1996) |	Romance |
     | 5854 |	32600 |	Eros (2004) |	Drama |
     | 9135 |	147002 |	Eros (2004) |	Drama\|Romance |
     | 2141 |	2851 |	Saturn 3 (1980) |	Adventure\|Sci-Fi\|Thriller |
     | 9468 |	168358 |	Saturn 3 (1980) |	Sci-Fi\|Thriller |
     | 5931 |	34048 |	War of the Worlds (2005) |	Action\|Adventure\|Sci-Fi\|Thriller |
     | 6932 |	64997 |	War of the Worlds (2005) |	Action\|Sci-Fi |
   
     Pada data ratings, movieId film yang memiliki genres lebih sedikit tersebut akan digantikan dengan yang memiliki genres lebih banyak.
   
   - Penanganan duplikat
   
     Setelah dilakukan penyamaan genres, pada data ratings, terdapat data duplikat dengan userId dan movieId yang sama, maka diperlukan dari salah satu data untuk di-drop. Setelah duplikat di-drop banyak data berubah menjadi 100832 entri.
        
 - Penyandian fitur.

   Variabel userId dan movieId merupakan ID yang dimiliki user dan movie, ID tersebut merupakan susunan dari bilangan. Untuk memudahkan mesin belajar dilakukan penyandian fitur.

 - Pembagian dataset.
  
   Pembagian dataset menjadi data train dan test perlu dilakukan untuk menguji model pada data test, dengan model yang telah dilatih pada data train.
   


## Modeling
Pada kasus ini diterapkan teknik collaborative filterin untuk membuat sistem rekomendasi karena dataset yang dimiliki sangat cocok untuk teknik ini, yaitu data rating dari user. Dari data rating pengguna akan diidentifikasi film -film yang mirip dan belum pernah dikunjungi oleh pengguna untuk direkomendasikan.


## Evaluation
Pada sistem rekomentasi ini, metrik yang digunakan adalah precision. Precision merupakan ... dengan rumus: 

[gambar]

Hasil Rekomendasi
```
Showing recommendations for users: 246
===========================
Movies with high ratings from user
--------------------------------
Sense and Sensibility (1995) : Drama|Romance
Happy Together (a.k.a. Buenos Aires Affair) (Chun gwong cha sit) (1997) : Drama|Romance
Pride and Prejudice (1995) : Drama|Romance
Porco Rosso (Crimson Pig) (Kurenai no buta) (1992) : Adventure|Animation|Comedy|Fantasy|Romance
Howl's Moving Castle (Hauru no ugoku shiro) (2004) : Adventure|Animation|Fantasy|Romance
--------------------------------
Genres user likes: {'Drama', 'Fantasy', 'Animation', 'Comedy', 'Romance', 'Adventure'}
--------------------------------
Top 10 movies recommendation
--------------------------------
Shawshank Redemption, The (1994) : Crime|Drama
Secrets & Lies (1996) : Drama
Princess Bride, The (1987) : Action|Adventure|Comedy|Fantasy|Romance
Lawrence of Arabia (1962) : Adventure|Drama|War
Apocalypse Now (1979) : Action|Drama|War
Ran (1985) : Drama|War
Amadeus (1984) : Drama
Guess Who's Coming to Dinner (1967) : Drama
Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) : Action|Animation|Drama|Fantasy|Sci-Fi
Three Billboards Outside Ebbing, Missouri (2017) : Crime|Drama
```


**---Ini adalah bagian akhir laporan---**

