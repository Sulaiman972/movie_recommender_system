# Laporan Proyek Machine Learning - Sulaiman Muharik

## Domain Proyek

Situs penyedia film menyediakan berbagai jenis film yang sangat beragam untuk ditonton oleh para pengguna. Namun, setiap pengguna memiliki preferensi film masing-masing untuk ditonton. Dalam mencari film yang pengguna sukai memakan waktu yang cukup lama. Oleh karena itu, diperlukan sistem rekomendasi yang dapat merekomendasikan film lain yang mungkin disukai dan belum pernah ditonton oleh pengguna. Teknik collaborative filtering dapat digunakan menggunakan dataset yang ada. Ini akan memungkinkan pengguna tidak perlu repot-repot untuk mencari film yang ingin ditonton dan peningkatan jumlah penonton pada film tersebut.

Referensi:

Najafi, Safir. ["Evaluating Prediction Accuracy for Collaborative Filtering Algorithms in Recommender Systems"](https://kth.diva-portal.org/smash/get/diva2:927356/FULLTEXT01.pdf). 2016. KTH Royal Institute of Technology.

Banerjee, Siddhartha. ["Collaborative Filtering for Movie Recommendations"](https://keras.io/examples/structured_data/collaborative_filtering_movielens/). 2020. Keras.

## Business Understanding

### Problem Statements

- Dengan data rating yang perusahaan miliki, bagaimana perusahaan dapat merekomendasikan film lain yang mungkin disukai dan belum pernah ditonton oleh pengguna? 

### Goals

- Menghasilkan sejumlah rekomendasi film yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik _collaborative filtering_.


## Data Understanding
Pada kasus ini, terdapat 2 data, ratings dan movies. Data ratings berisi semua rating film yang diberikan rating oleh user dan dataset movies berisi data semua film yang ada. Dataset rating memiliki banyak rating sebanyak 100836. Data movies memiliki banyak film sebanyak 9742. File dataset: [Movie Recommender System Dataset](https://www.kaggle.com/gargmanas/movierecommenderdataset).

### Variabel-variabel pada Movie Recommender System Dataset adalah sebagai berikut:
Untuk memahami data dilakukan tahap univariate exploratory data analysis.

- rating variabel
  
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
  
  
- movies variabel
  
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
  
  Terdapat 9737 film yang memiliki genres dan terdapat 5 film memiliki genres yang berbeda atau ganda.
  ```
  Jumlah movieId:  9742
  Jumlah film:  9737
  ```


## Data Preparation
Pada file movie.csv terdapat 5 judul film yang memiliki variabel genres lebih dari 1. Oleh karena itu, perlu dilakukan data preparation pada dataset.
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
        
 - Encoding fitur.

   Variabel userId dan movieId merupakan ID yang dimiliki user dan movie, ID tersebut merupakan susunan dari bilangan. Untuk memudahkan mesin belajar dilakukan penyandian fitur.
   
   | | userId |	movieId |	rating |
   | --- | --- | --- | --- |
   | 0 |	0 |	0 |	4.0 |
   | 1 |	0 |	2 |	4.0 |
   | 2 |	0 |	5 |	4.0 |
   | 3 |	0 |	43 |	5.0 |
   | 4 |	0 |	46 |	5.0 |
   | ... |	... |	... |	... |
   | 100831 |	609 |	9434 |	4.0 |
   | 100832 |	609 |	9461 |	5.0 |
   | 100833 |	609 |	9462 |	5.0 |
   | 100834 |	609 |	9463 |	5.0 |
   | 100835 |	609 |	9503 |	3.0 |

 - Penentuan variabel dan normalisasi

   Sebelum dilakukan pembagian dataset, variabel perlu ditentukan agar data dapat diolah oleh alogritma. Data user dan movie menjadi satu value x dan data rating menjadi value y. Pada value y dilakukan normalisasi agar data menjadi mudah diolah oleh algoritma.
   ```python
   # Membuat variabel x untuk mencocokkan data user dan movie menjadi satu value
   x = encoded_rating_df[['userId', 'movieId']].values
   
   # Membuat variabel y untuk membuat rating dari hasil 
   y = encoded_rating_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

   ```

 - Pembagian dataset.
  
   Pembagian dataset menjadi data train dan test perlu dilakukan untuk menguji model pada data test, dengan model yang telah dilatih pada data train. Dataset diacak terlebih dahulu menggunakan fungsi .sample() dan kemudian dibagi menjadi train dan validation dengan rasio 8:2 secara manual.
   


## Modeling
Pada kasus ini, dataset yang dimiliki adalah data rating film dari pengguna dan data film. Dalam data ratings terdapat rating film dari setiap pengguna. Dataset ini cocok apabila menggunakan pendekatan _Collaborative filtering_ karena data yang dimiliki merupakan pendapat komunitas pengguna dan susunan data sudah sesuai dalam bentuk kolom yang dibutuhkan sehingga tidak diperulakan persiapan data yang rumit. Model membandingkan data rating film antar pengguna untuk mencari hubungan agar model dapat mengidentifikasi film-film yang mirip dan belum pernah ditonton oleh pengguna untuk direkomendasikan.

Pendekatan yang digunakan adalah _Collaborative filtering_ berbasis model deep learning. Model deep learning yang digunakan adalah RecommenderNet dengan langkah-langkah dalam model sebagai berikut, (1) memetakan userId menjadi "user vector" melalui matriks embedding, (2) memetakan movieId menjadi "movie vector" melalui matriks embedding, (3) menghitung hasil dot product antara "user vector" dan "movie vector", untuk mendapatkan skor pasangan antara user dan movie (rating prediksi), (4) melatih embedding melalui gradient descent menggunakan semua pasangan user-movie. Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean square error (RMSE) sebagai metrics evaluation.

Setelah dilakukan training, model dievaluasi dan didapatkan hasil sebagai berikut:

```
train loss:  0.5993366241455078
val loss:  0.6133087873458862
train rmse:  0.19040383398532867
val rmse:  0.2050754725933075
```

Hasil evaluasi dari train dan val memiliki nilai yang mirip sehingga dapat dikatakan bahwa mesin dapat belajar.

Dengan model yang sudah dilakukan training, model memprediksikan top 10 recommendation dari salah satu pengguna. Model akan memberikan rekomendasi film berdasarkan film-film yang diberi rating tinggi oleh pengguna.

Hasil Rekomendasi
```
Showing recommendations for users: 260
===========================
Movies with high ratings from user
--------------------------------
Harold and Maude (1971) : Comedy|Drama|Romance
Titus (1999) : Drama
Ghost World (2001) : Comedy|Drama
Grave of the Fireflies (Hotaru no haka) (1988) : Animation|Drama|War
Triplets of Belleville, The (Les triplettes de Belleville) (2003) : Animation|Comedy|Fantasy
--------------------------------
Genres user likes: {'Animation', 'Comedy', 'War', 'Romance', 'Drama', 'Fantasy'}
--------------------------------
Top 10 movies recommendation
--------------------------------
Lawrence of Arabia (1962) : Adventure|Drama|War
Amadeus (1984) : Drama
Seventh Seal, The (Sjunde inseglet, Det) (1957) : Drama
Touch of Evil (1958) : Crime|Film-Noir|Thriller
Seven Samurai (Shichinin no samurai) (1954) : Action|Adventure|Drama
Celebration, The (Festen) (1998) : Drama
Guess Who's Coming to Dinner (1967) : Drama
Battle of Algiers, The (La battaglia di Algeri) (1966) : Drama|War
Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) : Action|Animation|Drama|Fantasy|Sci-Fi
Three Billboards Outside Ebbing, Missouri (2017) : Crime|Drama
```

Dari hasil rekomendasi, terdapat 10 rekomendasi film yang belum pernah ditonton oleh pengguna. Dari 10 film tersebut, terdapat 9 film rekomendasi yang memiliki genre relevan dengan user dan 1 film rekomendasi yang memiliki genre tidak relevan dengan user yaitu film Touch of Evil (1958). 

## Evaluation
Pada sistem rekomentasi ini, metrik yang digunakan adalah RMSE dan precision. RMSE (Root Mean Square Error) menghitung nilai mean dari semua selisih kuadrat antara hasil benar dan hasil prediksi ratings dan kemudian menghitung akar kuadrat dari hasilnya, dapat dituliskan dengan rumus: 

![image](https://user-images.githubusercontent.com/76271668/153878078-492d9902-679c-4a2b-8726-471cca6fe97d.png)


Precision merupakan perbandingan antara banyaknya rekomendasi yang relevan dengan banyaknya rekomendasi, dengan rumus: 

![image](https://user-images.githubusercontent.com/76271668/153868680-287ea3c5-f149-4626-92c5-7c1129b05d8f.png)

Hasil Training Model (RMSE)

![image](https://user-images.githubusercontent.com/76271668/153881049-ce83cb67-7062-4319-8e24-9de3ad7d097d.png)

Proses training model cukup smooth dan model sedikit konvergen pada epoch sekitar 20. Dari proses ini, diperoleh nilai error akhir sebesar sekitar 0.19 dan error pada data validasi sebesar 0.20. Nilai tersebut cukup baik untuk sistem rekomendasi.

Pada hasil rekomendasi yang didapatkan terdapat 9 film rekomendasi yang memiliki genre relevan dengan user dan 1 film rekomendasi yang memiliki genre tidak relevan dengan user yaitu film Touch of Evil (1958), maka precision yang didapatkan untuk sistem rekomendasi ini adalah 90%.

**---Ini adalah bagian akhir laporan---**

Laporan ini disusun untuk memenuhi tugas proyek akhir untuk kelas Machine Learning Terapan di Dicoding Indonesia.
