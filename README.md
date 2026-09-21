# Customer Segmentation and Classification Pipeline

Machine Learning pipeline untuk melakukan **segmentasi nasabah berdasarkan karakteristik demografi dan perilaku transaksi**, kemudian menggunakan hasil segmentasi tersebut sebagai label untuk membangun model klasifikasi.

Proyek ini menggabungkan dua pendekatan:

* **Unsupervised Learning** menggunakan K-Means untuk menemukan pola dan membentuk segmen nasabah.
* **Supervised Learning** menggunakan Decision Tree dan Random Forest untuk memprediksi segmen nasabah baru.

---

## Overview

Dalam konteks perbankan, setiap nasabah memiliki karakteristik dan pola transaksi yang berbeda. Dengan melakukan segmentasi, data tersebut dapat digunakan untuk memahami kelompok nasabah dengan perilaku yang serupa.

Pipeline pada proyek ini dirancang untuk menjawab dua kebutuhan:

1. Menemukan kelompok nasabah berdasarkan pola data yang tersedia.
2. Membangun model yang dapat memprediksi cluster nasabah baru secara otomatis.

Hasil clustering kemudian digunakan sebagai target pada tahap classification.

```text
Raw Dataset
     │
     ▼
Data Preprocessing
     │
     ├── Missing Value Handling
     ├── Categorical Encoding
     └── Standardization
     │
     ▼
K-Means Clustering
     │
     ├── Cluster 0
     └── Cluster 1
     │
     ▼
PCA Visualization
     │
     ▼
Cluster Label
     │
     ▼
Classification Dataset
     │
     ├── Decision Tree
     └── Random Forest
            │
            ▼
      GridSearchCV Tuning
            │
            ▼
      Best Classification Model
```

---

## Business Objective

Segmentasi nasabah dapat membantu perusahaan memahami perbedaan karakteristik pelanggan dan menggunakannya sebagai dasar untuk strategi yang lebih terarah.

Tujuan utama proyek ini adalah:

### Customer Profiling

Mengidentifikasi pola perilaku dan karakteristik nasabah yang memiliki kemiripan.

### Targeted Marketing

Menggunakan karakteristik setiap segmen sebagai dasar untuk menentukan pendekatan pemasaran yang sesuai.

Contoh pendekatan yang dapat dikembangkan dari hasil segmentasi:

* Nasabah dengan saldo relatif tinggi dan aktivitas transaksi rendah dapat diarahkan ke produk simpanan atau investasi.
* Nasabah dengan aktivitas transaksi tinggi dapat diberikan program transaksi, cashback, atau loyalty program.

Rekomendasi tersebut merupakan contoh pemanfaatan hasil segmentasi dan bukan bagian dari model prediksi secara langsung.

### Automated Segmentation

Setelah cluster terbentuk, model klasifikasi digunakan untuk memprediksi label cluster pada data nasabah baru tanpa perlu melakukan proses clustering ulang secara manual.

---

## Dataset

Dataset berisi informasi mengenai karakteristik nasabah, baik dari sisi demografi maupun aktivitas transaksi.

Data yang digunakan melalui beberapa tahap preprocessing sebelum masuk ke model machine learning, termasuk:

* Penanganan missing values
* Encoding fitur kategorikal
* Penghapusan data yang tidak relevan
* Standardisasi fitur numerik
* Pemeriksaan outlier bila diperlukan

Dataset hasil preprocessing kemudian digunakan dalam proses clustering dan classification.

---

# Machine Learning Pipeline

## 1. Clustering

Tahap pertama menggunakan pendekatan **Unsupervised Learning** dengan algoritma K-Means.

### Preprocessing

Beberapa proses yang dilakukan sebelum clustering:

```python
Missing Value Handling
Categorical Encoding
StandardScaler
```

Tujuan standardisasi adalah memastikan setiap fitur memiliki skala yang sebanding sehingga tidak ada fitur tertentu yang mendominasi perhitungan jarak pada K-Means.

### K-Means

K-Means digunakan untuk mengelompokkan nasabah berdasarkan kemiripan karakteristik.

Jumlah cluster yang digunakan pada eksperimen akhir adalah:

```text
k = 2
```

Hasil clustering menghasilkan dua segmen utama.

| Cluster   | Karakteristik                                                                                                                             |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Cluster 0 | Nasabah dengan saldo relatif tinggi, aktivitas transaksi lebih rendah, dan cenderung berasal dari kelompok profesional atau nasabah mapan |
| Cluster 1 | Nasabah dengan aktivitas transaksi lebih tinggi dan cenderung berasal dari kelompok pelajar atau nasabah usia lebih muda                  |

Nama segmen tersebut digunakan sebagai **interpretasi bisnis terhadap pola data**, bukan label asli yang diberikan oleh dataset.

### PCA Visualization

**Principal Component Analysis (PCA)** digunakan untuk mereduksi dimensi data sehingga hasil clustering dapat divisualisasikan dalam ruang dua dimensi.

PCA digunakan untuk membantu melihat apakah kedua cluster memiliki pemisahan yang cukup jelas secara visual.

---

## 2. Inverse Transformation

Setelah proses standardisasi dan clustering selesai, data dikembalikan ke skala aslinya menggunakan inverse transformation.

Tujuannya adalah agar hasil cluster lebih mudah dipahami ketika dianalisis oleh stakeholder bisnis.

Contohnya:

```text
Scaled Balance = 1.42
```

dapat dikembalikan menjadi nilai saldo dalam skala asli sehingga lebih mudah diinterpretasikan.

Dataset hasil proses ini disimpan sebagai:

```text
data_clustering_inverse.csv
```

---

# 3. Classification

Label cluster yang dihasilkan oleh K-Means kemudian digunakan sebagai **target variable** untuk tahap supervised learning.

Dengan pendekatan ini, model classification belajar hubungan antara fitur nasabah dan cluster yang telah dihasilkan sebelumnya.

### Feature Encoding

Fitur kategorikal dikonversi menggunakan One-Hot Encoding:

```python
pd.get_dummies()
```

### Train-Test Split

Dataset dibagi menjadi:

```text
80% Training
20% Testing
```

Stratified splitting digunakan untuk menjaga proporsi masing-masing class pada data training dan testing.

---

## Models

Dua algoritma classification digunakan dalam eksperimen:

### Decision Tree

Decision Tree digunakan sebagai model baseline karena mudah diinterpretasikan dan dapat menunjukkan pola keputusan berdasarkan fitur input.

Model disimpan dalam:

```text
decision_tree_model.h5
```

### Random Forest

Random Forest digunakan untuk meningkatkan stabilitas model melalui kombinasi beberapa decision tree.

Model awal disimpan dalam:

```text
explore_random_forest_classification.h5
```

---

# 4. Hyperparameter Tuning

Random Forest kemudian dioptimalkan menggunakan `GridSearchCV`.

Beberapa parameter yang diuji antara lain:

```text
n_estimators
max_depth
min_samples_split
```

Tujuan tuning adalah mencari kombinasi parameter yang memberikan performa terbaik berdasarkan data training.

Model hasil tuning disimpan sebagai:

```text
tuning_classification.h5
```

---

# Model Evaluation

Performa model classification dievaluasi menggunakan beberapa metrik:

* Accuracy
* Precision
* Recall

Pada data pengujian yang digunakan dalam proyek ini, model classification menghasilkan:

| Metric    | Score |
| --------- | ----: |
| Accuracy  |  1.00 |
| Precision |  1.00 |
| Recall    |  1.00 |

Hasil tersebut menunjukkan bahwa pada **test set yang digunakan dalam eksperimen**, model mampu memprediksi label cluster dengan sangat baik.

Perlu dicatat bahwa label classification pada proyek ini berasal dari hasil K-Means pada dataset yang sama. Oleh karena itu, nilai evaluasi tersebut menggambarkan kemampuan model dalam mereplikasi hasil clustering, bukan membuktikan bahwa model akan selalu mencapai akurasi yang sama pada dataset baru dengan karakteristik berbeda.

---

# Technologies

Proyek ini dibuat menggunakan Python dan beberapa library utama:

| Technology                      | Usage                     |
| ------------------------------- | ------------------------- |
| Python 3                        | Programming language      |
| Pandas                          | Data manipulation         |
| NumPy                           | Numerical computation     |
| Scikit-Learn                    | Machine learning          |
| Matplotlib                      | Data visualization        |
| Seaborn                         | Statistical visualization |
| Joblib                          | Model serialization       |
| Jupyter Notebook / Google Colab | Development environment   |

---

# Repository Structure

```text
.
├── data_clustering.csv
├── data_clustering_inverse.csv
├── PCA_model_clustering.h5
├── model_clustering.h5
├── decision_tree_model.h5
├── explore_random_forest_classification.h5
├── tuning_classification.h5
└── ML_Pipeline_Notebook.ipynb
```

### File Description

| File                                      | Description                                                                       |
| ----------------------------------------- | --------------------------------------------------------------------------------- |
| `data_clustering.csv`                     | Dataset hasil preprocessing dan clustering dalam skala yang telah distandardisasi |
| `data_clustering_inverse.csv`             | Dataset hasil clustering yang telah dikembalikan ke skala asli                    |
| `PCA_model_clustering.h5`                 | Artefak model/proses clustering yang berkaitan dengan PCA                         |
| `model_clustering.h5`                     | Model K-Means                                                                     |
| `decision_tree_model.h5`                  | Model Decision Tree                                                               |
| `explore_random_forest_classification.h5` | Model Random Forest sebelum tuning                                                |
| `tuning_classification.h5`                | Model Random Forest hasil hyperparameter tuning                                   |
| `ML_Pipeline_Notebook.ipynb`              | Notebook utama yang berisi seluruh proses analisis dan pemodelan                  |

---

# How to Run

## 1. Clone Repository

```bash
git clone https://github.com/username/repo-name.git
cd repo-name
```

## 2. Install Dependencies

Pastikan Python 3 telah terpasang, kemudian install library yang digunakan:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

## 3. Open Notebook

Buka notebook menggunakan Jupyter:

```bash
jupyter notebook
```

Kemudian buka:

```text
ML_Pipeline_Notebook.ipynb
```

Alternatifnya, notebook dapat dijalankan menggunakan **Google Colab**.

## 4. Run All Cells

Pastikan dataset berada pada direktori yang sesuai, kemudian jalankan seluruh cell dari preprocessing hingga evaluation.

---

# Workflow Summary

Secara ringkas, proses dalam proyek ini adalah:

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Standardization
   │
   ▼
K-Means Clustering
   │
   ├───────────────┐
   ▼               ▼
Cluster        PCA Visualization
   │
   ▼
Cluster Label
   │
   ▼
One-Hot Encoding
   │
   ▼
Train / Test Split
   │
   ├───────────────┐
   ▼               ▼
Decision Tree   Random Forest
                   │
                   ▼
              GridSearchCV
                   │
                   ▼
             Tuned Model
                   │
                   ▼
              Evaluation
```

---

# Key Takeaways

Proyek ini menunjukkan bagaimana hasil **unsupervised learning** dapat digunakan sebagai dasar untuk membangun model **supervised learning**.

K-Means digunakan terlebih dahulu untuk menemukan struktur cluster pada data nasabah. Setelah cluster terbentuk, hasil tersebut digunakan sebagai target classification menggunakan Decision Tree dan Random Forest.

Pendekatan seperti ini dapat digunakan ketika dataset belum memiliki label segmentasi sebelumnya, tetapi perusahaan ingin membangun sistem yang dapat memberikan label segmentasi secara otomatis pada data baru.

---

# Future Development

Beberapa pengembangan yang dapat dilakukan pada pipeline ini antara lain:

* Menambahkan lebih banyak fitur perilaku transaksi.
* Menguji jumlah cluster menggunakan Silhouette Score atau metode evaluasi lainnya.
* Membandingkan K-Means dengan algoritma clustering lain.
* Menggunakan pipeline preprocessing yang lebih terstruktur dengan `Pipeline` dan `ColumnTransformer`.
* Menambahkan model deployment menggunakan Flask atau FastAPI.
* Membuat dashboard untuk memantau distribusi dan karakteristik setiap segmen.
* Mengintegrasikan model classification ke sistem yang dapat melakukan prediksi secara otomatis.

---

## Author

**Henry**

Student of Informatics Engineering
Interested in Machine Learning, Data Analysis, Software Development, and Information Technology.

---

## License

This project is intended for **learning, experimentation, and portfolio purposes**.
