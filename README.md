# Customer Segmentation and Classification Pipeline

Machine Learning project that segments customers from transaction and demographic data, then trains classification models to reproduce the resulting cluster labels for new records.

The project combines **unsupervised learning** (K-Means) and **supervised learning** (Decision Tree and Random Forest) in a single workflow.

## Project Overview

The workflow is divided into two stages:

1. **Customer segmentation** identify customer groups with similar characteristics using K-Means.
2. **Cluster classification** use the cluster labels as targets for supervised models.

This approach is useful when a dataset does not already contain a business segmentation label, but the resulting clusters need to be assigned consistently to future records.

## Pipeline

```text
Raw transaction data
        │
        ▼
Data cleaning
        │
        ├── Missing values
        ├── Duplicate rows
        ├── ID / date / IP-related columns removed
        └── Categorical encoding
        │
        ▼
Outlier handling + StandardScaler
        │
        ▼
K-Means clustering
        │
        ├── Silhouette-based k selection
        └── Cluster interpretation
        │
        ▼
Inverse transformation
        │
        ▼
Cluster labels (Target)
        │
        ▼
One-Hot Encoding
        │
        ▼
80/20 Stratified split
        │
        ├── Decision Tree
        └── Random Forest
               │
               ▼
         GridSearchCV tuning
               │
               ▼
         Model evaluation
```

## Dataset

The original dataset contains **2,537 rows and 16 columns**. After missing value removal, duplicate removal, feature selection, and outlier handling, **1,945 records** are used for clustering and classification.

The main numerical features are:

- `TransactionAmount`
- `CustomerAge`
- `TransactionDuration`
- `LoginAttempts`
- `AccountBalance`

Categorical features include transaction type, location, channel, customer occupation, and the derived `AgeGroup`.

### Preprocessing

The notebook performs the following steps:

- Check and remove missing rows.
- Remove duplicate records.
- Remove columns related to identifiers, dates, and IP addresses that are not used for modeling.
- Encode categorical variables with `LabelEncoder`.
- Create `AgeGroup` using `pd.qcut`.
- Remove numerical outliers using the IQR rule.
- Standardize numerical features with `StandardScaler`.

After preprocessing, the modeling dataset contains **1,945 records**.

## Clustering

### K-Means

K-Means is used to group customers according to similarities in the processed feature space.

The notebook evaluates cluster counts from **2 to 9** using the silhouette metric. The selected configuration is:

```text
Number of clusters : 2
Random state       : 42
Silhouette score   : 0.5722
```

The two clusters are then analyzed using their numerical ranges and dominant categorical values.

### Cluster profile

| Cluster | Avg. Transaction | Avg. Age | Avg. Duration | Avg. Balance | Dominant Occupation | Dominant Channel | Dominant Location |
|---|---:|---:|---:|---:|---|---|---|
| 0 | 255.55 | 45.06 | 121.12 s | 5,142.17 | Doctor | Branch | Charlotte |
| 1 | 258.15 | 44.33 | 117.30 s | 5,058.81 | Student | Branch | Tucson |

These labels are **interpretations of patterns in this dataset**, not predefined customer categories. The difference between the clusters is relatively small for some numerical features, so the business interpretation should be considered together with the full dataset rather than from a single variable.

### Clustering visualization

#### Silhouette-based cluster selection

![Silhouette Elbow](assets/02-silhouette-elbow.png)

#### PCA projection of the clusters

![PCA Clusters](assets/03-pca-clusters.png)

The PCA plot is used for visualization only. The K-Means model itself is trained on the preprocessed feature space.

## Classification

The resulting cluster label is renamed to `Target` and used as the target variable for the classification stage.

The dataset is split using a stratified 80/20 split:

```text
Training samples : 1,556
Test samples     :   389
```

The class distribution is:

```text
Cluster 0 : 980
Cluster 1 : 965
```

### Models

Two baseline models are trained:

- Decision Tree Classifier
- Random Forest Classifier

Random Forest is then tuned with `GridSearchCV` using 5-fold cross-validation.

Hyperparameters searched:

```text
n_estimators      : [50, 100, 200]
max_depth         : [None, 10, 20]
min_samples_split : [2, 5]
```

The selected parameter combination from the notebook is:

```text
n_estimators      = 50
max_depth         = None
min_samples_split = 2
```

The best cross-validation accuracy reported by `GridSearchCV` is **1.00**.

## Evaluation

On the held-out test set, the notebook reports:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |
| Tuned Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |

![Classification Report](assets/04-classification-report.png)

The reported 1.00 scores apply to this specific test split of 389 records. Since the target labels are generated by the K-Means stage, the classification task measures how well the supervised models reproduce the cluster assignments. It should not be interpreted as proof that the same performance will hold on unseen real-world data.

## Exploratory Analysis

The notebook also contains exploratory analysis before modeling.

![Correlation Matrix](assets/01-correlation-matrix.png)

The correlation matrix shows that most numerical features have weak linear relationships with one another, while `CustomerAge` and `AccountBalance` have the strongest correlation among the numerical variables at approximately **0.32** in this dataset.

## Repository Structure

```text
.
├── assets/
│   ├── 01-correlation-matrix.png
│   ├── 02-silhouette-elbow.png
│   ├── 03-pca-clusters.png
│   ├── 04-classification-report.png
│
├── data_clustering.csv
├── data_clustering_inverse.csv
│
├── model_clustering.h5
├── PCA_model_clustering.h5
├── decision_tree_model.h5
├── explore_random_forest_classification.h5
├── tuning_classification.h5
│
├── [Clustering]_Submission_Akhir_BMLP_Muhammad_Henry_Alifianto.ipynb
├── [Klasifikasi]_Submission_Akhir_BMLP_Muhammad_Henry_Alifianto.ipynb
│
├── [clustering]_submission_akhir_bmlp_muhammad_henry_alifianto.py
└── [klasifikasi]_submission_akhir_bmlp_muhammad_henry_alifianto.py
```

### Saved model files

The model files use a `.h5` extension in the submission package, but they are serialized with **Joblib/scikit-learn**, not stored as HDF5 models.

- `model_clustering.h5` — K-Means model used for the main clustering result.
- `PCA_model_clustering.h5` — K-Means model trained on a PCA-reduced representation.
- `decision_tree_model.h5` — Decision Tree classifier.
- `explore_random_forest_classification.h5` — baseline Random Forest classifier.
- `tuning_classification.h5` — `GridSearchCV` object containing the tuned Random Forest model.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

### 2. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

### 3. Open the notebooks

For the clustering stage:

```text
[Clustering]_Submission_Akhir_BMLP_Muhammad_Henry_Alifianto.ipynb
```

For the classification stage:

```text
[Klasifikasi]_Submission_Akhir_BMLP_Muhammad_Henry_Alifianto.ipynb
```

The clustering notebook should be completed first because its output dataset (`data_clustering.csv` / `data_clustering_inverse.csv`) is used by the classification notebook.

## Limitations and Notes

A few points are important when interpreting the results:

- K-Means is distance-based, so feature scaling has a direct effect on the clusters.
- The clustering notebook uses `LabelEncoder` for categorical variables. For a production-grade segmentation system, one-hot encoding, target-aware approaches, or other representations should be evaluated depending on the modeling objective.
- The cluster names such as "Senior", "Conservative", "Student", or "Dynamic" are interpretations of the observed dataset and are not ground-truth labels.
- The classification model learns the labels produced by K-Means. Therefore, a high classification score mainly shows that the cluster assignments can be reproduced from the available features.
- The notebooks and saved models are intended for learning and portfolio demonstration rather than direct production deployment.

## Possible Next Steps

- Compare K-Means with other clustering algorithms.
- Evaluate cluster stability across multiple random seeds.
- Use more suitable categorical feature representations before clustering.
- Add confusion matrix and feature-importance analysis to the classification stage.
- Wrap preprocessing and prediction into a single scikit-learn pipeline.
- Expose the trained model through an API using Flask or FastAPI.
- Build a dashboard for customer segment exploration.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook / Google Colab

## Author

**Muhammad Henry Alifianto**

Informatics Engineering Student  
Interests: Machine Learning, Data Analysis, Software Development, and Information Technology.
