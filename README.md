# Breast Cancer Transcriptome Classifier

This repository contains a complete data analysis and classification pipeline based on gene expression profiles from breast cancer samples. It includes preprocessing, visualization (heatmaps, UMAP), outlier detection, and a classification model using Random Forest with synthetic augmentation.

## Dataset

Folder: `/kaggle/input/transcrip-mama/`

- `genewisecount.tsv`: Gene expression matrix (genes as rows, samples as columns).
- `metadata.tsv`: Sample annotations (subtype, condition, etc.).
- `clinical_data.tsv`: Additional clinical data for patients.
  (Information obtained from TCGA)
## Pipeline Steps

### 1. **Data Loading**
- Load expression, metadata, and clinical data.
- Ensure metadata index matches expression columns.

### 2. **Marker Selection**
- Select a predefined set of marker genes (`ENSG...` IDs).
- Filter and scale expression data for these markers.

### 3. **Heatmap & Clustering**
- Sample 10 representative samples per subtype.
- Generate heatmap of marker expression using hierarchical clustering.

### 4. **UMAP Visualization**
- Dimensionality reduction using UMAP.
- Visualize subtype separation based on marker gene expression.
- Saves plots:
  - `umap.png`: Subset of samples.
  - `umap_markers_full.png`: All samples.

### 5. **Outlier Detection**
- Apply `IsolationForest` on each subtype.
- Keep only "typical" samples (non-outliers).
- These samples are used for model training.

### 6. **Random Forest Classification**
- Train `RandomForestClassifier` on typical samples.
- Uses synthetic data augmentation (custom method, not SMOTE).
- Trains using multiple seeds (0–199) and selects the best seed based on weighted F1 score.

## Outputs

- `best_result.txt`: Best configuration, confusion matrix, and classification report.
- `best_classification_report.csv`: Tabular report (precision, recall, F1).
- `best_classification_report.json`: Full classification report in JSON.
- `confusion_matrix.png`: Visual representation of classification performance.
- `best_random_forest.pkl`: Trained model serialized with `joblib`.
