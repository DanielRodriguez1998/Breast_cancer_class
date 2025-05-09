# Breast Cancer Transcriptome Analysis

This repository contains scripts to analyze breast cancer RNA-seq data, focusing on subtype classification, clustering, dimensionality reduction, and machine learning modeling.

---

## 📂 Scripts

### 1️⃣ `load_data.py`
- Load gene expression (`genewisecount.tsv`), metadata (`metadata.tsv`), and clinical data (`clinical_data.tsv`).
- Check dataset dimensions and alignment.
- Align metadata indices to match expression columns.

### 2️⃣ `marker_selection.py`
- Select 18 immune-related marker genes (by Ensembl IDs).
- Scale and transpose the expression matrix.
- Filter metadata and create combined sample labels.
- Prepare data for clustering and visualization.

### 3️⃣ `heatmap_clustering.py`
- Perform hierarchical clustering (`clustermap`) on selected markers.
- Visualize subtype patterns as a heatmap.
- Save heatmap plots for exploratory analysis.

### 4️⃣ `umap_visualization.py`
- Perform UMAP dimensionality reduction.
- Visualize subtype separation on 2D scatter plots.
- Generate two UMAP plots:
  - One for representative sample subsets.
  - One for the full dataset.
- Save UMAP figures as PNG files.

### 5️⃣ `outlier_detection.py`
- Use `IsolationForest` to detect and exclude outlier samples per subtype (`Basal`, `Her2`, `LumA`, `LumB`).
- Filter the dataset to retain only typical (non-outlier) samples.
- Report final dataset dimensions.

### 6️⃣ `random_forest_model.py`
- Train a Random Forest classifier over 200 random seeds.
- Apply synthetic data augmentation for minority classes (`Basal` and `Her2`).
- Select the best model based on weighted F1-score.
- Save:
  - Best classification report (`CSV`, `JSON`)
  - Confusion matrix plot (`PNG`)
  - Best trained model (`PKL`)
  - Summary text report (`TXT`)

---

## 🚀 How to Run

- Use Kaggle or Jupyter Notebook to execute scripts in order.
- Make sure to install required packages:
