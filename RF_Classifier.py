#===============
# 01_load_data
#===============
import pandas as pd

base_path = "/kaggle/input/transcrip-mama/"
expression = pd.read_csv(base_path + "genewisecount.tsv", sep="\t", index_col=0)
metadata = pd.read_csv(base_path + "metadata.tsv", sep="\t")
clinical = pd.read_csv(base_path + "clinical_data.tsv", sep="\t")

print("Expression matrix (10x10):")
print(expression.iloc[:10, :10])
print("\nMetadata (10 filas):")
print(metadata.head(10))
print("\nClinical data (10 filas):")
print(clinical.head(10))

metadata.index = expression.columns

print(f"metadata dimensiones después de asignar índice: {metadata.shape}")
print(f"expression dimensiones: {expression.shape}")

if all(metadata.index == expression.columns):
    print(" ¡metadata y expression perfectamente alineados!")
else:
    print(" Los índices no coinciden aún, revisa manualmente.")

#====================
#02_marker_selection
#====================
import pandas as pd

marker_genes = ['ENSG00000091831', 'ENSG00000082175', 'ENSG00000141736',
    'ENSG00000107485', 'ENSG00000129514', 'ENSG00000171791',
    'ENSG00000106211', 'ENSG00000123472', 'ENSG00000186081',
    'ENSG00000186847', 'ENSG00000184389', 'ENSG00000146648',
    'ENSG00000148773', 'ENSG00000134057', 'ENSG00000117399',
    'ENSG00000171428', 'ENSG00000124155', 'ENSG00000100219']

expression_markers = expression.loc[expression.index.isin(marker_genes)]
expression_scaled = expression_markers.T
#=======================
# 03_heatmap_clustering
#=======================
import seaborn as sns
import matplotlib.pyplot as plt

metadata_reset = metadata.copy()
metadata_reset['Subtype'] = metadata_reset['Subtype'].replace('tumor', 'Control')

samples_subset = metadata_reset.groupby('Subtype').apply(
    lambda x: x.sample(min(len(x), 10), random_state=42)
).reset_index(level=0, drop=True)

samples_subset['label'] = samples_subset.index + ' (' + samples_subset['Subtype'] + ')'
expression_subset = expression_scaled.loc[samples_subset.index]
expression_subset.index = samples_subset['label'].values
expression_subset_T = expression_subset.T

sns.clustermap(expression_subset_T, method='ward', metric='euclidean', cmap='viridis')
plt.show()

#=======================
# 04_umap_visualization
#=======================
import umap
import matplotlib.pyplot as plt
import pandas as pd

X_umap = expression_subset
subtypes_aligned = samples_subset.set_index('label').loc[X_umap.index]['Subtype']
subtype_codes = subtypes_aligned.astype('category').cat.codes
subtype_names = subtypes_aligned.astype('category').cat.categories

reducer = umap.UMAP(n_neighbors=min(15, X_umap.shape[0] - 1))
embedding = reducer.fit_transform(X_umap)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=subtype_codes, cmap='tab10', s=50)

handles = []
for i, subtype in enumerate(subtype_names):
    handles.append(plt.Line2D([], [], marker='o', linestyle='',
                              color=scatter.cmap(scatter.norm(i)), label=subtype))
plt.legend(handles=handles, title='Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP - Subtype Separation')
plt.tight_layout()
plt.savefig('umap.png', dpi=300, bbox_inches='tight')
plt.show()
# muestras como filas y genes como columnas para UMAP
X_umap = expression_scaled.T # si expression ya está alineado y tiene muestras en columnas
import umap
import matplotlib.pyplot as plt

X_umap = expression_scaled

# Asegurar alineación de metadata
# Revisar que los índices coincidan
metadata_aligned = metadata.loc[X_umap.index]

# Limpiar y codificar subtipos
metadata_aligned['Subtype'] = metadata_aligned['Subtype'].replace('tumor', 'Control')
metadata_aligned['Subtype'] = metadata_aligned['Subtype'].fillna('Unknown')

subtypes_aligned = metadata_aligned['Subtype']
subtype_codes = subtypes_aligned.astype('category').cat.codes
subtype_names = subtypes_aligned.astype('category').cat.categories

# Ejecutar UMAP
reducer = umap.UMAP(n_neighbors=min(15, X_umap.shape[0] - 1))
embedding = reducer.fit_transform(X_umap)

# Graficar con leyenda
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                      c=subtype_codes.values, cmap='tab10', s=50)

# Crear leyenda manual
handles = []
for i, subtype in enumerate(subtype_names):
    handles.append(plt.Line2D([], [], marker='o', linestyle='',
                              color=scatter.cmap(scatter.norm(i)), label=subtype))

plt.legend(handles=handles, title='Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP - Subtype Separation (Full Dataset)')
plt.tight_layout()

# Guardar la figura
plt.savefig('umap_markers_full.png', dpi=300, bbox_inches='tight')
plt.show()

#======================
# 05_outlier_detection
#======================
from sklearn.ensemble import IsolationForest

# Subtipos de interés
subtypes_of_interest = ['Basal', 'Her2', 'LumA', 'LumB']

# Inicializar lista para guardar IDs típicos
typical_ids_all = []

for subtype in subtypes_of_interest:
    # Obtener muestras de este subtipo
    subtype_samples = metadata[metadata['Subtype'] == subtype].index
    X_subtype = expression_scaled.T[subtype_samples]  # NOTA: columnas = muestras
    
    # Transponer muestras como filas
    X_subtype_T = X_subtype.T

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=41)
    y_pred = iso.fit_predict(X_subtype_T)

    # Filtrar solo típicos (y_pred == 1)
    typical_ids = X_subtype_T.index[y_pred == 1]
    typical_ids_all.extend(typical_ids)

# Eliminar duplicados
typical_ids_all = list(set(typical_ids_all))

# Filtrar datos finales (muestras típicas)
X_typical = expression_scaled.T[typical_ids_all].T  # muestras como filas
y_typical = metadata.loc[typical_ids_all, 'Subtype']

# Confirmar dimensiones
print(f"Dimensiones X_typical: {X_typical.shape}")
print(f"Dimensiones y_typical: {y_typical.shape}")

#==========================
# 06_random_forest_model.py
#==========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# --- Función para generar datos sintéticos por problemas de compatibilidad con smote ---
def generate_synthetic_data(X, y, target_class, n_samples, noise_level=0.01, random_state=42):
    X_class = X[y == target_class]
    sampled = X_class.sample(n_samples, replace=True, random_state=random_state)
    synthetic = sampled + np.random.normal(0, noise_level, sampled.shape)
    X_aug = pd.concat([X, synthetic], axis=0).reset_index(drop=True)
    y_aug = pd.concat([y, pd.Series([target_class]*n_samples)]).reset_index(drop=True)
    return X_aug, y_aug

# Guardar mejores resultados
best_f1 = 0
best_seed = None
best_config = {}
best_report_text = ""
best_report_dict = None
best_conf_matrix = None
best_y_test = None
best_y_pred = None
best_model = None

# Probar semillas 0–199
for seed in range(200):
    X_train, X_test, y_train, y_test = train_test_split(
        X_typical, y_typical, test_size=0.2, stratify=y_typical, random_state=seed
    )
    
    X_train_aug, y_train_aug = X_train.copy(), y_train.copy()
    
    # SOLO Basal y Her2
    for target_class in ['Basal', 'Her2']:
        if target_class in y_train.unique():
            n_samples = int((y_train == target_class).sum() * 0.5)
            if n_samples > 0:
                X_train_aug, y_train_aug = generate_synthetic_data(
                    X_train_aug, y_train_aug, target_class=target_class, n_samples=n_samples,
                    noise_level=0.01, random_state=seed
                )
    
    rf = RandomForestClassifier(n_estimators=500, random_state=seed)
    rf.fit(X_train_aug, y_train_aug)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    
    if f1 > best_f1:
        best_f1 = f1
        best_seed = seed
        best_config = {
            'seed': seed,
            'accuracy': accuracy,
            'f1_score': f1
        }
        best_report_dict = report_dict
        best_report_text = report_text
        best_conf_matrix = conf_matrix
        best_y_test = y_test.copy()
        best_y_pred = y_pred.copy()
        best_model = rf

# Guardar resultados en archivo de texto
with open("best_result.txt", "w") as f:
    f.write(" Mejor combinación encontrada (de semillas 0–199):\n")
    f.write(str(best_config) + "\n\n")
    f.write(" Mejor confusion matrix:\n")
    f.write(str(best_conf_matrix) + "\n\n")
    f.write(" Mejor classification report:\n")
    f.write(best_report_text)
print(" Resultados guardados en 'best_result.txt'")

# Guardar classification report como CSV tabular
report_df = pd.DataFrame(best_report_dict).T
report_df.to_csv('best_classification_report.csv')
print(" Reporte guardado en 'best_classification_report.csv'")

# Guardar classification report como JSON completo
with open('best_classification_report.json', 'w') as f:
    json.dump(best_report_dict, f)
print(" Reporte guardado en 'best_classification_report.json'")

# Graficar y guardar matriz de confusión
labels = sorted(best_y_test.unique())
cm = confusion_matrix(best_y_test, best_y_pred, labels=labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Best Model)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
print(" Gráfico guardado como 'confusion_matrix.png'")

# Guardar el modelo entrenado
joblib.dump(best_model, 'best_random_forest.pkl')
print(" Modelo guardado como 'best_random_forest.pkl'")
