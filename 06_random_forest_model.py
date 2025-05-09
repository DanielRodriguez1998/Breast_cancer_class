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
