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
