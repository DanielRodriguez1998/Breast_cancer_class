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
