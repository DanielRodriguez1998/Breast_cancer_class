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
    print("✅ ¡metadata y expression perfectamente alineados!")
else:
    print("⚠ Los índices no coinciden aún, revisa manualmente.")
