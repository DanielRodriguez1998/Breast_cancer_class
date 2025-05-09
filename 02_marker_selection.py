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
