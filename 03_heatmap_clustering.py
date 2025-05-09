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
