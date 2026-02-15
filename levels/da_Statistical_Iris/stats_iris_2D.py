import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from etc_utils.data_harvest_io import load_uci_data
from etc_utils.viz_bloks import smart_show

# 1. Load the UCI Iris data
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
X, y, y_labels = load_uci_data.load_uci_data(returnData=True, plotData=False)
X_std = StandardScaler().fit_transform(X)

# 3. PCA Analysis
pca = PCA(n_components=2)  # Starting with 2D for clear cluster visualization
X_pca = pca.fit_transform(X_std)

# Mapping PCA back to original features to see "influence"
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
print("--- PC Loadings (Weight Influence) ---")
print(loadings)

# 4. Visualization: 2D Projection and Density
plt.figure(figsize=(8, 8))  ## (14, 6)

# Subplot 1: Scatter with variety
plt.subplot(1, 1, 1)

# plt.axis('equal')
xyLim = [-3,3]
plt.xlim(xyLim)
plt.ylim(xyLim)

x1=X_pca[:, 0]
y2=X_pca[:, 1]
xx=np.unique(x1)
yy=np.unique(y2)
ny2 = yy.size
xx2=xx[xx>0.5]
xx3=xx[xx>0.5]
sns.scatterplot(x=x1, y=y2, hue=y, style=y, palette='gist_rainbow', s=100)
# Setosa:
xx1 = -0.98+np.zeros((ny2,))
# Versicolor:
yy2 = (4*xx2 - 3.5)/2
# Virginica:
yy3 = (-2.2*xx3 + 1.7)/0.5
plt.plot(xx1,yy,'b')
plt.plot(xx2,yy2,'c')
plt.plot(xx3,yy3,'g')

plt.title('2D PCA Projection: Variety Clusters')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

plt.tight_layout()
# plt.show()
smart_show.smart_show(fgSaveFigures=True,
                      selectedFigures={1: "stats_iris_2D"})
