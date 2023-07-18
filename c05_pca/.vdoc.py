# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np
import pandas as pd

ratings = pd.read_csv('../datasets/netflix/train_ratings_all.csv', header = None)
miss_cong = pd.read_csv('../datasets/netflix/train_y_rating.csv', header = None, names = ['score'])
movies = pd.read_csv('../datasets/netflix/movie_titles.csv', header = None, names = ['year', 'title'])
#
#
#
#| echo: false
print()
#
#
#
#
X = ratings.values[:,:14]

print(X.shape)
```
#
#
#
#| echo: false
print()
#
#
#
#
print(X[:5, :5])
```
#
#
#
#
#
#
#
#
#
#
#
#
# currently..
X.mean(axis=0)
#
#
#
#
# centering X: subtracting the mean from each column
X_centered = X - X.mean(axis=0)

print(X_centered.mean(axis=0))
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| code-line-numbers: "|1|3-4|6-7|"

from sklearn.decomposition import PCA

# instantiating PCA object
pca = PCA()

# performing PCA
pca.fit(X_centered)
```
#
#
#
#| echo: false

from sklearn.decomposition import PCA

# instantiating PCA object
pca = PCA()

# performing PCA
_ = pca.fit(X_centered)
#
#
#
#
#
#
#
#| echo: false

# Helper function for better pandas styling
def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color
#
#
#
#
#
W = pca.components_.T

print(W.shape)
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
means = pd.DataFrame(X.mean(axis = 0))
means.columns = ['mean_rating']
loadings2 = pd.DataFrame(W[:, :2], columns = ['PC0', 'PC1'])
first_2_PCs = pd.concat([movies[:14]['title'], means, loadings2], axis = 1).set_index('title')

first_2_PCs.head(10).style.applymap(color_negative_red).format("{:.2f}")
#
#
#
#
#
#
#
#
#
#
#
#
import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.scatterplot(x='PC0', y='PC1', size='mean_rating', data=first_2_PCs)
for i, point in first_2_PCs.iterrows():
    ax.text(point['PC0'], point['PC1'], str(i))
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
T = X_centered @ W # make sure this is the same as pca.transform(X_centered)

print(T.shape)
#
#
#
#
ax = sns.jointplot(x=T[:,0], y=T[:,1])
ax.set_axis_labels('PC1: Popular Vote', 'PC2: Romance vs. Action', fontsize=16)
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
PC1_top_100 = T[:, 0].argsort()[-100:]
print('Top PC1:', miss_cong.iloc[PC1_top_100, :].groupby('score').size())
```
#
#
#
#
#
#
PC1_bottom_100 = T[:, 0].argsort()[:100]
print('Bottom PC1:', miss_cong.iloc[PC1_bottom_100, :].groupby('score').size())
```
#
#
#
#
#
#
#
#
#
#
#
#
#
PC2_top_100 = T[:, 1].argsort()[-100:]
print('Top PC2:', miss_cong.iloc[PC2_top_100, :].groupby('score').size())
```
#
#
#
PC2_bottom_100 = T[:, 1].argsort()[:100]
print('Bottom PC2:', miss_cong.iloc[PC2_bottom_100, :].groupby('score').size())
#
#
#
