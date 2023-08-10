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
#| echo: false

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X = np.concatenate([
    np.random.normal(-2, size=100),
    np.random.normal(2, size=100),
    np.random.normal(7, size=1000)
])
sns.kdeplot(X, bw_adjust=1)
plt.ylabel('Density')
plt.xlabel('X')
plt.title('KDE as Clustering: Three modes in Px')
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
#| echo: false

# inspired by https://jakevdp.github.io/PythonDataScienceHandbook/06.00-figure-code.html

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

rng = np.random.RandomState(42)
centers = [0, 4] + rng.randn(4, 2)

def draw_points(ax, c, factor=1):
    ax.scatter(X[:, 0], X[:, 1], c=c, cmap='viridis',
               s=50 * factor, alpha=0.3)
    
def draw_centers(ax, centers, factor=1, alpha=1.0):
    ax.scatter(centers[:, 0], centers[:, 1],
               c=np.arange(4), cmap='viridis', s=200 * factor,
               alpha=alpha)
    ax.scatter(centers[:, 0], centers[:, 1],
               c='black', s=50 * factor, alpha=alpha)

def make_ax(fig, gs):
    ax = fig.add_subplot(gs)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    return ax
#
#
#
#| echo: false

fig = plt.figure(figsize=(4.55, 5))
ax = fig.add_subplot()
draw_points(ax, 'gray', factor=2)
draw_centers(ax, centers, factor=2)
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
#| echo: false

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# E-step
y_pred = pairwise_distances_argmin(X, centers)
draw_points(axes[0], y_pred)
draw_centers(axes[0], centers)

# M-step
new_centers = np.array([X[y_pred == i].mean(0) for i in range(4)])
draw_points(axes[1], y_pred)
draw_centers(axes[1], centers, alpha=0.3)
draw_centers(axes[1], new_centers)
for i in range(4):
    axes[1].annotate('', new_centers[i], centers[i],
                    arrowprops=dict(arrowstyle='->', linewidth=1))
    

# Finish iteration
centers = new_centers
_ = axes[0].text(0.95, 0.95, "Assign", transform=axes[0].transAxes, ha='right', va='top', size=14)
_ = axes[1].text(0.95, 0.95, "Update", transform=axes[1].transAxes, ha='right', va='top', size=14)
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
#| echo: false

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# E-step
y_pred = pairwise_distances_argmin(X, centers)
draw_points(axes[0], y_pred)
draw_centers(axes[0], centers)

# M-step
new_centers = np.array([X[y_pred == i].mean(0) for i in range(4)])
draw_points(axes[1], y_pred)
draw_centers(axes[1], centers, alpha=0.3)
draw_centers(axes[1], new_centers)
for i in range(4):
    axes[1].annotate('', new_centers[i], centers[i],
                    arrowprops=dict(arrowstyle='->', linewidth=1))
    

# Finish iteration
centers = new_centers
_ = axes[0].text(0.95, 0.95, "Assign", transform=axes[0].transAxes, ha='right', va='top', size=14)
_ = axes[1].text(0.95, 0.95, "Update", transform=axes[1].transAxes, ha='right', va='top', size=14)
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
#| echo: false

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# E-step
y_pred = pairwise_distances_argmin(X, centers)
draw_points(axes[0], y_pred)
draw_centers(axes[0], centers)

# M-step
new_centers = np.array([X[y_pred == i].mean(0) for i in range(4)])
draw_points(axes[1], y_pred)
draw_centers(axes[1], centers, alpha=0.3)
draw_centers(axes[1], new_centers)
for i in range(4):
    axes[1].annotate('', new_centers[i], centers[i],
                    arrowprops=dict(arrowstyle='->', linewidth=1))
    

# Finish iteration
centers = new_centers
_ = axes[0].text(0.95, 0.95, "Assign", transform=axes[0].transAxes, ha='right', va='top', size=14)
_ = axes[1].text(0.95, 0.95, "Update", transform=axes[1].transAxes, ha='right', va='top', size=14)
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
#| echo: false

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('../datasets/netflix/train_ratings_all.csv', header = None)
miss_cong = pd.read_csv('../datasets/netflix/train_y_rating.csv', header = None, names = ['score'])
movies = pd.read_csv('../datasets/netflix/movie_titles.csv', header = None, names = ['year', 'title'])

netflix_X = ratings.iloc[:, :14]
netflix_X.columns = movies['title'][:14]
netflix_Y = miss_cong.iloc[:, 0]

NE_Xtr, NE_Xte, NE_Ytr, NE_Yte = train_test_split(netflix_X, netflix_Y, test_size=0.2, random_state=42)
#
#
#
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
_ = kmeans.fit(NE_Xtr)
#
#
#
#
#
#| output-location: fragment
print(kmeans.cluster_centers_.shape)
#
#
#
#| output-location: fragment
print(kmeans.labels_[:10])
#
#
#
#| output-location: fragment
print(f'{kmeans.inertia_:.2f}')
#
#
#
#
#
#| output-location: fragment
test_labels = kmeans.predict(NE_Xte)
test_labels[:10]
#
#
#
#
#
#
#
pd.DataFrame({'title': kmeans.feature_names_in_,
    'mean_score': NE_Xtr.mean(axis = 0),
    'm_1': kmeans.cluster_centers_[0],
    'm_2': kmeans.cluster_centers_[1]}).set_index('title').head(8).round(2)
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

from sklearn.decomposition import PCA

X_centered = NE_Xtr - NE_Xtr.mean(axis=0)
pca = PCA(n_components=2)
pca.fit(X_centered)

T = pca.transform(X_centered)

c_dict = {0:'Cluster1', 1:'Cluster2', 2:'Cluster3', 3:'Cluster4'}
clusters = np.vectorize(c_dict.get)(kmeans.labels_)
ax = sns.jointplot(x=T[:, 0], y=T[:, 1], hue=clusters, height=5)
ax.set_axis_labels('PC1: Popular Vote', 'PC2: Romance vs. Action', fontsize=10)
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
#| code-fold: true

kmeans = KMeans(n_clusters=4)
kmeans.fit(NE_Xtr)

clusters = np.vectorize(c_dict.get)(kmeans.labels_)
ax = sns.jointplot(x=T[:, 0], y=T[:, 1], hue=clusters, height=5)
ax.set_axis_labels('PC1: Popular Vote', 'PC2: Romance vs. Action', fontsize=10)
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
W_C = []
for K in range(2, 20):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(NE_Xtr)
    W_C.append(kmeans.inertia_)
#
#
#
#| echo: false
plt.figure(figsize=(6, 4))
plt.plot(range(2, 20), W_C, '--bo')
plt.title('KMeans on Netflix: Within Clusters SS vs. K')
plt.ylabel('W(C)')
plt.xlabel('K')
plt.xticks([2,6,10,14,18])
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
#| code-fold: true
#| output-location: fragment

from sklearn.preprocessing import StandardScaler

n = 30
m1 = 2
m2 = 0
sig = np.eye(2)
rng = np.random.RandomState(4)
X1 = rng.multivariate_normal(mean=[-m1, m2], cov=sig, size=n)
X2 = rng.multivariate_normal(mean=[m1, m2], cov=sig, size=n)
X = np.concatenate([X1, X2], axis=0)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
axes[0].set_xlim((-6, 6))
axes[0].set_ylim((-6, 6))
axes[0].set_title('K-means without standardizing')
X_stan = StandardScaler().fit_transform(X)
kmeans_stan = KMeans(n_clusters=2, random_state=0).fit(X_stan)
axes[1].scatter(X_stan[:, 0], X_stan[:, 1], c=1-kmeans_stan.labels_, cmap='viridis')
axes[1].set_xlim((-6, 6))
axes[1].set_ylim((-6, 6))
axes[1].set_title('K-means with standardizing')
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
#| code-fold: true

from sklearn.datasets import make_circles

X_circles, _ = make_circles(n_samples=500, factor=0.5, noise=0.05)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X_circles)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=kmeans.labels_)
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
#| code-fold: true

X_square = np.random.rand(1000, 2)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_square)
plt.scatter(X_square[:, 0], X_square[:, 1], c=kmeans.labels_)
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
#| code-fold: true

centers = np.eye(2) * 5
X_out, _ = make_blobs(n_samples=100, centers=centers, random_state=0)
X_out = np.concatenate([X_out, np.array([[10, 10]])], axis=0)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X_out)
plt.scatter(X_out[:, 0], X_out[:, 1], c=1-kmeans.labels_)
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
#| code-fold: true

X_varied, y = make_blobs(n_samples=[300, 150, 300], cluster_std=[1.0, 2.5, 0.5], random_state=170)

kmeans = KMeans(n_clusters=3, random_state=170).fit(X_varied)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=kmeans.labels_)
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
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1.0, min_samples=10, metric='euclidean')
_ = dbscan.fit(NE_Xtr)
#
#
#
#
#
#| output-location: fragment
print(dbscan.core_sample_indices_.shape)
#
#
#
#| output-location: fragment
print(dbscan.labels_[:10])
clusters, counts = np.unique(dbscan.labels_, return_counts=True)
d = dict(zip(clusters, counts))
print(d)
print(f'no. of noise points: {np.sum(dbscan.labels_ == -1)}')
#
#
#
#
#
#| output-location: fragment
#| error: true
test_labels = dbscan.predict(NE_Xte)
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

from sklearn.decomposition import PCA

X_centered = NE_Xtr - NE_Xtr.mean(axis=0)
pca = PCA(n_components=2)
pca.fit(X_centered)

T = pca.transform(X_centered)

c_dict = {-1: 'Noise', 0:'Cluster1', 1:'Cluster2', 2:'Cluster3', 3:'Cluster4'}
clusters = np.vectorize(c_dict.get)(dbscan.labels_)
ax = sns.jointplot(x=T[:, 0], y=T[:, 1], hue=clusters, height=5)
ax.set_axis_labels('PC1: Popular Vote', 'PC2: Romance vs. Action', fontsize=10)
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
#| code-fold: true

dbscan = DBSCAN(eps=0.2).fit(X_circles)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=dbscan.labels_)
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
#| code-fold: true

dbscan = DBSCAN(eps=0.2).fit(X_square)
plt.scatter(X_square[:, 0], X_square[:, 1], c=dbscan.labels_)
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
#| code-fold: true

dbscan = DBSCAN(eps=1.2).fit(X_out)
plt.scatter(X_out[:, 0], X_out[:, 1], c=dbscan.labels_)
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
#| code-fold: true

dbscan = DBSCAN(eps=0.9).fit(X_varied)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=dbscan.labels_)
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
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

print(X_train.shape)
print(y_train.shape)
#
#
#
y_train[:10]
#
#
#
#| code-fold: true
fig, axes = plt.subplots(1, 3, figsize=(6, 3))
axes[0].imshow(X_train[0], cmap="binary")
axes[1].imshow(X_train[1], cmap="binary")
axes[2].imshow(X_train[2], cmap="binary")
_ = axes[0].axis('off')
_ = axes[1].axis('off')
_ = axes[2].axis('off')
plt.show()
#
#
#
#| echo: false

fnist_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}
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
X_train_flat = X_train.reshape((X_train.shape[0], -1))
print(X_train_flat.shape)
#
#
#
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train_flat)

print(pd.crosstab(y_train, kmeans.labels_).rename(index=fnist_dict))
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
X_train_flat_centered = X_train_flat - X_train_flat.mean(axis=0)
pca = PCA(n_components = 2)
pca.fit(X_train_flat_centered)
T = pca.transform(X_train_flat_centered)
#
#
#
#| code-fold: true
clusters = np.vectorize(fnist_dict.get)(y_train)

sns.scatterplot(x=T[:, 0], y=T[:, 1], hue=clusters, s=10, palette='tab10')
plt.xlabel('PC1')
plt.ylabel('PC2')
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
kmeans = KMeans(n_clusters=10)
kmeans.fit(T)

print(pd.crosstab(y_train, kmeans.labels_).rename(index=fnist_dict))
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
pca = PCA(n_components = 30)
pca.fit(X_train_flat_centered)
T = pca.transform(X_train_flat_centered)

kmeans = KMeans(n_clusters=10)
kmeans.fit(T)

print(pd.crosstab(y_train, kmeans.labels_).rename(index=fnist_dict))
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
#| code-line-numbers: "|4-8|9-13|14|15|16|"
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

stacked_encoder = Sequential([
    Flatten(input_shape=[28, 28]),
    Dense(100, activation="relu"),
    Dense(30, activation="relu"),
])
stacked_decoder = Sequential([
    Dense(100, activation="relu", input_shape=[30]),
    Dense(28 * 28, activation="sigmoid"), # make output 0-1
    Reshape([28, 28])
])
stacked_ae = Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss='mse', optimizer='adam')
history = stacked_ae.fit(X_train, X_train, epochs=20, verbose=0)
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
def show_reconstructions(sae, images, n_images=5):
  reconstructions = sae.predict(images[:n_images], verbose=0)
  fig = plt.figure(figsize=(n_images * 1.5, 3))
  for image_index in range(n_images):
      plt.subplot(2, n_images, 1 + image_index)
      plt.imshow(images[image_index], cmap='binary')
      plt.axis('off')
      plt.subplot(2, n_images, 1 + n_images + image_index)
      plt.imshow(reconstructions[image_index], cmap='binary')
      plt.axis('off')
show_reconstructions(stacked_ae, X_test)
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
T = stacked_encoder.predict(X_train, verbose=0)

print(T.shape)
#
#
#
kmeans = KMeans(n_clusters=10)
kmeans.fit(T)

print(pd.crosstab(y_train, kmeans.labels_).rename(index=fnist_dict))
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

from sklearn.manifold import TSNE

T_te = stacked_encoder.predict(X_test, verbose=False)

tsne = TSNE(init='pca', learning_rate='auto')
X_test_2D = tsne.fit_transform(T_te)
X_test_2D = (X_test_2D - X_test_2D.min()) / (X_test_2D.max() - X_test_2D.min())
#
#
#
#| code-fold: true
#| eval: false

clusters = np.vectorize(fnist_dict.get)(y_test)
sns.scatterplot(x=X_test_2D[:, 0], y=X_test_2D[:, 1], hue=clusters, s=10, palette='tab10')
plt.xlabel('D1')
plt.ylabel('D2')
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
