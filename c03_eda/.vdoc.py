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
#| output-location: fragment

import pandas as pd

df = pd.read_csv('../datasets/drugs.csv')

print(df.head())
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
#
#
#
#| output-location: fragment

import json

data = dict()

with open('../datasets/test.json') as f:
    data=json.load(f)

data
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
#
#
#
#| output-location: fragment

with open('../datasets/test.txt') as f:
    lines = f.readlines(1000)

lines
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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| output-location: fragment

from bs4 import BeautifulSoup

with open('../datasets/test.html') as f:
    soup = BeautifulSoup(f, 'html.parser')

print(soup.prettify())
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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

url = 'https://en.wikipedia.org/wiki/The_Beatles_discography'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')
#
#
#
#
#
#
#
#
#
#
def get_release_details(release_col):
    release_date = None
    release_label = None
    if release_col is not None:
        release_list = release_col.find('ul')
        if release_list is not None:
            release_list_elements = release_list.find_all('li')
            for element in release_list_elements:
                element_text = element.get_text()
                if element_text.startswith('Released: '):
                    release_date = re.search('Released: ([0-9a-zA-Z ]+)',\
                                             element_text).group(1)
                if element_text.startswith('Label: '):
                    release_label = re.search('Label: ([0-9a-zA-Z,\(\) ]+)', \
                                              element_text).group(1)
    return release_date, release_label
#
#
#
#
#
#
#
#
#
#
albums = dict()
id = 0
albums[id] = dict()
tables = soup.find_all('table')
for table in tables:
    caption = table.find('caption')
    if caption is not None:
        header = caption.get_text()
        if re.match(re.compile('^List of(.+?)albums'), header):
            rows = table.find_all('tr')
            for row in rows:
                title_col = row.find('th')
                if title_col is not None and 'scope' in title_col.attrs and\
                title_col.attrs['scope'] == 'row':            
                    title_cell = title_col.find('a')
                    if title_cell is not None and title_cell.attrs is not None and\
                    'title' in title_cell.attrs:
                        albums[id]['name'] = title_cell.attrs['title']
                        release_col = row.find('td')
                        release_date, release_label = get_release_details(release_col)
                        if release_date is not None or release_label is not None:
                            albums[id]['release_date'] = release_date
                            albums[id]['release_label'] = release_label
                            id += 1
                            albums[id] = dict()
#
#
#
#
#
#
#
#
#
#
albums_df = pd.DataFrame.from_dict(albums, orient ='index')
albums_df.head(5)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
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
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
X = np.random.chisquare(5, 1000)
#
#
#
sns.boxplot(y = X)
plt.ylabel('X')
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
sns.swarmplot(y = X)
plt.ylabel('X')
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
sns.distplot(X, kde = False)
plt.ylabel('Frequency')
plt.xlabel('X')
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
sns.distplot(X, hist = False)
plt.ylabel('Density')
plt.xlabel('X')
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
#| code-fold: true

plt.figure(figsize =(14, 5))
plt.subplot(1, 3, 1)
sns.kdeplot(X, bw=0.1)
plt.ylabel('Density')
plt.xlabel('X')
plt.title('Width=0.1: Too narrow')
plt.subplot(1, 3, 2)
sns.kdeplot(X, bw=1)
plt.ylabel('Density')
plt.xlabel('X')
plt.title('Width=1: About right')
plt.subplot(1, 3, 3)
sns.kdeplot(X, bw=10)
plt.ylabel('Density')
plt.xlabel('X')
plt.title('Width=10: Too wide')
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
b0 = 2
b1 = 3
Y = X * b1 + b0 + np.random.normal(0, 10, 1000)
sns.scatterplot(X, Y)
plt.ylabel('Density')
plt.xlabel('X')
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
mean = np.mean(X)
median = np.median(X)
hist, _ = np.histogram(X, bins=range(20))
mode = list(range(20))[hist.argsort()[::-1][0]]
sns.distplot(X, bins = range(20), kde = False)
plt.plot([mean, mean], [0, 160], linewidth=2, color='r')
plt.plot([median, median], [0, 160], linewidth=2, color='g')
plt.plot([mode, mode], [0, 160], linewidth=2, color='b')
plt.legend({'Mean':mean,'Median':median,'Mode':mode})
plt.xlabel('X')
plt.ylabel('Frequency')
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
print(f'90th percentile: {np.percentile(X, 90) :.2f}')
print(f'Range: {np.max(X) - np.min(X) :.2f}')
print(f'IQR: {np.percentile(X, 75) - np.percentile(X, 25) :.2f}')
print(f'Variance: {np.var(X) :.2f}')
print(f'Standard Deviation: {np.std(X) :.2f}')
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
#
#
#
#
from scipy import stats

print(f'Skewness: {stats.skew(X) :.2f}')
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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
tips = sns.load_dataset('tips')
_ = plt.figure(figsize =(5, 5))
_ = sns.pairplot(tips)
plt.show()
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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
