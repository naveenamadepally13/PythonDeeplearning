import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import seaborn as sns

#read dataset
dataset = pd.read_csv('CC.csv')

#assign
x_train = dataset.iloc[:1000,[1,3,13,14]]
print((x_train == 0).sum())

bfMean = x_train['BALANCE'].mean()
x_train['BALANCE'] = x_train['BALANCE'].replace(0, bfMean)

bfMean = x_train['PURCHASES'].mean()
x_train['PURCHASES'] = x_train['PURCHASES'].replace(0, bfMean)

bfMean = x_train['CREDIT_LIMIT'].mean()
x_train['CREDIT_LIMIT'] = x_train['CREDIT_LIMIT'].replace(0, bfMean)

bfMean = x_train['PAYMENTS'].mean()
x_train['PAYMENTS'] = x_train['PAYMENTS'].replace(0, bfMean)

print((x_train== 0).sum())

df = x_train

#Preprocessing the data
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
X_scaled_array = scaler.transform(x_train)
X_scaled = pd.DataFrame(X_scaled_array, columns = x_train.columns)


from sklearn import metrics
wcss = []
# ##elbow method to know the number of clusters
for i in range(2,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x_train)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(x_train, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()


