from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

# You can add the parameter data_home to wherever to where you want to download your data
dataset = pd.read_csv('CC.csv')
x = dataset.iloc[:,[1,2,3,4]]
y = dataset.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x)

# Apply transform to both the training set and the test set.
x_scaler = scaler.transform(x)


from sklearn import metrics
wcss = []
# ##elbow method to know the number of clusters
for i in range(2,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x_scaler)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(x_scaler, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))

plt.plot(range(1,4),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2,dataset[['TENURE']]],axis=1)
print(finaldf)

from sklearn import metrics
wcss = []
# ##elbow method to know the number of clusters
for i in range(2,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(finaldf)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(finaldf, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))
plt.plot(range(1, 4), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
