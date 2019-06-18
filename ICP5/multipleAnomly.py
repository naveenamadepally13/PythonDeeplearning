import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')

data = pd.concat([df_train['SalePrice'], df_train['GarageArea']], axis=1)
print(data)
data.plot.scatter(x='GarageArea', y='SalePrice');

error = stats.zscore(data)
#print(stats.zscore(data))

data1 = data[(error < 3).all(axis=1)]
#print((error < 3).all(axis=1))
data1.plot.scatter(x='GarageArea', y='SalePrice');

plt.show()