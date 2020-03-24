from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

proir = pd.read_csv("./data/order_products__prior.csv")

product = pd.read_csv("./data/products.csv")

ordern = pd.read_csv("./data/orders.csv")

aixixs = pd.read_csv("./data/aisles.csv")

_mg = pd.merge(proir, product, on=['product_id', 'product_id'])

_mg = pd.merge(_mg, ordern, on=["order_id", 'order_id'])

mt = pd.merge(_mg, aixixs, on=['aisle_id', 'aisle_id'])

#print(mt.head(10))

cross = pd.crosstab(mt['user_id'], mt['aisle'])

#cross.head(10)

# 主成分提取
pc = PCA(n_components=0.9)

data = pc.fit_transform(cross)

#print(data)

# 假设分为4个类别
x = data[:500]
# print(x.shape)

km = KMeans(n_clusters=4)

km.fit(x)

k_pre = km.predict(x)


# 显示聚类结果
plt.figure(figsize=(10, 10))

# 建立四个颜色列表
colored = ["orange", "green", "blue", "purple"]
colr = [colored[i] for i in k_pre]
plt.scatter(x[:, 1], x[:, 20], color=colr)

plt.xlabel("1")
plt.ylabel("2")
plt.show()

# 评判聚类效果
silhouette_score(x, k_pre)