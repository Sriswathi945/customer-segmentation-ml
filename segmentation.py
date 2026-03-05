import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("customers.csv")

model = KMeans(n_clusters=3)
data["segment"] = model.fit_predict(data[["purchase_amount"]])

print(data.head())
