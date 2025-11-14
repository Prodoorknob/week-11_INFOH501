import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from time import time


def kmeans(X, k):
    """
    Performs k-means clustering on a numerical NumPy array X.
 
    """
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels_

    return (centroids, labels)


diamonds_df = sns.load_dataset('diamonds')
diamonds_numeric = diamonds_df.select_dtypes(include=np.number)

def kmeans_diamonds(n, k):
    
    data_subset = diamonds_numeric.iloc[:n]
    # Convert dataframe to array
    data_array = data_subset.values

    return kmeans(data_array, k)


def kmeans_timer(n, k, n_iter=5):

    run_times = []
    for _ in range(n_iter):

        start_time = time()
        kmeans_diamonds(n, k)
        end_time = time()
        elapsed = end_time - start_time
        run_times.append(elapsed)

    average_time = sum(run_times) / len(run_times)
    return average_time