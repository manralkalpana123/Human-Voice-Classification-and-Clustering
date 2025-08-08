import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json


def best_kmeans(X, k_range=range(2,8)):
    best = None
    best_score = -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        print('k=', k, 'silhouette=', score)
        if score > best_score:
            best_score = score
            best = (k, km, score)
    return best


if __name__ == '__main__':
    import numpy as np
    import os
    os.makedirs('models', exist_ok=True)

    data = np.load('data/processed/train_test.npz')
    X_train = data['X_train']

    k, km_model, score = best_kmeans(X_train, range(2,10))
    print('Best k:', k, 'score:', score)
    joblib.dump(km_model, 'models/kmeans.joblib')
