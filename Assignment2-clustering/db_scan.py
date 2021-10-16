from Levenshtein import distance
import numpy as np
from sklearn.cluster import dbscan
data = ["ACCTCCTAGAAG", "ACCTACTAGAAGTT", "GAATATTAGGCCGA"]
def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return distance(data[i], data[j])

X = np.arange(len(data)).reshape(-1, 1)
results = dbscan(X, metric=lev_metric, eps=5, min_samples=3)

print(results)