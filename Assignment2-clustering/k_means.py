
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time

import numpy as np

INPUT_FILE_PATH = '../data/dblp50000_clustering.txt'


def parseDatasetFromFile(file_name):
    titles = []
    years = []

    f = open(file_name, "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("\n", "")
        line_data = line.split('#')
        titles.append(line_data[0])
        years.append(line_data[1])
    f.close()

    return titles, years


# #############################################################################
print("Parsing...")
titles, years = parseDatasetFromFile(INPUT_FILE_PATH)
print("Parsing done.")

# names = ["Chris", "Kristof", "Bart", "Bas", "Brecht", "Bram", "Brent", "Inguru", "Kristel", "Kris"]
#
# titles = names
print("%d documents" % len(titles))
print()

num_clusters = 20

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()

vectorizer = TfidfVectorizer(max_df=15, max_features=10,
                             min_df=1, stop_words='english',
                             use_idf=True)

X = vectorizer.fit_transform(titles)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()


# #############################################################################
# Do the actual clustering

# if opts.minibatch:
#     km = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
#                          init_size=1000, batch_size=1000)
km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
# print()
#
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()

print("Top terms per cluster:")

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
features = km.labels_


terms = vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()