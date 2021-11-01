import numpy as np
from Levenshtein import distance
from sklearn.cluster import AffinityPropagation


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

def find_similar_topics(topic_list):
    topics = np.asarray(topic_list) # So that indexing with a list will work
    lev_similarity = -1*np.array([[distance(w1, w2) for w1 in topics] for w2 in topics])

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        cluster_topic = topics[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(topics[np.nonzero(affprop.labels_ == cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - Topic %s:   * %s" % (cluster_topic, cluster_str))


print("Parsing...")
titles, years = parseDatasetFromFile(INPUT_FILE_PATH)

names = ["Chris", "Kristof", "Bart", "Bas", "Brecht", "Bram", "Brent", "Inguru", "Kristel", "Kris"]

print("Parsing done.")
find_similar_topics(names)
