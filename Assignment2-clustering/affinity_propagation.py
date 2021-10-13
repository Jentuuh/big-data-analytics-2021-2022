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


# Levenshtein distance (algorithm from https://www.geeksforgeeks.org/printing-longest-common-subsequence)
def lcs(s1, s2, m, n):
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif s1[i - 1] == s2[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Following code is used to print LCS
    index = L[m][n]
    length_of_lcs = index

    # Create a character array to store the lcs string
    lcs = [""] * (index + 1)
    lcs[index] = ""

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:

        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if s1[i - 1] == s2[j - 1]:
            lcs[index - 1] = s1[i - 1]
            i -= 1
            j -= 1
            index -= 1

        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs, length_of_lcs


def edit_distance(s1, s2):
    longest_common_subseq, len_lcs = lcs(s1, s2, len(s1), len(s2))
    return len(s1) + len(s2) - 2 * len_lcs


def find_similar_topics(topic_list):
    topics = np.asarray(topic_list) # So that indexing with a list will work
    lev_similarity = -1*np.array([[distance(w1, w2) for w1 in topics] for w2 in topics])

    print(lev_similarity)

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        cluster_topic = topics[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(topics[np.nonzero(affprop.labels_ == cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - Topic %s:   * %s" % (cluster_topic, cluster_str))


print("Parsing...")
titles, years = parseDatasetFromFile(INPUT_FILE_PATH)
print("Parsing done.")
find_similar_topics(titles)
