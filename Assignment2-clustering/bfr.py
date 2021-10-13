import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation
import matplotlib.pyplot as plt


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
    #print("Longest common subsequence: " + ''.join(longest_common_subseq))
    return len(s1) + len(s2) - 2 * len_lcs


def create_discard_set(points):
    # TODO: Summarize discard set by obtaining the number of points, the SUM and SUMSQ vectors
    return


def bfr(dataset):

    ds = []
    cs = []
    rs = []

    # TODO: INITIALIZE K CENTROIDS : take k random points (entries) from our initial load /
    #                                take a small random sample and cluster optimally /
    #                                take a sample, pick random point, then k-1 more points, each as far as possible

    return


# k-means experimenting

X = np.array([[1, 2], [1, 4], [1, 0],
               [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print(edit_distance("The man sits in the park", "The woman sits in the park"))
