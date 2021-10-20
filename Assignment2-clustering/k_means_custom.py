import os

import numpy as np
import random
from Levenshtein import distance
from time import process_time
from datetime import datetime
from os import mkdir

INPUT_FILE_PATH = '../data/dblp_clustering.txt'
INFINITY = 2**31
NUM_CLUSTERS = 100
REPETITIONS = 1
SEED = 459392
FOLDER_NAME = str(datetime.now())


def parseDatasetFromFile(file_name: str, periodStart: int, periodEnd: int):
    titles = []
    min_year = 9999
    max_year = 0

    f = open(file_name, "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("\n", "")
        line_data = line.rsplit('#', 1)


        # Only parse the data from the assigned period
        if periodStart <= int(line_data[1]) <= periodEnd:
            titles.append(line_data[0])

            if int(line_data[1]) < int(min_year):
                min_year = line_data[1]
            if int(line_data[1]) > int(max_year):
                max_year = line_data[1]
    f.close()

    return titles, min_year, max_year


def calculate_closest_distance(title: str, clustroids: list[str]):
    # Calculate the levenshtein distance between each 'point' (title) and the clustroids
    lev_distance = np.array([distance(title, t2)for t2 in clustroids])

    # Index of the minimum distance
    min_index = np.argmin(lev_distance)
    closest_dist = lev_distance[min_index]

    min_distance_pair = {"distance": closest_dist, "index": min_index }
    return min_distance_pair


def update_clustroids(points: list[str], clusters: list[int]):

    new_clustroids = []
    cluster_points_list = []

    # Make lists of the points within each cluster
    for i in range(NUM_CLUSTERS):
        clustroid_points = []
        for j in range(len(clusters)):
            if clusters[j] == i:
                clustroid_points.append(points[j])
        cluster_points_list.append(clustroid_points)

    # For each cluster, we try to find the point with the minimal maximum distance to the other points in that cluster,
    # this will be the new clustroid.
    for c_points in cluster_points_list:
        lev_distance = np.array([[distance(t1, t2) for t2 in c_points] for t1 in c_points])
        min_max_distance = INFINITY
        new_clustroid_index = -1

        if len(lev_distance) == 0:
            continue

        for index, row in enumerate(lev_distance):
            # Filter 0's (we don't care about the distance to a point itself)
            # masked_row = np.ma.masked_array(row, mask=row == 0)
            max_index = np.argmax(row)
            if row[max_index] < min_max_distance:
                new_clustroid_index = index
                min_max_distance = row[max_index]
        new_clustroids.append(c_points[new_clustroid_index])

    return new_clustroids


def print_kmeans_results(num_clusters: int, clusters: list[int], points: list[str], time_per_repetition: list[float],
                         steps_per_repetition: list[int], start_yr: int, end_yr: int):
    cluster_dict = {}
    for index, c in enumerate(clusters):
        if c not in cluster_dict.keys():
            cluster_dict[c] = [points[index]]
        else:
            cluster_dict[c].append(points[index])

    print("============================================\n"
          "Finished K-means clustering for " + str(num_clusters) + " clusters.\n"
          "============ PERIOD: " + str(start_yr) + "-" + str(end_yr) + " =============\n"
          "============================================\n")
    for i, t in enumerate(time_per_repetition):
        print("============== Repetition " + str(i) + " ================")
        print("Steps: " + str(steps_per_repetition[i]))
        print("Total time: " + str(t) + "s")
        print()
        for key, value in sorted(cluster_dict.items()):
            print("Cluster " + str(key) + ": " + str(value))
    print()
    return


def write_kmeans_results_to_file(file_name: str, num_clusters: int, clusters: list[int], points: list[str], time_per_repetition: list[float],
                                 steps_per_repetition: list[int], start_yr: int, end_yr: int):

    f = open(file_name, "w")

    cluster_dict = {}
    for index, c in enumerate(clusters):
        if c not in cluster_dict.keys():
            cluster_dict[c] = [points[index]]
        else:
            cluster_dict[c].append(points[index])

    f.write("============================================\n"
          "Finished K-means clustering for " + str(num_clusters) + " clusters.\n"
          "============ PERIOD: " + str(start_yr) + "-" + str(end_yr) + " =============\n"
          "============================================\n")
    for i, t in enumerate(time_per_repetition):
        f.write("============== Repetition " + str(i) + " ================\n")
        f.write("Steps: " + str(steps_per_repetition[i]) + "\n")
        f.write("Total time: " + str(t) + "s\n")
        f.write("\n")
        for key, value in sorted(cluster_dict.items()):
            f.write("Cluster " + str(key) + ": " + str(value) + "\n")
    f.write("\n")
    return


def kmeans(num_clusters: int, repetitions: int, points: list[str], start_year: int):

    # Pseudo-random
    random.seed(SEED)
    # Store the best results (in case we do multiple repetitions
    best_dist_sum = INFINITY
    best_clusters = []
    steps_per_repetition = []
    time_per_repetition = []

    for i in range(repetitions):
        starttime = process_time()
        num_steps = 0

        # Initialize clustroids
        initial_clustroid_indices = random.sample(range(1, len(points)), num_clusters)
        clustroids = []
        for clustroid_i in initial_clustroid_indices:
            clustroids.append(points[clustroid_i])

        # Initialize the clusters that each point belongs to
        clusters = [-1] * len(points)

        changed = True
        while changed:
            num_steps += 1
            changed = False
            distance_sum = 0

            for p_index, t in enumerate(points):
                # Calculate which clustroid is closest
                closest_distance_and_clustroid_index = calculate_closest_distance(t, clustroids)
                # Keep track of the total sum of distances (the one with the smallest sum will be the best result)
                distance_sum += closest_distance_and_clustroid_index["distance"]

                # Check if the point has found a closer clustroid (and thus needs to be moved to another cluster)
                if closest_distance_and_clustroid_index["index"] != clusters[p_index]:
                    clusters[p_index] = closest_distance_and_clustroid_index["index"]
                    changed = True

            if changed:
                clustroids = update_clustroids(points, clusters)
            # Keep track of best clustering
            if distance_sum < best_dist_sum:
                best_clusters = clusters
                best_dist_sum = distance_sum

        time_per_repetition.append(process_time() - starttime)
        steps_per_repetition.append(num_steps)

    # Output to CLI and file
    print_kmeans_results(num_clusters, clusters, points, time_per_repetition, steps_per_repetition, start_year - 1,
                         start_year + 11)
    write_kmeans_results_to_file("../data/output/" + FOLDER_NAME + "/" + str(start_year - 1) + '-' +
                                 str(start_year + 11) + ".txt", num_clusters, clusters, points, time_per_repetition,
                                 steps_per_repetition, start_year - 1, start_year + 11)
    return


def main():
    mkdir("../data/output/" + FOLDER_NAME + "/")
    print("Parsing...")
    titles, min_yr, max_yr = parseDatasetFromFile(INPUT_FILE_PATH, 0, 9999)
    print("Minimal year:" + min_yr)
    print("Maximal year:" + max_yr)
    print("Parsing done.")

    start_year = int(min_yr)
    while start_year <= int(max_yr):
        titles, min_yr, max_yr = parseDatasetFromFile(INPUT_FILE_PATH, start_year - 1, start_year + 11)
        print("Performing K-Means using Levenshtein distance...")
        print("Entries: " + str(len(titles)))
        kmeans(NUM_CLUSTERS, REPETITIONS, titles, start_year)
        start_year += 10


main()

