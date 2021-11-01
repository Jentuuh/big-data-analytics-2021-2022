import matplotlib.pyplot
import numpy as np
import random
from Levenshtein import distance
from time import process_time
from datetime import datetime
from os import mkdir, path
from matplotlib import pyplot

# Our implementation of K-means with clustroids.
# Before running this file, you should preprocess the dataset by running the 'parser.py' script. This will transform
# the XML data into our own custom format to speed up the parsing process. For us, the input file for this script is
# located at '../data/dblp_clustering.txt', but you can change this to your own preference right below this comment.

# The outputs of the clustering for each batch will be written to the CLI as well as a file located in the
# '../data/output/<timestamp>/<time_period>.txt'

# We used Python version 3.9.7 to make this assignment. Note that we had problems installing the Levenshtein package
# on version 3.10.0 .

# Jente Vandersanden and Ingo Andelhofs, Big Data Analytics 2021 - 2022, Hasselt University.

INPUT_FILE_PATH = '../data/dblp_clustering.txt'
OUTPUT_FOLDER_PATH = '../data/output/'
FOLDER_NAME = str(datetime.now())
FOLDER_PATH = OUTPUT_FOLDER_PATH + FOLDER_NAME + "/"
PLOTS_FOLDER_PATH = OUTPUT_FOLDER_PATH + 'analysis_plots/'

INFINITY = 2**31
REPETITIONS = 1
SEED = 459392


def parseDatasetFromFile(file_name: str, periodStart: int, periodEnd: int):
    titles = []
    min_year = INFINITY
    max_year = 0

    f = open(file_name, "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("\n", "")
        line_data = line.rsplit('#', 1)

        title = line_data[0]
        period = int(line_data[1])

        # Only parse the data from the assigned period
        if periodStart <= period <= periodEnd:
            titles.append(title)

            if period < min_year:
                min_year = period
            if period > max_year:
                max_year = period
    f.close()

    return titles, min_year, max_year


def calculate_closest_distance(title: str, clustroids: 'list[str]'):
    # Calculate the levenshtein distance between each 'point' (title) and the clustroids
    lev_distance = np.array([distance(title, t2)for t2 in clustroids])

    # Index of the minimum distance
    min_index = np.argmin(lev_distance)
    closest_dist = lev_distance[min_index]

    min_distance_pair = {"distance": closest_dist, "index": min_index }
    return min_distance_pair


def update_clustroids(points: 'list[str]', clusters: 'list[int]', num_clusters: int):

    new_clustroids = []
    cluster_points_list = []

    # Make lists of the points within each cluster
    for i in range(num_clusters):
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
            max_index = np.argmax(row)
            if row[max_index] < min_max_distance:
                new_clustroid_index = index
                min_max_distance = row[max_index]
        new_clustroids.append(c_points[new_clustroid_index])

    return new_clustroids


def print_kmeans_results(
    num_clusters: int, 
    clusters: 'list[int]', 
    clustroids: 'list[str]',
    points: 'list[str]', 
    time_per_repetition: 'list[float]',
    steps_per_repetition: 'list[int]', 
    start_yr: int, 
    end_yr: int):

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
    cluster_i = 0
    for i, t in enumerate(time_per_repetition):
        print("============== Repetition " + str(i) + " ================")
        print("Steps: " + str(steps_per_repetition[i]))
        print("Total time: " + str(t) + "s")
        print()
        for key, value in sorted(cluster_dict.items()):
            print("Cluster " + str(key) + ": Clustroid: " + str(clustroids[cluster_i]) + str(value) + "\n")
            cluster_i += 1
    print()
    return


def write_kmeans_results_to_file(
    file_name: str, 
    num_clusters: int, 
    clusters: 'list[int]', 
    clustroids: 'list[str]',
    points: 'list[str]', 
    time_per_repetition: 'list[float]',
    steps_per_repetition: 'list[int]', 
    start_yr: int, 
    end_yr: int):

    f = open(file_name, "w")

    cluster_dict = {}
    for index, c in enumerate(clusters):
        if c not in cluster_dict.keys():
            cluster_dict[c] = [points[index]]
        else:
            cluster_dict[c].append(points[index])

    f.write("#============================================\n"
          "#Finished K-means clustering for " + str(num_clusters) + " clusters.\n"
          "#============ PERIOD: " + str(start_yr) + "-" + str(end_yr) + " =============\n"
          "#============================================\n")
    cluster_i = 0
    for i, t in enumerate(time_per_repetition):
        f.write("#============== Repetition " + str(i) + " ================\n")
        f.write("#Steps: " + str(steps_per_repetition[i]) + "\n")
        f.write("#Total time: " + str(t) + "s\n")
        f.write("\n")
        for key, value in sorted(cluster_dict.items()):
            f.write(str(key) + "§" + str(clustroids[cluster_i]) + "§" + "§".join(value) + "\n")
            # f.write("Cluster " + str(key) + ": Clustroid: " + str(clustroids[cluster_i]) + str(value) + "\n")
            cluster_i += 1
    f.write("\n")
    return


def kmeans(num_clusters: int, repetitions: int, points: 'list[str]', start_year: int):

    # Pseudo-random
    random.seed(SEED)
    # Store the best results (in case we do multiple repetitions
    steps_per_repetition = []
    time_per_repetition = []

    for i in range(repetitions):
        starttime = process_time()
        num_steps = 0

        # Initialize clustroids
        initial_clustroid_indices = random.sample(range(0, len(points)), num_clusters)
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
                clustroids = update_clustroids(points, clusters, num_clusters)

        time_per_repetition.append(process_time() - starttime)
        steps_per_repetition.append(num_steps)

    # Output to CLI and file

    print_kmeans_results(
        num_clusters,
        clusters,
        clustroids,
        points,
        time_per_repetition,
        steps_per_repetition,
        start_year,
        start_year + 10)

    write_kmeans_results_to_file(
        FOLDER_PATH + str(start_year) + '-' + str(start_year + 10) + ".txt",
        num_clusters,
        clusters,
        clustroids,
        points,
        time_per_repetition,
        steps_per_repetition,
        start_year,
        start_year + 10)
    return clusters, clustroids


def find_optimal_k(min_yr: int, max_yr: int):
    # Create a output directory for our data
    output_folder_exists = path.exists(PLOTS_FOLDER_PATH)

    if not output_folder_exists:
        mkdir(PLOTS_FOLDER_PATH)

    # Parse the dataset
    print("Parsing...")
    titles, min_yr, max_yr = parseDatasetFromFile(INPUT_FILE_PATH, min_yr, max_yr)
    print("Minimal year: {0}".format(min_yr))
    print("Maximal year: {0}".format(max_yr))
    print("Parsing done.")

    k_values = []
    avgs = []

    current_k = 10
    while current_k < 200:
        k_values.append(current_k)
        # Perform k-means
        print("K : " + str(current_k))
        clusters, clustroids = kmeans(current_k, REPETITIONS, titles, min_yr)
        total_avg = calculate_wcss(clusters, clustroids, titles)
        avgs.append(total_avg)
        current_k += 10
    pyplot.plot(k_values, avgs)
    pyplot.title("WCSS per K for clustering of period " + str(min_yr) + "-" + str(max_yr))
    pyplot.xlabel("Value of K")
    pyplot.ylabel("WCSS")
    pyplot.savefig(PLOTS_FOLDER_PATH + 'WCSS_K' + str(min_yr) + '-' + str(max_yr) + '.png')


def calculate_wcss(clusters: list[int], clustroids: list[str], points: list[str]):

    distances_to_clustroid = [0] * len(clustroids)
    num_points_in_cluster = [0] * len(clustroids)

    # Sum all the distances to the centroid for each
    for index, cluster_index in enumerate(clusters):
        dist_to_clustroid = distance(clustroids[cluster_index], points[index])
        distances_to_clustroid[cluster_index] += dist_to_clustroid
        num_points_in_cluster[cluster_index] += 1

    # Divide all intra-cluster distances by its number of points to get an average
    for i in range(0, len(distances_to_clustroid)):
        distances_to_clustroid[i] /= num_points_in_cluster[i]

    total_sum = 0
    for avg_dist in distances_to_clustroid:
        total_sum += avg_dist
    total_avg = total_sum / len(distances_to_clustroid)
    return total_avg


def main():
    k = [50, 75, 50, 60, 30, 65, 50, 55, 65, 80, 80]
    # Create a output directory for our data
    output_folder_exists = path.exists(OUTPUT_FOLDER_PATH)

    if not output_folder_exists:
        mkdir(OUTPUT_FOLDER_PATH)

    mkdir(FOLDER_PATH)

    # Parse the dataset
    print("Parsing...")
    titles, min_yr, max_yr = parseDatasetFromFile(INPUT_FILE_PATH, 0, INFINITY)
    print("Minimal year: {0}".format(min_yr))
    print("Maximal year: {0}".format(max_yr))
    print("Parsing done.")

    # Perform k-means
    start_year = min_yr
    i = 0
    while start_year <= max_yr:
        start = start_year
        end = start_year + 10

        titles, min_yr, max_yr = parseDatasetFromFile(INPUT_FILE_PATH, start, end)
        print("Minimal year: {0}".format(min_yr))
        print("Maximal year: {0}".format(max_yr))
        print("Performing K-Means using Levenshtein distance...")
        print("Entries: {0}".format(len(titles)))
        kmeans(k[i], REPETITIONS, titles, start_year)
        start_year += 5
        i += 1


# find_optimal_k(1994, 2004)
main()
