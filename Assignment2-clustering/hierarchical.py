import numpy as np
from Levenshtein import distance


INPUT_FILE_PATH = '../data/dblp50000_clustering.txt'
INFINITY = 2**31


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


def find_similar_topics(topic_list: list[str],  cluster_count: int, clusters: list[int]):

    curr_pass = 1
    while cluster_count > 1:
        # Find minimum intercluster distance
        min_distance_indices = []
        for c1 in range(cluster_count):
            c1_min = INFINITY
            c2_min_index = -1
            for c2 in range(cluster_count):

                min_distance, min_distance_pair = calculate_intercluster_distance(clusters, c1, c2, topic_list)
                if min_distance < c1_min:
                    c2_min_index = min_distance_pair["second"]
                    c1_min = min_distance
            min_distance_indices.append(c2_min_index)
        print(min_distance_indices)

        clusters, cluster_count = merge_clusters(clusters, min_distance_indices, cluster_count)
        print(cluster_count)
        print("Clusters pass " + str(curr_pass) + ": " + str(clusters))
        curr_pass += 1

    # # Make new cluster and combine the closest topics in this cluster
    # cluster_count += 1
    # clusters[index] = cluster_count
    # clusters[min_index] = cluster_count


def find_clustroid():
    # TODO: given a list of titles and for each title the cluster that they belong to, calculate the clustroid of each cluster
    return


def calculate_intercluster_distance(clusters: list[int], cluster1: int, cluster2: int, titles: list[str]):

    cluster1_titles: list[dict] = []
    cluster2_titles: list[dict] = []

    for index, title_cluster in enumerate(clusters):
        if title_cluster == cluster1:
            cluster1_titles.append({"title": titles[index], "index": index})
        elif title_cluster == cluster2:
            cluster2_titles.append({"title": titles[index], "index": index})

    # If one of the cluster numbers was not found, we can stop here already
    if len(cluster1_titles) == 0 or len(cluster2_titles) == 0:
        return INFINITY, {}

    lev_distance = np.array([[distance(t1["title"], t2["title"]) for t1 in cluster1_titles] for t2 in cluster2_titles])

    # Find the minimum of the distances between any two points, one from each cluster
    min_distance = INFINITY
    min_distance_pair = {"first": -1, "second": -1}
    for index, row in enumerate(lev_distance):
        # Filter 0's
        masked_row = np.ma.masked_array(row, mask=row == 0)
        min_index = np.argmin(masked_row)
        if masked_row[min_index] < min_distance:
            min_distance_pair["first"] = cluster1_titles[index]["index"]
            min_distance_pair["second"] = cluster2_titles[min_index]["index"]
            min_distance = masked_row[min_index]

    return min_distance, min_distance_pair


def merge_clusters(clusters: list[int], min_distance_indices: list[int], cluster_count: int):
    modified_clusters = clusters.copy()
    for i, min_distance_index in enumerate(min_distance_indices):
        if min_distance_index > i:
            # Merge the closest clusters into one cluster
            modified_clusters[min_distance_index] = clusters[i]
            cluster_count -= 1
    clusters = modified_clusters
    return clusters, cluster_count


def main():
    print("Parsing...")
    titles, years = parseDatasetFromFile(INPUT_FILE_PATH)
    print("Parsing done.")

    clusters = [x for x in range(len(titles))]
    cluster_count = len(titles)

    find_similar_topics(titles, cluster_count, clusters)

main()