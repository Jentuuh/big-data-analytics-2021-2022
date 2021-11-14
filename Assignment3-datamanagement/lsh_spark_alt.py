# This is needed to start a Spark session from the notebook
# You may adjust the memory used by the driver program based on your machine's settings
import os
from collections import defaultdict

import numpy
from pyspark.sql import SparkSession
import numpy as np
import sys
from time import time
from preprocessing import parser

sys.setrecursionlimit(2 ** 27)

os.environ['PYSPARK_SUBMIT_ARGS'] = "--conf spark.driver.memory=8g  pyspark-shell"

INPUT_FILE_PATH = '../data/crypto.txt'
SHINGLE_SIZE = 5
NUM_HASH_FUNCTIONS = 100
BANDS = 16
THRESHOLD = 0.4
AMOUNT_BUCKETS = 100000
RUN_TEST_WITH_JACCARD_SIMILARITY = False
INF = 2 ** 32


def init_spark():
    try:
        spark
        print("Spark application already started. Terminating existing application and starting new one")
        spark.stop()
    except:
        pass

    # Create a new spark session (note, the * indicates to use all available CPU cores)
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .appName("assignment3-datamanagement") \
        .getOrCreate()

    # When dealing with RDDs, we work the sparkContext object.
    # See https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext
    sc = spark.sparkContext

    # We print the sparkcontext. This prints general information about the spark instance we have connected to.
    # In particular, the hyperlink allows us to open the spark UI (useful for seeing what is going on)
    print(sc._conf.getAll())
    return sc


def shingle_generator(size: int, f: [str]):
    for i in range(0, len(f) - size + 1):
        yield tuple(f[i:i + size])


def jaccard_similarity(list1: [int], list2: [int]):
    s1, s2 = set(list1), set(list2)
    print(s1)
    print(s2)
    return len(s1 & s2) / len(s1 | s2)


def generate_hashed_shingles_with_id(element: tuple):
    words = element[0].split(' ')
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    hashed_shingles = set()

    for shingle in shingles:
        # Hash to 32-bit integer
        hashed_shingles.add(hash(shingle) & 0xffffffff)

    # Hashed Shingles, PostID
    return hashed_shingles, element[1]


def generate_hashed_shingles(element: tuple):
    words = element[0].split(' ')
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    hashed_shingles = set()

    for shingle in shingles:
        # Hash to 32-bit integer
        hashed_shingles.add(hash(shingle) & 0xffffffff)

    # Hashed Shingles, PostID
    return hashed_shingles


def doc_to_body(line: str):
    line = line.rstrip('§§§\n')
    line = line.split('#')

    # Tokens, PostID
    return line[2], line[0]


def pair_to_jacc_sim(pair: set):
    pair_list = list(pair)
    similarity = jaccard_similarity(pair_list[0], pair_list[1])
    return similarity


def hash_generator(amount: int, modulo: int):
    h1 = lambda x: (3491 * x + 5003) % modulo
    h2 = lambda x: (1999 * x + 1409) % modulo

    for i in range(amount):
        yield lambda value: (h1(value) + i * h2(value)) % modulo


def find_min_hash(list_to_hash: tuple, num_rows: int):
    signature = []
    hash_funcs = hash_generator(NUM_HASH_FUNCTIONS, num_rows)
    for h in hash_funcs:
        min_val = INF
        for e in list_to_hash[0]:
            hash_val = h(e)
            if hash_val < min_val:
                min_val = hash_val

        signature.append(min_val)

    # Document signature, PostID
    return signature, list_to_hash[1]


def lsh_hash(band_col: tuple):
    hashed_values = []
    for i in band_col[0]:
        entry_i = {"bucket": hash(tuple(i.tolist())) % AMOUNT_BUCKETS, "postId": band_col[1]}
        hashed_values.append(entry_i)

    # Hashed bucket index, PostID
    return hashed_values


def reduce_to_sim_dict(a: [defaultdict], b: [defaultdict]):
    result = [defaultdict(list) for _ in range(BANDS)]

    # for i in range(BANDS):
    #     for key in a[i].keys():
    #         if key not in result[i].keys():
    #             result[i][key] = [a[i][key]]
    #         else:
    #             result[i][key].append(a[i][key])
    #
    #     for key in b[i].keys():
    #         if key not in result[i].keys():
    #             result[i][key] = [b[i][key]]
    #         else:
    #             result[i][key].append(b[i][key])
    for i in range(BANDS):
        for d in (a[i], b[i]):
            for key, value in d.items():
                for e in value:
                    if e not in result[i][key]:
                        result[i][key].append(e)
    return result


def fold_to_one_dict(list_of_dicts: np.array(tuple)):
    result_dict = defaultdict(list)

    for entry in list_of_dicts:
        result_dict[entry['bucket']].append(entry['postId'])

    return result_dict


def filter_out_singles(sim_dict: defaultdict):
    reduced_d = {key: val for key, val in sim_dict.items() if len(val) > 1}
    return reduced_d


def test_with_jaccard_similarity(shingles_per_id: [tuple], similar_candidates: [frozenset]):
    the_ultimate_test = []
    for candidate in similar_candidates[1:]:

        # Retrieve hashed shingles per post ID
        post_hashed_shingles = defaultdict(list)
        for entry in shingles_per_id:
            for candidate_entry in list(candidate):
                if candidate_entry == entry[1]:
                    post_hashed_shingles[candidate_entry] = list(entry[0])

        # Calculate jaccard similarity between candidate posts
        for i, post1 in enumerate(list(candidate)):
            post1_hashed_shingles = list(post_hashed_shingles[post1])

            for post2 in list(candidate)[:i]:
                post2_hashed_shingles = list(post_hashed_shingles[post2])
                jacc_sim = jaccard_similarity(post1_hashed_shingles, post2_hashed_shingles)

                if jacc_sim >= THRESHOLD:
                    the_ultimate_test.append((jacc_sim, post1, post2))

        print(the_ultimate_test)


def spark_test(sc: SparkSession.sparkContext):
    docRDD = sc.textFile(INPUT_FILE_PATH)

    # ---------------- SHINGLING -----------------
    bodyRDD = docRDD.sample(False, 1).map(doc_to_body)
    print("Amount documents: ", bodyRDD.count())
    shingleRDD = bodyRDD.map(generate_hashed_shingles_with_id)

    # Convert each document body into shingles
    flatUniqueShingleRDD = bodyRDD.flatMap(generate_hashed_shingles).distinct()

    # Calculate amount of unique shingles and documents
    amount_unique_shingles = flatUniqueShingleRDD.count()

    # ---------------- MIN HASHING -----------------
    signatureRDD = shingleRDD.map(lambda e: find_min_hash(e, amount_unique_shingles))

    # -------------------- LSH ---------------------
    bandsRDD = signatureRDD.map(lambda x: (np.array_split(x[0], BANDS), x[1]))
    bandsRDD = bandsRDD.map(lsh_hash)

    # Transpose the array of bucket entries per band (so we can use a more efficient map function instead of a reduce)
    bands_array = np.array(bandsRDD.collect())
    transposed_bands_array = bands_array.T
    transposedBandsRDD = sc.parallelize(transposed_bands_array)

    # Fold array of postId, bucketNumber pairs to one dictionary
    transposedBandsRDD = transposedBandsRDD.map(fold_to_one_dict)
    # Filter out the buckets with only one entry
    filteredTransposedBandsRDD = transposedBandsRDD.map(filter_out_singles)

    sim_dict_per_band = filteredTransposedBandsRDD.collect()

    print("Creating count dict...")

    count_sim_dict = defaultdict(int)
    for sim_dict in sim_dict_per_band:
        for entry in sim_dict.values():
            count_sim_dict[frozenset(entry)] += 1

    print("Creating God dict...")
    the_ultimate_god_dict = defaultdict(int)
    similar_candidates = []

    for key, value in count_sim_dict.items():
        the_ultimate_god_dict[value] += 1
        if value / BANDS >= THRESHOLD:
            similar_candidates.append(key)
    print(the_ultimate_god_dict)
    print("Similar candidates: ", similar_candidates)

    for i, candidate in enumerate(similar_candidates):
        parser.find_posts_by_ids(list(candidate), '../data/Posts.xml', '../data/output/result_data_management' + str(i) + '.txt')

    if RUN_TEST_WITH_JACCARD_SIMILARITY:
        shingles_per_id = shingleRDD.collect()
        test_with_jaccard_similarity(shingles_per_id, similar_candidates)
    return


def main():
    start_time = time()
    sc = init_spark()
    spark_test(sc)

    sc.stop()
    total_time = time() - start_time
    print("Total runtime: ", total_time, "s")


main()
