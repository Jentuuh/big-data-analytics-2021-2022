# This is needed to start a Spark session from the notebook
# You may adjust the memory used by the driver program based on your machine's settings
import os
from collections import defaultdict
from pyspark.sql import SparkSession
import numpy as np
import sys
sys.setrecursionlimit(2**27)


os.environ['PYSPARK_SUBMIT_ARGS'] = "--conf spark.driver.memory=8g  pyspark-shell"

INPUT_FILE_PATH = '../data/crypto.txt'
SHINGLE_SIZE = 5
NUM_HASH_FUNCTIONS = 100
BANDS = 20
THRESHOLD = 0.4
AMOUNT_BUCKETS = 100000
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


def generate_shingles(body: str):
    words = body.split(' ')
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    return shingles


def shingle_generator(size: int, f: [str]):
    for i in range(0, len(f) - size + 1):
        yield tuple(f[i:i + size])


def jaccard_similarity(list1: [int], list2: [int]):
    s1, s2 = set(list1), set(list2)
    return len(s1 & s2) / len(s1 | s2)


def generate_hashed_shingles(body: str):
    words = body.split(' ')
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    hashed_shingles = set()

    for shingle in shingles:
        # Hash to 32-bit integer
        hashed_shingles.add(hash(shingle) & 0xffffffff)

    return hashed_shingles


def doc_to_body(line: str):
    line = line.rstrip('§§§\n')
    line = line.split('#')
    return line[2]


def pair_to_jacc_sim(pair: set):
    pair_list = list(pair)
    similarity = jaccard_similarity(pair_list[0], pair_list[1])
    return similarity


def hash_generator(amount: int, modulo: int):
    h1 = lambda x: (3491 * x + 5003) % modulo
    h2 = lambda x: (1999 * x + 1409) % modulo

    for i in range(amount):
        yield lambda value: (h1(value) + i * h2(value)) % modulo


def find_min_hash(list_to_hash: [int], num_rows: int):
    signature = []
    hash_funcs = hash_generator(NUM_HASH_FUNCTIONS, num_rows)
    for h in hash_funcs:
        min_val = INF
        for e in list_to_hash:
            hash_val = h(e)
            if hash_val < min_val:
                min_val = hash_val

        signature.append(min_val)
    return signature


def lsh_hash(band_col: np.array):
    hashed_values = []
    for i in band_col:
        dict_i = defaultdict(list)
        dict_i[hash(tuple(i.tolist())) % AMOUNT_BUCKETS] = i.tolist()
        hashed_values.append(dict_i)
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
                    result[i][key].append(e)
    return result


def spark_test(sc: SparkSession.sparkContext):
    docRDD = sc.textFile(INPUT_FILE_PATH)

    # ---------------- SHINGLING -----------------
    bodyRDD = docRDD.sample(False, 0.05).map(doc_to_body)
    shingleRDD = bodyRDD.map(generate_hashed_shingles)

    # Convert each document body into shingles
    flatUniqueShingleRDD = bodyRDD.flatMap(generate_hashed_shingles).distinct()

    # Calculate amount of unique shingles and documents
    amount_docs = bodyRDD.count()
    amount_unique_shingles = flatUniqueShingleRDD.count()

    # ---------------- MIN HASHING -----------------
    signatureRDD = shingleRDD.map(lambda e: find_min_hash(e, amount_unique_shingles))

    # -------------------- LSH ---------------------
    bandsRDD = signatureRDD.map(lambda x: np.array_split(x, BANDS))
    bandsRDD = bandsRDD.map(lsh_hash)

    print("Reducing...")
    sim_dictRDD = bandsRDD.reduce(lambda a, b: reduce_to_sim_dict(a, b))
    print(sim_dictRDD)
    return


def main():
    sc = init_spark()
    spark_test(sc)

    sc.stop()


main()
