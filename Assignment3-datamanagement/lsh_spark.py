# This is needed to start a Spark session from the notebook
# You may adjust the memory used by the driver program based on your machine's settings
import os
from collections import defaultdict

os.environ['PYSPARK_SUBMIT_ARGS'] = "--conf spark.driver.memory=8g  pyspark-shell"
from pyspark.sql import SparkSession
import random

INPUT_FILE_PATH = '../data/crypto.txt'
SHINGLE_SIZE = 5


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


def make_shingle_index_map(unique_shingles: [int]):
    shingle_index_map = defaultdict()
    for i, shingle in enumerate(unique_shingles):
        shingle_index_map[shingle] = i
    return shingle_index_map


def count_unique_shingles():
    pass


def build_sparse_matrix(docs: [set], amount_unique: int, amount_docs: int, shingle_index_map: defaultdict):
    sparse_matrix = [0] * (amount_unique * amount_docs)

    for i, doc in enumerate(docs):
        for j, shingle in enumerate(doc):
            shingle_index = shingle_index_map[shingle]
            sparse_matrix[shingle_index * amount_docs + i] = 1

    return sparse_matrix


def min_hash():
    # 1 . Build sparse matrix (with 0's and 1's)

    # 2 . For each row r in this matrix, compute h_i(r), for each hash function h_i

    # ----------------------------------------------------------------------------

    # 3 . For each column c

    # 4 . If c has a 1 in row r, loop through the hash functions calculated above and assign the minimum value to
    #     M(i,c)

    return


def spark_test(sc: SparkSession.sparkContext):
    fileName = '../data/crypto.txt'
    docRDD = sc.textFile(fileName)
    bodyRDD = docRDD.sample(False, 0.05).map(doc_to_body)
    shingleRDD = bodyRDD.map(generate_hashed_shingles)

    # Convert each document body into shingles
    flatUniqueShingleRDD = bodyRDD.flatMap(generate_hashed_shingles).distinct()

    # Calculate amount of unique shingles and documents
    amount_docs = bodyRDD.count()
    amount_unique_shingles = flatUniqueShingleRDD.count()

    # ------- MIN HASHING --------
    print("Building shingle index map...")
    shingle_idx_map = make_shingle_index_map(flatUniqueShingleRDD.collect())
    print("Building sparse matrix...")
    sparse_matrix = build_sparse_matrix(shingleRDD.collect(), amount_unique_shingles, amount_docs, shingle_idx_map)
    
    # Find better way to store the sparse matrix
    print(sparse_matrix)


def main():
    sc = init_spark()
    spark_test(sc)

    sc.stop()


main()
