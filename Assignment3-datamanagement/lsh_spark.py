# This is needed to start a Spark session from the notebook
# You may adjust the memory used by the driver program based on your machine's settings
import os
from collections import defaultdict
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import SparseMatrix
import numpy as np

os.environ['PYSPARK_SUBMIT_ARGS'] = "--conf spark.driver.memory=8g  pyspark-shell"

INPUT_FILE_PATH = '../data/crypto.txt'
SHINGLE_SIZE = 5
NUM_HASH_FUNCTIONS = 100
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


def make_shingle_index_map(unique_shingles: [int]):
    shingle_index_map = defaultdict()
    for i, shingle in enumerate(unique_shingles):
        shingle_index_map[shingle] = i
    return shingle_index_map


# 1 . Build sparse matrix (with 0's and 1's)
def build_sparse_matrix(docs: [set], amount_unique: int, amount_docs: int, shingle_index_map: defaultdict):
    col_ptrs = [0]
    row_ptrs = []
    values = []

    current_col_ptr = 0
    for i, doc in enumerate(docs):
        for j, shingle in enumerate(doc):
            shingle_index = shingle_index_map[shingle]
            row_ptrs.append(shingle_index)
            values.append(1)
            current_col_ptr += 1
        col_ptrs.append(current_col_ptr)

    sparse_matrix = SparseMatrix(amount_unique, amount_docs, col_ptrs, row_ptrs, values)

    return sparse_matrix


def generate_hash_results(value: int, num_rows: int):
    h1 = lambda x: (3491 * x + 5003) % num_rows
    h2 = lambda x: (1999 * x + 1409) % num_rows

    results = []

    for i in range(NUM_HASH_FUNCTIONS):
        results.append((h1(value) + i * h2(value)) % num_rows)

    return results


def min_hash(num_rows: int, num_cols: int, sparse_matrix: SparseMatrix):
    signature_matrix = [[INF] * num_cols for i in range(NUM_HASH_FUNCTIONS)]

    # 2 . For each row r in this matrix, compute h_i(r), for each hash function h_i
    for r in range(num_rows):
        hash_results = generate_hash_results(r + 1, num_rows)

        # 3 . For each column c
        # 4 . If c has a 1 in row r, loop through the hash functions calculated above and assign the minimum value to
        #     M(i,c)
        for c in range(num_cols):
            if sparse_matrix[(r, c)] == 1:
                min_index = np.argmin(hash_results)
                signature_matrix[min_index][c] = hash_results[min_index]

    return signature_matrix


def map_to_signature_value(hash_min: int, min_index: int, row_nr: int, num_cols: int, sparse_matrix: SparseMatrix):

    row_indices = sparse_matrix.rowIndices
    results = []

    if row_nr in row_indices:
        for i in range(num_cols):
            if sparse_matrix[(row_nr, i)] == 1:
                results.append([hash_min, (min_index, i)])

    return results


def spark_test(sc: SparkSession.sparkContext):
    docRDD = sc.textFile(INPUT_FILE_PATH)
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

    print("Creating pairRDD...")
    rowsRDD = sc.parallelize(range(1, amount_unique_shingles + 1))
    rowsRDD = rowsRDD.map(lambda value: (value, generate_hash_results(value, amount_unique_shingles)))
    minHashRDD = rowsRDD.map(lambda x: (x[0], min(x[1]), np.argmin(x[1])))

    print(minHashRDD.take(3))

    signatureRDD = minHashRDD.map(lambda x: map_to_signature_value(x[1], x[2], x[0], amount_docs, sparse_matrix))
    print(signatureRDD.collect())

    print("Building signature matrix...")

    # print(min_hash(amount_unique_shingles, amount_docs, sparse_matrix))


def main():
    sc = init_spark()
    spark_test(sc)

    sc.stop()


main()
