# This is needed to start a Spark session from the notebook
# You may adjust the memory used by the driver program based on your machine's settings
import os

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


def generate_hashed_shingles(words: [str]):
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


def brute_force_spark(sc: SparkSession.sparkContext):
    fileName = '../data/crypto.txt'
    docRDD = sc.textFile(fileName)
    bodyRDD = docRDD.sample(False, 0.1).map(doc_to_body)

    # Convert each document body into shingles
    shingleRDD = bodyRDD.map(generate_shingles)
    # Generate pairs of shingle sets (to later calculate jaccard similarity in next pass)
    pairRDD = shingleRDD.cartesian(shingleRDD).filter(lambda x: x[1] != x[0])
    # Calculate Jaccard similarities
    similarityRDD = pairRDD.map(pair_to_jacc_sim).filter(lambda x: x != 0.0)

    similarityRDD.foreach(lambda x: print(x))


def main():
    sc = init_spark()
    brute_force_spark(sc)

    sc.stop()


main()
