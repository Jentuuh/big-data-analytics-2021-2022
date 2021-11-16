# This is needed to start a Spark session from the notebook
# You may adjust the memory used by the driver program based on your machine's settings
import os
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from matplotlib import pyplot as plt

os.environ['PYSPARK_SUBMIT_ARGS'] = "--conf spark.driver.memory=8g pyspark-shell"

# Our Brute-Force approach implemented on top of Spark. Before running this script, make sure you parsed the XML using
# the 'preprocessing/parser.py' script.

# Jente Vandersanden and Ingo Andelhofs, Big Data Analytics 2021 - 2022, Hasselt University.

INPUT_FILE_PATH = '../data/crypto.txt'
SHINGLE_SIZE = 5


def init_spark():
    try:
        spark
        print("Spark application already started. Terminating existing application and starting new one")
        spark.stop()
    except:
        pass

    # When dealing with RDDs, we work the sparkContext object.
    # See https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext
    conf = (SparkConf()
            .set("spark.driver.maxResultSize", "2g"))
    sc = SparkContext(conf=conf)

    # We print the sparkcontext. This prints general information about the spark instance we have connected to.
    # In particular, the hyperlink allows us to open the spark UI (useful for seeing what is going on)
    print(sc._conf.getAll())
    return sc


def generate_shingles(body: str):
    """
    Given a string of tokens (body), converts these tokens into a list of shingles.
    :param body: The token string
    :return: A list of shingles
    """
    words = body.split(' ')
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    return shingles


def shingle_generator(size: int, tokens: [str]):
    """
    Generates shingles of a certain size, given a list of tokens.
    :param size: Length of the shingles
    :param tokens: Tokens that will be used to generate the shingles
    :return: Yields all possible shingles of length `size` that can be made out of tokens
    """
    for i in range(0, len(tokens) - size + 1):
        yield tuple(tokens[i:i + size])


def jaccard_similarity(list1: [int], list2: [int]):
    """
    Calculates the Jaccard Similarity for 2 lists of (hashed) shingles.
    :param list1: List 1 of hashed shingles (doc 1)
    :param list2: List 2 of hashed shingles (doc 2)
    :return: The Jaccard Similarity
    """
    s1, s2 = set(list1), set(list2)
    return len(s1 & s2) / len(s1 | s2)


def generate_hashed_shingles(words: [str]):
    """
    Given a list of tokens, converts these tokens into a list of hashed shingles.
    :param words: List of tokens
    :return: A list of hashed shingles
    """
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    hashed_shingles = set()

    for shingle in shingles:
        # Hash to 32-bit integer
        hashed_shingles.add(hash(shingle) & 0xffffffff)

    return hashed_shingles


def doc_to_body(line: str):
    """
    Parses our custom .txt format to a tuple that we can use to analyze the posts.
    :param line: A line of the .txt file we are reading.
    :return: A tuple containing the tokens of the post, together with the PostID.
    """
    line = line.rstrip('§§§\n')
    line = line.split('#')
    return line[2]


def pair_to_jacc_sim(pair: set):
    """
    Calculates the Jaccard Similarity of a pair (set of 2 items)
    :param pair: A set of 2 items
    :return: The Jaccard Similarity for the given pair.
    """
    pair_list = list(pair)
    similarity = jaccard_similarity(pair_list[0], pair_list[1])
    return similarity


def brute_force_spark(sc: SparkSession.sparkContext):
    """
    Our brute force approach implemented on top of Spark.
    :param sc: The Spark Context
    """
    fileName = '../data/crypto.txt'
    docRDD = sc.textFile(fileName)
    bodyRDD = docRDD.sample(False, 0.2).map(doc_to_body)

    # Convert each document body into shingles
    shingleRDD = bodyRDD.map(generate_hashed_shingles)

    # Generate pairs of shingle sets (to later calculate jaccard similarity in next pass)
    combinedRDD = shingleRDD.cartesian(shingleRDD).filter(lambda x: x[1] != x[0])

    # Calculate Jaccard similarities
    similarityRDD = combinedRDD.map(pair_to_jacc_sim)

    similarities = similarityRDD.collect()
    print(len(similarities))
    # Plot histogram (note that we filtered out the 0.0 similarities to get a better view)
    num_bins = 50
    plt.hist(similarities, num_bins, facecolor='blue', alpha=0.5)
    plt.title("20% random sample of Crypto StackExchange posts (Brute force approach in Spark)")
    plt.xlabel("Similarities")
    plt.show()


def main():
    sc = init_spark()
    brute_force_spark(sc)

    sc.stop()


main()
