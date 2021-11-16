import os
from collections import defaultdict
from pyspark.sql import SparkSession
import numpy as np
import sys
from time import time
from preprocessing import parser

sys.setrecursionlimit(2 ** 27)
os.environ['PYSPARK_SUBMIT_ARGS'] = "--conf spark.driver.memory=8g  pyspark-shell"

# Our LSH approach implemented on top of Spark. Before running this script, make sure you parsed the XML using the
# 'preprocessing/parser.py' script.

# If desired, a supplementary comparison of the results with the Jaccard Similarity can be turned on by setting
# RUN_TEST_WITH_JACCARD_SIMILARITY to True. To write the content of the posts that were found to be candidates to
# output files, set OUTPUT_POST_CONTENT to True.

# Jente Vandersanden and Ingo Andelhofs, Big Data Analytics 2021 - 2022, Hasselt University.

INPUT_FILE_PATH = '../data/crypto.txt'
SHINGLE_SIZE = 5
NUM_HASH_FUNCTIONS = 100
BANDS = 16
THRESHOLD = 0.4
AMOUNT_BUCKETS = 100000
INF = 2 ** 32

RUN_TEST_WITH_JACCARD_SIMILARITY = False
OUTPUT_POST_CONTENT = False


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
    print(s1)
    print(s2)
    return len(s1 & s2) / len(s1 | s2)


def generate_hashed_shingles_with_id(element: tuple):
    """
    Given a tuple consisting of a string of tokens and the PostID, converts these tokens into a list of hashed shingles.
    :param element: Tuple containing the token string and PostID
    :return: A tuple containing a list of hashed shingles, and the PostID
    """
    words = element[0].split(' ')
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    hashed_shingles = set()

    for shingle in shingles:
        # Hash to 32-bit integer
        hashed_shingles.add(hash(shingle) & 0xffffffff)

    # Hashed Shingles, PostID
    return hashed_shingles, element[1]


def generate_hashed_shingles(element: tuple):
    """
    Given a tuple consisting of a string of tokens and the PostID, converts these tokens into a list of hashed shingles.
    :param element: Tuple containing the token string and PostID
    :return: A list of hashed shingles
    """
    words = element[0].split(' ')
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    hashed_shingles = set()

    for shingle in shingles:
        # Hash to 32-bit integer
        hashed_shingles.add(hash(shingle) & 0xffffffff)

    # Hashed Shingles
    return hashed_shingles


def doc_to_body(line: str):
    """
    Parses our custom .txt format to a tuple that we can use to analyze the posts.
    :param line: A line of the .txt file we are reading.
    :return: A tuple containing the tokens of the post, together with the PostID.
    """
    line = line.rstrip('§§§\n')
    line = line.split('#')

    # Tokens, PostID
    return line[2], line[0]


def hash_generator(amount: int, modulo: int):
    """
    Generates an amount of hash functions that are able to map values to an interval [0:modulo].
    :param amount: The amount of hash functions to be generated.
    :param modulo: Defines the range of the interval to which values can be hashed.
    :return: yields `amount` hash functions.
    """
    h1 = lambda x: (3491 * x + 5003) % modulo
    h2 = lambda x: (1999 * x + 1409) % modulo

    for i in range(amount):
        yield lambda value: (h1(value) + i * h2(value)) % modulo


def find_min_hash(list_to_hash: tuple, num_rows: int):
    """
    For a list of shingles (document), executes a number of hash functions on these shingles and appends the minimal
    hash value per hash function to the signature. This results in finding one column of the signature matrix.
    :param list_to_hash: The list of shingles (document) to hash.
    :param num_rows: The amount of rows in the sparse matrix (the amount of unique shingles)
    :return: Returns a tuple of the resulting column of the signature matrix, together with the PostID belonging to that
             column.
    """
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


def lsh_hash(signature_column: tuple):
    """
    Hashes a signature matrix' column (corresponding to a document/post) to a bucket's hash key.
    :param signature_column: The signature matrix' column, containing a list of hashed shingles (split into bands) and
                             a PostId.
    :return: The bucket hash keys for each column-segment (band) of that document (signature-column)
    """
    hashed_values = []
    for i in signature_column[0]:
        entry_i = {"bucket": hash(tuple(i.tolist())) % AMOUNT_BUCKETS, "postId": signature_column[1]}
        hashed_values.append(entry_i)

    # Hashed bucket index, PostID
    return hashed_values


def fold_to_one_dict(list_of_dicts: np.array(tuple)):
    """
    Combines a list of dictionaries into one dictionary containing each element's entries.
    :param list_of_dicts: A list containing dictionaries
    :return: A list of dictionaries
    """
    result_dict = defaultdict(list)

    for entry in list_of_dicts:
        result_dict[entry['bucket']].append(entry['postId'])

    return result_dict


def filter_out_singles(sim_dict: defaultdict):
    """
    Filters out buckets which only have 1 entry from a dictionary.
    :param sim_dict: The dictionary to be filtered
    :return: The filtered dictionary
    """
    reduced_d = {key: val for key, val in sim_dict.items() if len(val) > 1}
    return reduced_d


def extract_similar_candidates(sim_dict_per_band: [defaultdict]):
    """
    Function that handles the final result transformations extracts the similar candidates for the LSH algorithm
    :return similar_candidates: A list of frozensets, in which each frozenset contains a group of candidates to be
                                similar to each other.
    """
    print("Extracting similar candidates...")
    # For each set of candidates, count in how many bands it occurs
    count_sim_dict = defaultdict(int)
    for sim_dict in sim_dict_per_band:
        for entry in sim_dict.values():
            count_sim_dict[frozenset(entry)] += 1

    # Check which sets of candidates have an occurence above the threshold
    similar_candidates = []
    for key, value in count_sim_dict.items():
        if value / BANDS >= THRESHOLD:
            similar_candidates.append(key)

    return similar_candidates


def test_with_jaccard_similarity(shingles_per_id: [tuple], similar_candidates: [frozenset]):
    """
    A test function that we wrote to compare our LSH results (which are an approximation of possible candidates, with
    possible false positives/negatives) with the actual Jaccard Similarities for the found candidates.
    :param shingles_per_id: A list of tuples, in which each tuple contains a PostId, and the hashed shingles for that Id
    :param similar_candidates: Possible candidates for similarity that we found with the LSH algorithm
    """
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


def spark_lsh(sc: SparkSession.sparkContext):
    """
    Our implementation of LSH on top of Spark.
    :param sc: The Spark Context
    """
    docRDD = sc.textFile(INPUT_FILE_PATH)

    # ---------------- SHINGLING -----------------
    bodyRDD = docRDD.sample(False, 1).map(doc_to_body)
    print("Amount documents: ", bodyRDD.count())
    print("Making Shingles...")
    shingleRDD = bodyRDD.map(generate_hashed_shingles_with_id)

    # Convert each document body into shingles
    flatUniqueShingleRDD = bodyRDD.flatMap(generate_hashed_shingles).distinct()

    # Calculate amount of unique shingles and documents
    amount_unique_shingles = flatUniqueShingleRDD.count()

    # ---------------- MIN HASHING -----------------
    print("Min Hashing...")
    signatureRDD = shingleRDD.map(lambda e: find_min_hash(e, amount_unique_shingles))

    # -------------------- LSH ---------------------
    print("Executing the LSH algorithm...")
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
    similar_candidates = extract_similar_candidates(sim_dict_per_band)
    print("Similar Candidates: ", similar_candidates)

    # ------------- RESULT VERIFICATION -------------

    # Finds the actual posts that are found to be candidates in the original XML file and writes them to a .txt output
    # file.
    if OUTPUT_POST_CONTENT:
        for i, candidate in enumerate(similar_candidates):
            parser.find_posts_by_ids(list(candidate), '../data/Posts.xml',
                                     '../data/output/result_data_management' + str(i) + '.txt')

    # Calculates the actual Jaccard Similarity for the candidates that were found
    if RUN_TEST_WITH_JACCARD_SIMILARITY:
        shingles_per_id = shingleRDD.collect()
        test_with_jaccard_similarity(shingles_per_id, similar_candidates)
    return


def main():
    start_time = time()
    sc = init_spark()
    spark_lsh(sc)

    sc.stop()
    total_time = time() - start_time
    print("Total runtime: ", total_time, "s")


main()
