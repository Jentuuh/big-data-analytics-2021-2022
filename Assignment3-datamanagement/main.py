# This is needed to start a Spark session from the notebook
# You may adjust the memory used by the driver program based on your machine's settings
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = "--conf spark.driver.memory=8g  pyspark-shell"
from pyspark.sql import SparkSession
import random

INPUT_FILE_PATH = '../data/crypto.txt'


def parse_dataset_from_file(file_name: str, sample_size: int):

    posts = []

    f = open(file_name, "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("§§§\n", "")
        line_data = line.split('#')
        posts.append({'id': line_data[0], 'body': line_data[2]})

    f.close()

    if len(posts) >= sample_size:
        return random.sample(posts, sample_size)
    else:
        return posts


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


def main():
    init_spark()
    return


main()