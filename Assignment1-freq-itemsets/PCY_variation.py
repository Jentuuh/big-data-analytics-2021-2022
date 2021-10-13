import itertools
import xml.sax
from time import process_time

# Our script for the PCY with extra optimizations approach.
# Before running this file, you should preprocess the dataset by running the 'parser.py' script. This will transform
# the XML data into our own custom format to speed up the parsing process. For us, the input file for this script is
# located at '../data/dblp.txt', but you can change this to your own preference right below this comment.

# Jente Vandersanden and Ingo Andelhofs, Big Data Analytics 2021 - 2022, Hasselt University.

INPUT_FILE_PATH = '../data/dblp.txt'
dataset = []
author_dict = {}
count_dict = {}
pairs_hash_table = {}

threshold = 300
BUCKET_COUNT = 5000000000


def dataset_to_file(dataset_file, filename):
    f = open(filename, "w")

    for data in dataset_file:
        formatted = "#".join(data)
        if formatted == "":
            continue
        f.write(formatted + "\n")

    f.close()


def process_dataset(filename):
    f = open(filename, "r")

    count_dict = {}
    pairs_hash_table = {}
    candidates = []
    curr_pass = 1
    
    # First pass 
    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("\n", "")
        line_data = line.split('#')

        # We don't have to consider the lines with a smaller amount of authors than k 
        if curr_pass > len(line_data):
            continue
        first_pass(line_data, count_dict, pairs_hash_table)
    f.close()

    # Find frequent items from first pass
    count_tuples(count_dict, threshold, curr_pass, candidates)

    # Other passes
    curr_pass = 2
    while len(candidates) > 0:
        starttime = process_time()

        curr_combinations = list(map(frozenset, itertools.combinations(candidates, curr_pass)))
        print("Candidates before PCY optimization: " + str(len(curr_combinations)))

        # PCY: We reduce the amount of candidate pairs
        curr_combinations = list(filter(lambda candidate: generate_hash_v1(candidate) in pairs_hash_table.keys() and
                                                          pairs_hash_table[generate_hash_v1(candidate)] >= threshold,
                                                          curr_combinations))

        print("Candidates after PCY optimization: " + str(len(curr_combinations)))
        print("Time to make combinations : {0} s".format(process_time() - starttime))

        starttime = process_time()

        candidates = []
        count_dict = {}

        # Counting frequent items for current pass
        count_curr_pass(filename, curr_pass, curr_combinations, count_dict)

        print("Time to go through dataset : {0} s".format(process_time() - starttime))
        
        # Check which items are frequent in this pass (and thus candidates for next pass)
        count_tuples(count_dict, threshold, curr_pass, candidates)

        candidates = list(set(candidates))

        curr_pass += 1 
    return 


def first_pass(authorlist, count_dict, pairs_hash_table):
    pairs = list(map(frozenset, itertools.combinations(authorlist, 2)))
    for author in authorlist:
        if author in count_dict.keys():
            count_dict[author] += 1
        else:
            count_dict[author] = 1
    # Hash pairs into buckets
    for pair in pairs:
        hash_value = generate_hash_v1(pair)
        if hash_value in pairs_hash_table.keys():
            pairs_hash_table[hash_value] += 1
        else:
            pairs_hash_table[hash_value] = 1


def count_curr_pass(data_filename, curr_pass, candidate_combinations, count_dict):
    f = open(data_filename, "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("\n", "")
        line_data = line.split('#')
        line_data_set = set(line_data)

        # We don't have to consider the lines with a smaller amount of authors than k
        if curr_pass > len(line_data):
            continue

        for comb in candidate_combinations:
            if comb.issubset(line_data_set):
                if comb in count_dict.keys():
                    count_dict[comb] += 1
                else:
                    count_dict[comb] = 1

        # Extra optimization: PCY hashing step also for 3-tuples (triples) --> Reduce candidates in pass 3
        if len(line_data) < 50 and curr_pass < 3:
            hash_tuples = list(map(frozenset, itertools.combinations(line_data, curr_pass + 1)))
            # Hash tuples into buckets (to reduce candidates in next pass)
            for hash_tuple in hash_tuples:
                hash_value = generate_hash_v1(hash_tuple)
                if hash_value in pairs_hash_table.keys():
                    pairs_hash_table[hash_value] += 1
                else:
                    pairs_hash_table[hash_value] = 1
    f.close()


def count_tuples(count_dict, threshold, curr_pass, candidates):
    max_count = 0
    max_freq_sets = []
    # Check which items are frequent in the first pass
    for key in count_dict.keys():
        if count_dict[key] >= threshold:
            candidates.append(key)
        if count_dict[key] > max_count:
            # Update the max freq count and reset list of max freq sets
            max_count = count_dict[key]
            max_freq_sets = []
            max_freq_sets.append(key)
        elif count_dict[key] == max_count:
            # Found a new set of the current max frequency, add it to the list
            max_freq_sets.append(key)
            
    print_max_freq_output(max_count, max_freq_sets, curr_pass, candidates)


def print_max_freq_output(max_count, max_freq_sets, curr_pass, candidates):
    print("Max frequency count pass " + str(curr_pass) + ": " + str(max_count))
    print("Max frequent sets pass " + str(curr_pass) + ": ", end="")
    print(max_freq_sets)


# Note that hash is undeterministic!
def generate_hash_v1(item):
    return abs(hash(item)) % BUCKET_COUNT


if __name__ == "__main__":
    starttime = process_time()

    process_dataset(INPUT_FILE_PATH)
    print("Total time to parse and process dataset : {0} s".format(process_time() - starttime))
