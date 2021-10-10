import itertools
import xml.sax
from time import process_time

dataset = []
prev_author_dict = {}
author_dict = {}
count_dict = {}
pairs_hash_table = {}
is_subset_dict = {}

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
    count_tuples(count_dict, threshold, curr_pass, candidates)
    
    curr_pass = 2
    # Other passes
    while len(candidates) > 0:
        starttime = process_time()
        max_freq_sets = []

        curr_combinations = []

        # During pass 2 we generate the pairs on the go and check if they need to stay,
        # otherwise they won't fit in memory
        # if curr_pass == 2:
        #     f = open(filename, "r")
        #
        #     check_candidates = set(candidates)
        #     while True:
        #         line = f.readline()
        #
        #         if not line:
        #             break
        #         line = line.replace("\n", "")
        #         line_data = line.split('#')
        #
        #         frequents = set(line_data).intersection(check_candidates)
        #         pairs = list(map(frozenset, itertools.combinations(frequents, 2)))
        #
        #         if len(pairs) > 0:
        #             pairs = list(filter(lambda candidate: generate_hash_v1(candidate) in pairs_hash_table.keys() and
        #                                                   pairs_hash_table[generate_hash_v1(candidate)] >= threshold,
        #                                                   pairs))
        #             curr_combinations.extend(pairs)
        #     f.close()
        #
        # else:
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

        f = open(filename, "r")
        
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

            for comb in curr_combinations:
                if comb.issubset(line_data_set):
                    if comb in count_dict.keys():
                        count_dict[comb] += 1
                    else:
                        count_dict[comb] = 1

            if len(line_data) < 50:
                hash_tuples = list(map(frozenset, itertools.combinations(line_data, curr_pass + 1)))
                # Hash tuples into buckets (to reduce candidates in next pass)
                for hash_tuple in hash_tuples:
                    hash_value = generate_hash_v1(hash_tuple)
                    if hash_value in pairs_hash_table.keys():
                        pairs_hash_table[hash_value] += 1
                    else:
                        pairs_hash_table[hash_value] = 1
                hash_tuples = []
        f.close()

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

        # We make another dict to fastly check if a combination is a subset of an authorlist
        extension = authorlist.extend(pair)
        hash_value2 = generate_hash_v1(extension)
        if hash_value2 in is_subset_dict.keys():
            is_subset_dict[hash_value2] += 1
        else:
            is_subset_dict[hash_value2] = 1

            
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
    # print("Candidates pass " + str(curr_pass) + ": ", end="")
    # print(candidates)


# Note that hash is undeterministic!
def generate_hash_v1(item):
    return abs(hash(item)) % BUCKET_COUNT

if __name__ == "__main__":

    starttime = process_time()

    process_dataset('../data/dblp.txt')
    print("Total time to parse and process dataset : {0} s".format(process_time() - starttime))

    curr_pass = 1

    # while curr_pass < 2177:
    #     candidates = []
    #     max_count = 0
    #     max_freq_sets = []
    # 
    #     # First pass
    #     for authorlist in dataset:
    #         pairs = list(map(frozenset, itertools.combinations(authorlist, 2)))
    #         for author in authorlist:
    #             if author in count_dict.keys():
    #                 count_dict[author] += 1
    #             else:
    #                 count_dict[author] = 1
    #         # Hash pairs into buckets
    #         for pair in pairs:
    #             hash_value = generate_hash_v1(pair)
    #             if hash_value in pairs_hash_table.keys():
    #                 pairs_hash_table[hash_value] += 1
    #             else:
    #                 pairs_hash_table[hash_value] = 1
    # 
    #     # Check which items are frequent in the first pass
    #     for key in count_dict.keys():
    #         if count_dict[key] >= threshold:
    #             candidates.append(key)
    #         if count_dict[key] > max_count:
    #             # Update the max freq count and reset list of max freq sets
    #             max_count = count_dict[key]
    #             max_freq_sets = []
    #             max_freq_sets.append(key)
    #         elif count_dict[key] == max_count:
    #             # Found a new set of the current max frequency, add it to the list
    #             max_freq_sets.append(key)
    # 
    #     print_max_freq_output(max_count, max_freq_sets, 1, candidates)
    # 
    #     curr_pass = 2
    #     while len(candidates) > 0:
    #         starttime = process_time()
    #         max_freq_sets = []
    # 
    #         curr_combinations = list(map(frozenset, itertools.combinations(candidates, curr_pass)))
    #         print("Candidates before PCY optimization: " + str(len(curr_combinations)))
    # 
    #         # PCY: We reduce the amount of candidate pairs
    #         if curr_pass == 2:
    #             curr_combinations = list(filter(lambda candidate: generate_hash_v1(candidate) in pairs_hash_table.keys() and
    #                                                               pairs_hash_table[generate_hash_v1(candidate)] >= threshold,
    #                                                               curr_combinations))
    # 
    #         print("Candidates after PCY optimization: " + str(len(curr_combinations)))
    #         print("Time to make combinations : {0} s".format(process_time() - starttime))
    # 
    #         starttime = process_time()
    # 
    #         candidates = []
    #         pairs_hash_table = {}
    #         count_dict = {}
    # 
    #         # We loop through the dataset ONCE and count the occurences of our list of candidates
    #         for authorlist in itertools.islice(dataset, len(dataset)):
    # 
    #             for comb in curr_combinations:
    #                 if comb.issubset(authorlist):
    #                     if comb in count_dict.keys():
    #                         count_dict[comb] += 1
    #                     else:
    #                         count_dict[comb] = 1
    # 
    #         print("Time to go through dataset : {0} s".format(process_time() - starttime))
    # 
    #         # Check which items are frequent in this pass (and thus candidates for next pass)
    #         max_count = 0
    # 
    #         for key in count_dict.keys():
    #             if count_dict[key] >= threshold:
    #                 candidates.append(key)
    #             if count_dict[key] > max_count:
    #                 # Update the max freq count and reset list of max freq sets
    #                 max_count = count_dict[key]
    #                 max_freq_sets = []
    #                 max_freq_sets.append(key)
    #             elif count_dict[key] == max_count:
    #                 # Found a new set of the current max frequency, add it to the list
    #                 max_freq_sets.append(key)
    # 
    #         candidates = list(set(candidates))
    # 
    #         print_max_freq_output(max_count, max_freq_sets, curr_pass, candidates)
    #         curr_pass += 1
    #     threshold = threshold - threshold*(1/4)
    # 
    # print(process_time())






