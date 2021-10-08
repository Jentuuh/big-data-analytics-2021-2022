import itertools
import xml.sax
from time import process_time


dataset = []
prev_author_dict = {}
author_dict = {}
count_dict = {}
pairs_hash_table = {}

threshold = 5
BUCKET_COUNT = 1500000


def run_through_and_decide_longest_author_list(dataset):
    largest_list_size = 0
    for authorlist in dataset:
        if len(authorlist) >= largest_list_size:
            largest_list_size = len(authorlist)
    print("Largest list of authors: " + str(largest_list_size))


def print_max_freq_output(max_count, max_freq_sets, curr_pass, candidates):
    print("Max frequency count pass " + str(curr_pass) + ": " + str(max_count))
    print("Max frequent sets pass " + str(curr_pass) + ": ", end="")
    print(max_freq_sets)
    print("Candidates pass " + str(curr_pass) + ": ", end="")
    print(candidates)



# def get_max_freq_sets(current_freq_sets, prev_freq_sets):
#     max_freq_sets = []
#     for prev in prev_freq_sets:
#         add = True
#         for curr in current_freq_sets:
#             if isinstance(prev, str):
#                 prev = frozenset([prev])
#             if prev.issubset(curr):
#                 add = False
#                 break
#         if add:
#             max_freq_sets.append(prev)
#     return max_freq_sets


# def filter(curr_comb):
#     print(pairs_hash_table.keys())
#     for item in curr_comb:
#         hash_value = generate_hash(item)
#         if hash_value in pairs_hash_table.keys() and pairs_hash_table[hash_value] >= threshold:
#             print('Candidate found!')

# Note that hash is undeterministic!
def generate_hash_v1(item):
    return abs(hash(item)) % BUCKET_COUNT


class PaperHandler(xml.sax.ContentHandler):
       def __init__(self):
          self.CurrentData = ""
          self.article = ""
          self.authors = []
          self.title = ""
          self.journal = ""
          self.year = ""
          self.currentAuthor = ""

       def startElement(self, tag, attributes):
            self.CurrentData = tag

        # Call when an elements ends
       def endElement(self, tag):
            self.CurrentData = ""
            if tag == "article" \
            or tag == "incollection" \
            or tag == "inproceedings" \
            or tag == "phdthesis" \
            or tag == "www":
                dataset.append(frozenset(self.authors))
                self.authors = []
            elif tag == "author":
                self.authors.append(self.currentAuthor)
                self.currentAuthor = ""

        # Call when a character is read
       def characters(self, content):
          if self.CurrentData == "author":
             self.currentAuthor += content


if __name__ == "__main__":
    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    Handler = PaperHandler()
    parser.setContentHandler( Handler )

    starttime = process_time()
    parser.parse("../data/dblp.xml")
    print("Time to parse dataset : {0} s".format(process_time() - starttime))

    # First pass
    for authorlist in dataset:
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

    candidates = []
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

    curr_pass = 2
    while len(candidates) > 0:
        starttime = process_time()
        max_freq_sets = []

        curr_combinations = list(map(frozenset, itertools.combinations(candidates, curr_pass)))
        print("Candidates before PCY optimization: " + str(len(curr_combinations)))

        # PCY: We reduce the amount of candidate pairs
        if curr_pass == 2:
            curr_combinations = list(filter(lambda candidate: generate_hash_v1(candidate) in pairs_hash_table.keys() and
                                                              pairs_hash_table[generate_hash_v1(candidate)] >= threshold,
                                                              curr_combinations))
        print("Candidates after PCY optimization: " + str(len(curr_combinations)))
        print("Time to make combinations : {0} s".format(process_time() - starttime))

        starttime = process_time()

        candidates = []
        pairs_hash_table = {}
        count_dict = {}

        # We loop through the dataset ONCE and count the occurences of our list of candidates
        for authorlist in itertools.islice(dataset, len(dataset)):
            for comb in curr_combinations:
                if comb.issubset(authorlist):
                    if comb in count_dict.keys():
                        count_dict[comb] += 1
                    else:
                        count_dict[comb] = 1

        print("Time to go through dataset : {0} s".format(process_time() - starttime))

        # Check which items are frequent in this pass (and thus candidates for next pass)
        max_count = 0

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

        candidates = list(set(candidates))

        print_max_freq_output(max_count, max_freq_sets, curr_pass, candidates)
        curr_pass += 1

    print(process_time())



