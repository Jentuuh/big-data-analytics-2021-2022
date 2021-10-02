import itertools
import xml.sax
import pandas as pd
from time import process_time
import seaborn as sns
import matplotlib.pyplot as plt

dataset = []
prev_author_dict = {}
author_dict = {}
count_dict = {}
pairs_hash_table = {}

threshold = 300


def run_through_and_decide_longest_author_list(dataset):
    largest_list_size = 0
    for authorlist in dataset:
        if len(authorlist) >= largest_list_size:
            largest_list_size = len(authorlist)
    print("Largest list of authors: " + str(largest_list_size))

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
                dataset.append(self.authors)
                self.authors = []
            elif tag == "author":
                self.authors.append(self.currentAuthor)
                self.currentAuthor = ""

        # Call when a character is read
       def characters(self, content):
          if self.CurrentData == "author":
             self.currentAuthor += content


if __name__ == "__main__":
    print(process_time())

    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # override the default ContextHandler

    Handler = PaperHandler()
    parser.setContentHandler( Handler )

    parser.parse("dblp.xml")

    run_through_and_decide_longest_author_list(dataset)
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
            hash_value = hash(pair)
            if str(hash_value) in pairs_hash_table.keys():
                pairs_hash_table[str(hash_value)] += 1
            else:
                pairs_hash_table[str(hash_value)] = 1

    candidates = []

    # Check which items are frequent in the first pass
    for key in count_dict.keys():
        if count_dict[key] >= threshold:
            candidates.append(key)
    print(candidates)

    curr_pass = 2
    while len(candidates) > 0:
        print(process_time())

        curr_combinations = list(map(frozenset, itertools.combinations(candidates, curr_pass)))
        print("Candidates before PCY optimization: " + str(len(curr_combinations)))

        # PCY: We reduce the amount of candidate pairs
        if curr_pass == 2:
            curr_combinations = list(filter(lambda candidate: str(hash(candidate)) in pairs_hash_table.keys() and
                                                              pairs_hash_table[str(hash(candidate))] >= threshold,
                                                              curr_combinations))
        print("Candidates after PCY optimization: " + str(len(curr_combinations)))

        candidates = []
        max_freq_sets = []
        max_count = 0

        count_dict = {}

        # We loop through the dataset ONCE and count the occurences of our list of candidates
        for authorlist in dataset:
            for comb in curr_combinations:
                if all(c in authorlist for c in list(comb)):
                    if comb in count_dict.keys():
                        count_dict[comb] += 1
                    else:
                        count_dict[comb] = 1

        # Check which items are frequent in this pass (and thus candidates for next pass)
        for key in count_dict.keys():
            if count_dict[key] >= threshold:
                candidates.append(key)
            if count_dict[key] >= max_count:
                max_count = count_dict[key]
                max_freq_sets.append(key)

        print("Max frequency count pass " + str(curr_pass) + ": " + str(max_count))
        candidates = list(set(candidates))
        print("Max frequent sets pass " + str(curr_pass) + ": ", end="")
        print(max_freq_sets)
        print("Candidates pass " + str(curr_pass) + ": ", end="")
        print(candidates)
        curr_pass += 1


    # while group_size < 5:
    #     print(process_time())
    #
    #     # for i in sorted(author_dict.values()):
    #     #     print(author_dict[i])
    #
    #     # Loop over authors in memory
    #     for authorlist in dataset:
    #         if not group_size == 1:
    #             # Check which authors are eligible
    #             prev_combinations = list(map(frozenset, itertools.combinations(authorlist, group_size - 1)))
    #             print(len(prev_combinations))
    #             eligible_authors = []
    #             for combination in prev_combinations:
    #                 if combination in prev_author_dict.keys() and prev_author_dict[combination] > threshold:
    #                     eligible_authors.extend(list(combination))
    #             eligible_authors = list(set(eligible_authors))
    #         else:
    #             eligible_authors = authorlist
    #
    #         combinations = list(map(frozenset, itertools.combinations(eligible_authors, group_size)))
    #         for combination in combinations:
    #
    #             if combination in author_dict:
    #                 author_dict[combination] += 1
    #             else:
    #                 author_dict[combination] = 1

    print(process_time())

    summary_dict = {}
    for key in author_dict:
        value = author_dict[key]

        if value in summary_dict:
            summary_dict[value] += 1
        else:
            summary_dict[value] = 1

    ss = dict(sorted(summary_dict.items()))

    df = pd.DataFrame.from_dict(ss, orient="index")

    # df.plot.bar()
    # plt.show()

    # sns.histplot(data=df)
    # plt.savefig("out.png")
    # plt.show()

    print(df)
    # print(ss)
    prev_author_dict = author_dict
    author_dict = {}


