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

threshold = 500


class PaperHandler( xml.sax.ContentHandler ):
       def __init__(self):
          self.CurrentData = ""
          self.article = ""
          self.authors = []
          self.title = ""
          self.journal = ""
          self.year = ""

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

        # Call when a character is read
       def characters(self, content):
          if self.CurrentData == "author":
             self.authors.append(content)

if __name__ == "__main__":
    # print(process_time())
    #
    # # create an XMLReader
    # parser = xml.sax.make_parser()
    # # turn off namepsaces
    # parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # # override the default ContextHandler
    #
    # Handler = PaperHandler()
    # parser.setContentHandler( Handler )
    #
    # parser.parse("dblp50000.xml")

    f = open("../data/dblp.txt", "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("\n", "")
        line_data = line.split('#')

        dataset.append(line_data)
    f.close()

    # First pass
    for authorlist in dataset:
        pairs = list(map(frozenset, itertools.combinations(authorlist, 2)))

        for author in authorlist:
            if author in count_dict.keys():
                count_dict[author] += 1
            else:
                count_dict[author] = 1

    candidates = []
    max_freq_sets = []
    max_count = 0

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

    print("Max frequency count pass 1: " + str(max_count))
    print("Max frequent sets pass 1: ", end="")
    print(max_freq_sets)

    curr_pass = 2
    while len(candidates) > 0:
        curr_combinations = list(map(frozenset, itertools.combinations(candidates, curr_pass)))
        print("Combinations pass " + str(curr_pass) + ": " + str(len(curr_combinations)))

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
            if count_dict[key] > max_count:
                # Update the max freq count and reset list of max freq sets
                max_count = count_dict[key]
                max_freq_sets = []
                max_freq_sets.append(key)
            elif count_dict[key] == max_count:
                # Found a new set of the current max frequency, add it to the list
                max_freq_sets.append(key)

        print("Max frequency count pass " + str(curr_pass) + ": " + str(max_count))
        candidates = list(set(candidates))
        print("Max frequent sets pass " + str(curr_pass) + ": ", end="")
        print(max_freq_sets)
        print("Candidates pass " + str(curr_pass) + ": ", end="")
        print(candidates)
        curr_pass += 1

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
    group_size += 1