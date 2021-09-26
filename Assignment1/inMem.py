import itertools
import xml.sax
import pandas as pd
from time import process_time
import seaborn as sns
import matplotlib.pyplot as plt


dataset = []
prev_author_dict = {}
author_dict = {}
group_size = 1

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
            # if tag == "article":
            #      print("Article parsed")
            # elif tag == "incollection":
            #     print("Incollection parsed")
            # elif tag == "inproceedings":
            #     print("Inproceedings parsed")
            # elif tag == "phdthesis":
            #     print("Thesis parsed")
            # elif tag == "www":
            #     print("Website parsed")
                 # title = attributes["title"]
                 # print("Title:", title)

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
    print(process_time())

    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # override the default ContextHandler

    Handler = PaperHandler()
    parser.setContentHandler( Handler )

    parser.parse("dblp.xml")

    while group_size < 5:
        print(process_time())

        # for i in sorted(author_dict.values()):
        #     print(author_dict[i])
        for authorlist in dataset:
            if not group_size == 1:
                prev_combinations = list(map(frozenset, itertools.combinations(authorlist, group_size - 1)))
                eligible_authors = []
                for combination in prev_combinations:
                    if combination in prev_author_dict.keys() and prev_author_dict[combination] > threshold:
                        eligible_authors.extend(list(combination))
                eligible_authors = list(set(eligible_authors))
            else:
                eligible_authors = authorlist

            combinations = list(map(frozenset, itertools.combinations(eligible_authors, group_size)))
            for combination in combinations:

                if combination in author_dict:
                    author_dict[combination] += 1
                else:
                    author_dict[combination] = 1


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
        group_size += 1
        prev_author_dict = author_dict
        author_dict = {}


