import xml.sax
from lxml import etree
import time

# Constants
# PATH_TO_DATASET = "../../data/dblp.xml"
PATH_TO_DATASET = "../../data/dblp50000.xml"


# Parsing
def dataset_to_file(dataset, filename):
    f = open(filename, "w")

    for data in dataset:
        formatted = "#".join(data)

        if formatted == "":
            continue

        f.write(formatted + "\n")

    f.close()


def dataset_from_file(filename):
    dataset = []
    f = open(filename, "r")

    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        dataset.append(line.split("#"))

    f.close()
    return dataset


def parse_formatted():
    start = time.process_time()
    print("START:PARSING: {0}s".format(start))

    dataset = dataset_from_file("../../data/dblp.txt")

    end = time.process_time()
    print("END:PARSING: {0}s".format(end))
    return dataset


def parse():
    start = time.process_time()
    print("START:PARSING: {0}s".format(start))

    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    dataset = []
    Handler = PaperHandler(dataset)
    parser.setContentHandler(Handler)

    parser.parse(PATH_TO_DATASET)
    end = time.process_time()
    print("END:PARSING: {0}s".format(end))

    return Handler.dataset


def parse_etree():
    file = "../../data/dblp.xml"

    data = []

    context = etree.iterparse(file, dtd_validation=True, events=('start', 'end'))
    # print(context.root.tag)

    data = []
    authors = []
    depth = 0

    for action, elem in context:
        if action == 'start':
            depth += 1

        if action == 'end':
            depth -= 1
            
            if depth == 1:
                data.append(authors)
                authors = []

        if action == 'start' and elem.tag == 'author':
            authors.append(elem.text)
            # data.append(depth)

        # data.append(elem.tag)

    print(data[0:20])
    # etree.DTD('../../data/dblp.dtd')
    # # parser = etree.XMLParser(dtd_validation=True)
    # tree = etree.iterparse("../../data/dblp.xml", dtd_validation=True)

    # root = tree.getroot()

    # data = []

    # for entry in root:
    #     authors = []
    #     xml_authors = entry.findall('author')

    #     for xml_author in xml_authors:
    #         authors.append(xml_author.text)

    #     data.append(authors)

    # print(data[0:35])


class PaperHandler(xml.sax.ContentHandler):
    def __init__(self, dataset):
        self.dataset = dataset
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
            self.dataset.append(self.authors)
            self.authors = []
        elif tag == "author":
            self.authors.append(self.currentAuthor)
            self.currentAuthor = ""

    # Call when a character is read
    def characters(self, content):
        if self.CurrentData == "author":
            self.currentAuthor += content
