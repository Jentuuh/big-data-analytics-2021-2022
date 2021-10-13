import xml.sax

# Converts the dataset to a custom format in order to speed up the parsing process during mining the dataset.
# Running this script with file ../../data/dblp.xml will result in a .txt-file ../../data/dblp.txt . This file
# has our custom format and can be used to run the FP-growth script.

# Jente Vandersanden and Ingo Andelhofs, Big Data Analytics 2021 - 2022, Hasselt University.

DBLP = "../../data/dblp.xml"
DBLP_SUB = "../../data/dblp50000.xml"
DBLP_SUBSUB = "../../data/dblp.1000.xml"

OUT_DBLP = "../../data/dblp_clustering.txt"
OUT_DBLP_SUB = "../../data/dblp50000_clustering.txt"
OUT_DBLP_SUBSUB = "../../data/dblp.1000_clustering.txt"


def dataset_to_file(dataset) -> None:
    f = open(OUT_DBLP_SUB, "w")

    for entry in dataset:
        formatted = "#".join(entry)

        if formatted == "":
            continue

        f.write(formatted + "\n")

    f.close()


def parse() -> list:
    # Sax parser
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    handler = ContentHandler()
    parser.setContentHandler(handler)
    parser.parse(DBLP_SUB)

    return handler.dataset


class ContentHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.dataset = []
        self.authors = []

        self.currentTitle = ''
        self.currentYear = ''
        self.isGraphicsPublication = False

        self.tag = ''
        self.chars = ''
        self.depth = 0

    def startElement(self, tag, _):
        self.tag = tag
        self.depth += 1
        self.chars = ''

    def endElement(self, tag):
        self.depth -= 1

        if '/siggraph' in self.chars or '/cgi' in self.chars or '/cgf' in self.chars or '/cg' in self.chars:
            self.isGraphicsPublication = True

        # An author tag ended
        if tag == 'title':
            self.currentTitle = self.chars
        if tag == 'year':
            self.currentYear = self.chars

        # End of entry (depth of a given entry)
        if self.depth == 1 and self.isGraphicsPublication:
            self.dataset.append([self.currentTitle, self.currentYear])
            self.currentTitle = ''
            self.currentYear = ''
            self.isGraphicsPublication = False

    def characters(self, content):
        self.chars += content


# Main
print("Parsing dataset...")
dataset = parse()
dataset_to_file(dataset)
print("Finished.")

