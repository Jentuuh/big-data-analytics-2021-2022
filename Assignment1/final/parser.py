import xml.sax

DBLP = "../../data/dblp.xml"
DBLP_SUB = "../../data/dblp50000.xml"
DBLP_SUBSUB = "../../data/dblp.1000.xml"

OUT_DBLP = "../../data/dblpw.txt"
OUT_DBLP_SUB = "../../data/dblp50000.txt"
OUT_DBLP_SUBSUB = "../../data/dblp.1000.txt"


def dataset_to_file(dataset) -> None:
    f = open(OUT_DBLP, "w")

    for data in dataset:
        formatted = "#".join(data)

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
    parser.parse(DBLP)

    return handler.dataset


class ContentHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.dataset = []
        self.authors = []

        self.tag = ''
        self.chars = ''
        self.depth = 0

    def startElement(self, tag, _):
        self.tag = tag
        self.depth += 1
        self.chars = ''

    def endElement(self, tag):
        self.depth -= 1

        # An author tag ended
        if tag == 'author':
            self.authors.append(self.chars)

        # End of entry (depth of a given entry)
        if self.depth == 1 and self.authors != []:
            self.dataset.append(self.authors)
            self.authors = []

    def characters(self, content):
        self.chars += content


# Main
dataset = parse()
dataset_to_file(dataset)
