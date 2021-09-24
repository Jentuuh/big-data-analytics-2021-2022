import xml.sax

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
            if tag == "article":
                 print("Article parsed")
            elif tag == "incollection":
                print("Incollection parsed")
            elif tag == "inproceedings":
                print("Inproceedings parsed")
            elif tag == "phdthesis":
                print("Thesis parsed")
            elif tag == "www":
                print("Website parsed")
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
                print("Authors:", self.authors)
                self.authors = []

        # Call when a character is read
       def characters(self, content):
          if self.CurrentData == "author":
             self.authors.append(content)
          elif self.CurrentData == "title":
             self.title = content
          elif self.CurrentData == "year":
             self.year = content
          elif self.CurrentData == "journal":
            self.journal = content

if __name__ == "__main__":
   # create an XMLReader
   parser = xml.sax.make_parser()
   # turn off namepsaces
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)
   # override the default ContextHandler
Handler = PaperHandler()
parser.setContentHandler( Handler )
parser.parse("dblp.xml")