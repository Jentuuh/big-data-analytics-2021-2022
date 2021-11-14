from lxml import etree
from itertools import islice, chain
import six
import bleach
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')


FILE_TO_PARSE = '../../data/Posts.xml'
OUT_CRYPTO_EXCH = '../../data/crypto.txt'
OUTPUT_RESULTS = '../../data/output/result_data_management'

# Efficient parsing of large XML files from
# http://stackoverflow.com/a/9814580/987185
def parse(fp):
    """Efficiently parses an XML file from the StackExchange data dump and
    returns a generator which yields one row at a time.
    """

    context = etree.iterparse(fp, events=("end",))

    for action, elem in context:
        if elem.tag == "row":
            # processing goes here
            assert elem.text is None, "The row wasn't empty"
            yield elem.attrib

        # cleanup
        # first empty children from current element
        # This is not absolutely necessary if you are also deleting
        # siblings, but it will allow you to free memory earlier.
        elem.clear()
        # second, delete previous siblings (records)
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        # make sure you have no references to Element objects outside the loop


def batch(iterable, size):
    """Creates a batches of size `size` from the `iterable`."""
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        try:
            yield chain([six.next(batchiter)], batchiter)
        except StopIteration:
            return


def filter_out_stop_words(str_to_filter: str):
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'[^\W\d_]+')
    word_tokens = tokenizer.tokenize(str_to_filter)

    filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]

    return filtered_sentence


def find_posts_by_ids(post_ids: [int], input_file_name: str, output_file_name: str):
    with open(input_file_name, "rb") as xml:
        # batch generator object
        row_generator = parse(xml)
        parsed_rows = batch(row_generator, 1)

        f = open(output_file_name, "w")
        counter = 0
        for row in parsed_rows:
            parsed_obj = list(row)[0]

            if parsed_obj['Id'] in post_ids:
                f.write("Body " + str(counter) + ":\n")
                post_body = parsed_obj['Body']
                post_body = post_body.replace('\n', " ")
                f.write(post_body + "\n")
                f.write("\n")
                counter += 1

        f.close()


def parse_xml_file():
    """
    Output file row format: PostID#PostTypeID#Body#OwnerID\n
    """
    with open(FILE_TO_PARSE, "rb") as xml:
        # batch generator object
        row_generator = parse(xml)
        parsed_rows = batch(row_generator, 1)

        f = open(OUT_CRYPTO_EXCH, "w")
        for row in parsed_rows:
            parsed_obj = list(row)[0]

            file_output_str = ""
            for field in parsed_obj:
                if field == 'Id' or field == "PostTypeId" or field == 'Body':
                    if field == 'Body':
                        # Clean out HTML code
                        html_cleaned = bleach.clean(parsed_obj[field], tags=[], strip=True)
                        stop_words_cleaned = filter_out_stop_words(html_cleaned)
                        stop_words_cleaned = filter(lambda x: len(x) > 2, stop_words_cleaned)
                        stop_words_cleaned_str = " ".join(stop_words_cleaned)
                        file_output_str += stop_words_cleaned_str

                    else:
                        file_output_str += parsed_obj[field]
                    file_output_str += "#"

            file_output_str = file_output_str.removesuffix('#')
            f.write(file_output_str + "§§§\n")

        f.close()


#parse_xml_file()
