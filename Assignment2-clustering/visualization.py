from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS


PATH = "/Users/jentevandersanden/Desktop/BDA/BigDataAnalytics-assignments/data/output/2021-11-01 " \
       "12:07:01.143847/1969-1979.txt"

filter_list = ['a', 'for', 'and', 'of', 'the', 'with', 'in', 'to', '-', 'an', 'on', 'by', 'abstract', 'bibliography',
               'title', 'data', 'base', 'paper', 'sigmod', 'acm', 'sigfidet', 'codasyl']
strip_list = ['"', "'", "(", ")", ";", ",", ":", '.', '/']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def contains_number(word: str):
    for n in numbers:
        if n in word:
            return True
    return False


def count_most_frequent_words_in_cluster(cluster_entries: list[str]):
    wordcounts = defaultdict()
    for point in cluster_entries:
        words = point.split(" ")
        for word in words:
            word = word.strip("".join(strip_list))
            word_lower = word.lower()
            if len(word_lower) == 0:
                continue
            if word_lower in filter_list:
                continue
            if contains_number(word_lower):
                continue
            if word_lower in wordcounts.keys():
                wordcounts[word_lower] += 1
            else:
                wordcounts[word_lower] = 1
    print(sorted(wordcounts.items(), key=lambda k_v: k_v[1], reverse=True)[0:5])


def visualize(words: list[str]):
    stopwords = set(STOPWORDS)
    wc = WordCloud(max_words=1000, stopwords=stopwords, margin=10,
                   random_state=1).generate(words)
    

def main():
    f = open(PATH, "r")

    while True:
        line = f.readline()

        if not line:
            break
        if line[0] == "#":
            continue
        if len(line) <= 1:
            continue

        line = line.replace("\n", "")
        line_data = line.split('§')

        cluster_number = line_data[0]
        clustroid = line_data[1]
        cluster_entries = line_data[2:]
        count_most_frequent_words_in_cluster(cluster_entries)
    f.close()


main()
