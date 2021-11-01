from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import random

# A helper script to visualize the clustering results into a word cloud (per time period).
# Make sure the clustering has already been done (by executing k_means_custom.py) before running this script. Also
# beware of the directory structure used in this file to read in the clustering results.

# Jente Vandersanden and Ingo Andelhofs, Big Data Analytics 2021 - 2022, Hasselt University.

DIRECTORYPATH = "../data/output/2021-11-01 14:20:09.454679/"

filter_list = ['a', 'for', 'and', 'of', 'the', 'with', 'in', 'to', '-', 'an', 'on', 'by', 'abstract', 'bibliography',
               'title', 'data', 'base', 'bases', 'paper', 'sigmod', 'acm', 'sigfidet', 'codasyl', 'at', 'using',
               'databases', 'database', 'efficient', 'system', 'systems', 'contents', 'content', "editor's",
               "conference", "management", "proceedings", "new"]
strip_list = ['"', "'", "(", ")", ";", ",", ":", '.', '/']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def cluster_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    color = int(word.split(" ")[1]) * 360/50
    return "hsl({color}, 100%, 50%)".format(color=color)


def contains_number(word: str):
    for n in numbers:
        if n in word:
            return True
    return False


def visualize(wordcountsdict: dict, start_yr: int):
    stopwords = set(STOPWORDS)
    wc = WordCloud(max_words=1000, stopwords=stopwords, margin=10,
                   random_state=1).generate_from_frequencies(wordcountsdict)
    wc = wc.recolor(color_func=cluster_color_func, random_state=3)
    default_colors = wc.to_array()
    plt.title("Result clustering periode: " + str(start_yr) + "-" + str(start_yr + 10))
    plt.imshow(default_colors, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("../data/output/clustering_word_clouds/" + str(start_yr) + "-" + str(start_yr + 10) + ".png")



def count_most_frequent_words_in_cluster(cluster_entries: list[str], cluster_number: int):
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
            word_lower += " " + str(cluster_number)
            if word_lower in wordcounts.keys():
                wordcounts[word_lower] += 1
            else:
                wordcounts[word_lower] = 1

    return wordcounts


def main():
    result_dict = defaultdict()

    start_year = 1969
    max_year = 2021

    while start_year <= max_year:
        FILEPATH = str(start_year) + "-" + str(start_year + 10) + ".txt"
        f = open(DIRECTORYPATH + FILEPATH, "r")
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
            cluster_entries = line_data[2:]
            word_count_dict = count_most_frequent_words_in_cluster(cluster_entries, int(cluster_number))
            result_dict = result_dict | word_count_dict
        visualize(result_dict, start_year)
        f.close()
        start_year += 5


main()
