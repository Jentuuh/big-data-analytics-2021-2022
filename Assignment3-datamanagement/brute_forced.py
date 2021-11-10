import os
import random
import hashlib

random.seed(12345)
INPUT_FILE_PATH = '../data/crypto.txt'


def parse_dataset_from_file(file_name: str, sample_size: int):
    posts = []

    f = open(file_name, "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("§§§\n", "")
        line_data = line.split('#')
        posts.append({'id': line_data[0], 'body': line_data[2]})

    f.close()

    if len(posts) >= sample_size:
        return random.sample(posts, sample_size)
    else:
        return posts


def shingle_generator(size, f):
    for i in range(0, len(f) - size + 1):
        yield tuple(f[i:i + size])


def main():
    posts = parse_dataset_from_file(INPUT_FILE_PATH, 1000)

    for post in posts[1:3]:
        post_shingles = [i for i in shingle_generator(5, post['body'].split(' '))]

        for shingle in post_shingles:

            # Hash to 32-bit integer
            print(hash(shingle) & 0xffffffff)

    return


main()
