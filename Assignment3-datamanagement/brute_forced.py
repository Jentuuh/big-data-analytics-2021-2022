import random
from matplotlib import pyplot as plt
from collections import defaultdict

INPUT_FILE_PATH = '../data/crypto.txt'
SHINGLE_SIZE = 5
SAMPLE_SIZE = 10000

seed = random.randint(0, 10000)
random.seed(seed)
print("Seed: " + str(seed))


def parse_dataset_from_file(file_name: str, sample_size: int):
    posts = []

    f = open(file_name, "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("§§§\n", "")
        line_data = line.split('#')
        post = {'id': line_data[0], 'body': line_data[2].split(' ')}
        if len(post['body']) >= SHINGLE_SIZE:
            posts.append(post)

    f.close()

    if len(posts) >= sample_size:
        return random.sample(posts, sample_size)
    else:
        return posts


def shingle_generator(size: int, f: [str]):
    for i in range(0, len(f) - size + 1):
        yield tuple(f[i:i + size])


def jaccard_similarity(list1: [int], list2: [int]):
    s1, s2 = set(list1), set(list2)
    return len(s1 & s2) / len(s1 | s2)


def generate_hashed_shingles(words: [str]):
    shingles = [i for i in shingle_generator(SHINGLE_SIZE, words)]
    hashed_shingles = set()

    for shingle in shingles:
        # Hash to 32-bit integer
        hashed_shingles.add(hash(shingle) & 0xffffffff)

    return hashed_shingles


def main():
    posts = parse_dataset_from_file(INPUT_FILE_PATH, SAMPLE_SIZE)
    similarities = []  # [0] * 1000 ** 2 - 1000

    for i, post1 in enumerate(posts):

        post1_hashed_shingles = generate_hashed_shingles(post1['body'])

        for post2 in posts[i:]:

            if post1 != post2:
                post2_hashed_shingles = generate_hashed_shingles(post2['body'])
                jacc_sim = jaccard_similarity(post1_hashed_shingles, post2_hashed_shingles)
                # if jacc_sim != 0.0:
                #     print(jacc_sim)
                if jacc_sim > 0.02:
                    print(post1['body'])
                    print(post2['body'])
                if jacc_sim != 0.0:
                    similarities.append(jacc_sim)

    # Plot histogram (note that we filtered out the 0.0 similarities to get a better view)
    num_bins = 50
    n, bins, patches = plt.hist(similarities, num_bins, facecolor='blue', alpha=0.5)
    plt.show()
    return


main()
