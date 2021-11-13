THRESHOLD = 0.4
NUM_HASH_FUNCTIONS = 100
epsilon = 0.00315


def calc_b_and_r(threshold: float):
    for b in range(1, NUM_HASH_FUNCTIONS + 1):
        for r in range(1, NUM_HASH_FUNCTIONS // b + 1):
            if (1 / b)**(1 / r) + epsilon >= THRESHOLD >= (1 / b)**(1 / r) - epsilon:
                return b, r


def main():
    b, r = calc_b_and_r(THRESHOLD)
    print("Found b: ", b)
    print("Found r: ", r)


main()
