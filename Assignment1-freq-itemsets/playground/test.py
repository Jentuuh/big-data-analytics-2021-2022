import itertools


val = set(map(frozenset, itertools.combinations(["a", "a", "a", "b"], 2)))
print(list(val))