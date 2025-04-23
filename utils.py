import random
from itertools import combinations
from collections import Counter

def count(iterable, item):
    c = 0
    for i in iterable:
        if i == item:
            c += 1
    return c


def exclusions(iterable1, iterable2):
    combs = set()
    for i in range(0, len(iterable1) + 1):
        for comb in combinations(iterable1, i):
            if len(comb) > 0:
                combs.add(''.join(comb))
    tmp = set(iterable1).difference(set(iterable2))
    if len(tmp) == 0:
        return combs
    for t in tmp:
        e1 = combs.copy()
        for comb in combs:
            if count(iterable1, t) != count(comb, t):
                e1.remove(comb)
                continue
            if t in comb:
                continue
            e1.remove(comb)
        combs = e1

    return combs

# между O(n^2) и O(n * 2^n) по времени
# O(n + k) по памяти
def exclusions_lazy(iterable1, iterable2):
    required = set(iterable1) - set(iterable2)
    required_counts = Counter(c for c in iterable1 if c in required)

    n = len(iterable1)
    iterable1 = tuple(iterable1)

    for r in range(1, n + 1):
        for comb in combinations(range(n), r):
            sub = tuple(iterable1[i] for i in comb)
            sub_counts = Counter(sub)

            if all(sub_counts[sym] == required_counts[sym] for sym in required):
                yield ''.join(sub)


def weighted_choice(weights):
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i
