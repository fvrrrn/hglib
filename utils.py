import random


def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


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


def weighted_choice(weights):
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i
