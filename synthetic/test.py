import itertools

l = [32]+[64] + [128] + [256]
r = [list(perm) for perm in itertools.combinations_with_replacement(l, 3) if sorted(perm) == list(perm)]

print(r)