import sys
import re
from os import times

# started time
start = times().elapsed

support = 100
top_k = 10

names = dict()
counts = []

# First pass over the data to populate names and counts
with open(sys.argv[1], "r") as f:
    for line in f:
        # Add logic to handle each item here
        # fill this part

        for strid in set(re.split(r"[^\w]+", line.strip())):
            if strid in names:
                counts[names[strid]] += 1
            else:
                names[strid] = len(names)
                counts.append(1)


# Construct frequent items
m = 0
for i in range(len(counts)):
    if counts[i] >= support:
        m += 1
        counts[i] = m
    else:
        counts[i] = 0


# Function to compute triangle index
def get_triangle_idx(i, j, m):
    # Compute the index in the triangle matrix
    # fill this part

    assert i != j, "i and j cannot be the same."
    if i > j:
        i, j = j, i

    # last -1 is for indexing
    return m * (i - 1) - ((i**2 - i) // 2) + j - i - 1


# Initialize the triangular array with zeros
triangular = [0] * (m * (m - 1) // 2)

# Second pass over the data to update the triangular matrix
with open(sys.argv[1], "r") as f:
    for line in f:
        # Add logic to update the triangular matrix here
        # fill this part
        strids = re.split(r"[^\w]+", line.strip())

        # only frequent items
        items = [it for it in map(lambda x: counts[names[x]], strids) if it > 0]

        # candidate pairs
        candidates = {
            (min(i1, i2), max(i1, i2)) for i1 in items for i2 in items if i1 != i2
        }
        for a, b in candidates:
            triangular[get_triangle_idx(a, b, m)] += 1


# Find frequent pairs
freq_pairs = []
num_freq_pairs = 0

# Add logic to find frequent pairs here
# fill this part

iterable = list(enumerate(names.items()))
for i, (strid1, v1) in iterable:
    for j, (strid2, v2) in iterable:
        if i < j and counts[v1] > 0 and counts[v2] > 0:
            cnt = triangular[get_triangle_idx(counts[v1], counts[v2], m)]
            if cnt > support:
                freq_pairs.append((cnt, min(strid1, strid2), max(strid1, strid2)))

num_freq_pairs = len(freq_pairs)

# Sort frequent pairs
freq_pairs = sorted(freq_pairs, key=lambda x: x[0], reverse=True)

# Output results
print(f"Number of frequent items: {m}")
print(f"Number of frequent pairs: {num_freq_pairs}")
print(f"Top-{top_k} frequent pairs:")
for pair in freq_pairs[:top_k]:
    print(pair)


# end time
end = times().elapsed
print(f"elapsed time: {round(end-start,2)}s")


### Desired output #####
# Number of frequent items: 647
# Number of frequent pairs: 1334
# Top-10 frequent pairs:
# (1592, 'DAI62779', 'ELE17451')
# (1412, 'FRO40251', 'SNA80324')
# (1254, 'DAI75645', 'FRO40251')
# (1213, 'FRO40251', 'GRO85051')
# (1139, 'DAI62779', 'GRO73461')
# (1130, 'DAI75645', 'SNA80324')
# (1070, 'DAI62779', 'FRO40251')
# (923, 'DAI62779', 'SNA80324')
# (918, 'DAI62779', 'DAI85309')
# (911, 'ELE32164', 'GRO59710')
