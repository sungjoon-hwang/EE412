import re
import sys
from pyspark import SparkConf, SparkContext
from os import times

# started time
start = times().elapsed

# Initialize Spark configuration and context

conf = SparkConf()
sc = SparkContext(conf=conf)


def get_friends(line):
    """Given a line, returns the user and their friends as a tuple."""
    # fill this part
    user, friends = line.split("\t")

    user = user.strip()
    friends = friends.strip()
    friends = tuple(friends.split(",")) if friends else ()

    return (user, friends)


def get_triplets(user, friends):
    """Given a user and their friends, generate all pairs (triplets) of friends."""
    # Students need to implement the logic for generating triplet candidates.
    # fill this part
    return [
        (user, friends[i], friends[j])
        for i in range(len(friends))
        for j in range(i, len(friends))
    ]


def is_triangle(triplet, friends_map):
    """Check if a triplet forms a triangle by verifying mutual friendship."""
    # Students need to implement the logic to verify if the triplet forms a triangle.
    # fill this part

    usr, a, b = triplet
    return b in friends_map[a] and a in friends_map[b]
    # return (
    #     len([f for f in friends_map[a] if f in (b, c)]) == 2
    #     and len([f for f in friends_map[b] if f in (a, c)]) == 2
    #     and len([f for f in friends_map[c] if f in (a, b)]) == 2
    # )


if len(sys.argv) < 2:
    print("Usage: spark-submit hw1_triangle.py <input_path>")
    sys.exit(1)

input_path = sys.argv[1]

# Read input file as an RDD
input_rdd = sc.textFile(input_path)

# Students should complete the steps to find triangles below:

# Step 1: Get all friends of each user
# user_friends = ...

user_friends = input_rdd.map(get_friends)


# Step 2: Broadcast the user-friends map
# friends_map = ...
# broadcast_friends_map = ...

friends_map = dict(user_friends.collect())
broadcast_friends_map = sc.broadcast(friends_map)


# Step 3: Generate triplets of users who could form triangles
# triplets = ...
triplets = user_friends.flatMap(lambda x: get_triplets(x[0], x[1]))


# Step 4: Filter out non-triangle triplets
# triangles = ...
triangles = triplets.filter(lambda x: is_triangle(x, broadcast_friends_map.value))


# Step 5: Eliminate duplicates by using distinct()
# unique_triangles = ...
unique_triangles = triangles.map(lambda x: tuple(sorted(x))).distinct()


# Step 6: Print the first 10 and last 10 triangles based on lexicographical order
# sorted_triangles = ...
sorted_triangles = sorted(
    unique_triangles.map(lambda x: f"{x[0]}\t{x[1]}\t{x[2]}").collect()
)

print("First 10 triangles:")
for ans in sorted_triangles[:10]:
    print(ans)

print("Last 10 triangles:")
for ans in sorted_triangles[-10:]:
    print(ans)

sc.stop()

# end time
end = times().elapsed
print(f"elapsed time: {round(end-start,2)}s")

###### Desired output #########

# First 10 triangles:
# 0       1       20
# 0       1       5
# 0       10      12
# 0       10      16
# 0       10      30
# 0       12      16
# 0       12      29
# 0       12      3
# 0       12      30
# 0       12      38

# Last 10 triangles:
# 9988    9992    9994
# 9988    9993    9994
# 9989    9990    9991
# 9989    9990    9993
# 9989    9990    9994
# 9989    9993    9994
# 9990    9992    9993
# 9990    9992    9994
# 9990    9993    9994
# 9992    9993    9994
