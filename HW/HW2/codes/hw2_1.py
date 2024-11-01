import sys
import math
import numpy as np
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)


def dist(x, y):
    """
    INPUT: two points x and y
    OUTPUT: the Euclidean distance between two points x and y

    DESCRIPTION: Returns the Euclidean distance between two points.
    """

    return np.linalg.norm((x - y), 2)


def parse_line(line):
    """
    INPUT: one line from input file
    OUTPUT: parsed line with numerical values

    DESCRIPTION: Parses a line to coordinates.
    """
    return np.array(list(map(float, line.split(" "))))


def pick_points(k):
    """
    INPUT: value of k for k-means algorithm
    OUTPUT: the list of initial k centroids.

    DESCRIPTION: Picks the initial cluster centroids for running k-means.
    """

    # read points from file
    with open(sys.argv[1], "r") as f:
        points = np.array([parse_line(line) for line in f])

    # calculate all distances
    N = points.shape[0]
    dist_bulk = np.array(
        [np.linalg.norm(points - points[i], ord=2, axis=1) for i in range(N)]
    )

    indices = [0]
    while len(indices) < k:
        index = dist_bulk[indices, :].min(axis=0).argmax()
        indices.append(index)

    return [points[idx] for idx in indices]


def assign_cluster(centroids, point):
    """
    INPUT: list of centorids and a point
    OUTPUT: a pair of (closest centroid, given point)

    DESCRIPTION: Assigns a point to the closest centroid.
    """

    centroid = min(centroids, key=lambda c: dist(c, point))
    return tuple(centroid), point


def compute_diameter(cluster):
    """
    INPUT: cluster
    OUTPUT: diameter of the given cluster

    DESCRIPTION: Computes the diameter of a cluster.
    """
    N = len(cluster)
    dist_bulk = np.array(
        [np.linalg.norm(cluster - cluster[i], ord=2, axis=1) for i in range(N)]
    )
    diameter = dist_bulk.max()

    return diameter


def kmeans(centroids):
    """
    INPUT: list of centroids
    OUTPUT: average diameter of the clusters

    DESCRIPTION:
    Runs the k-means algorithm and computes the cluster diameters.
    Returns the average diameter of the clusters.

    You may use PySpark things at this function.
    """

    input_rdd = sc.textFile(sys.argv[1])

    points = input_rdd.map(parse_line)
    assigned_points = points.map(lambda p: assign_cluster(centroids, p))

    clusters = assigned_points.groupByKey()
    diameters = clusters.map(lambda x: (compute_diameter(np.array(list(x[1]))), 1))

    # calculating mean
    dsum, cnt = diameters.reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    return dsum / cnt


if __name__ == "__main__":
    """
    This is just an example of the main function.
    """
    k = int(sys.argv[2])
    centroids = pick_points(k)
    average_diameter = kmeans(centroids)
    print(average_diameter)
