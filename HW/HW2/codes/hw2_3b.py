import sys

import numpy as np

# Take average of top-k similar user's ratings
topk_users_to_average = 10
# Take average of top-k similar items ratings
topk_items_to_average = 10
# Considering items 1 to 1000
num_items_for_prediction = 1000
# Top-k predictions of items with highest ratings
topk_items = 20  # TODO: change to 5
# Target user's id
target_user_id = 600


def cosine(a, b):
    """
    INPUT: two vectors a and b
    OUTPUT: cosine similarity between a and b

    DESCRIPTION:
    Takes two vectors and returns the cosine similarity.
    """

    return np.nansum(a * b) / np.sqrt(np.nansum(a**2) * np.nansum(b**2))
    # return (np.dot(a, b)) / (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))


def get_matrix(file_name):
    """
    INPUT: file name
    OUTPUT: utility matrix from the file

    DESCRIPTION:
    Reads the utility matrix from the file.
    """
    users = dict()
    users_inv = dict()

    movies = dict()
    movies_inv = dict()

    data = []

    with open(file_name, "r") as f:
        for line in f:
            # <USER ID>,<MOVIE ID>,<RATING>,<TIMESTAMP>
            user_id, movie_id, rating, ts = list(
                map(
                    lambda x, cast: cast(x),
                    line.strip().split(","),
                    [int, int, float, int],
                )
            )

            if user_id not in users:
                idx = len(users)
                users[user_id] = idx
                users_inv[idx] = user_id

            if movie_id not in movies:
                idx = len(movies)
                movies[movie_id] = idx
                movies_inv[idx] = movie_id

            data.append((user_id, movie_id, rating, ts))

    # create utility matrix
    mtx = np.full((len(users), len(movies)), np.nan)

    for user_id, movie_id, rating, ts in data:
        # mtx by mapped ids
        mtx[users[user_id], movies[movie_id]] = rating

    return mtx, users, users_inv, movies, movies_inv


def user_based(umatrix, user_id, movies_inv):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using user-based collaborative
    filtering.
    """

    # apply normalization
    nmatrix = umatrix - np.nanmean(umatrix, axis=1)[:, np.newaxis]

    candidates = []

    for i in range(nmatrix.shape[0]):
        if i != user_id:
            cs = cosine(nmatrix[user_id], nmatrix[i])
            candidates.append((i, cs))

    sim_users, weights = zip(
        *sorted(candidates, key=lambda x: x[1], reverse=True)[:topk_users_to_average]
    )
    weights = np.array(weights)

    numerator = np.nansum(
        umatrix[sim_users, :] * weights[:, np.newaxis],
        axis=0,
    )
    denominator = np.nansum(
        np.where(np.isnan(umatrix[sim_users, :]), 0, 1) * weights[:, np.newaxis],
        axis=0,
    )

    prediction = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )

    items = [
        (movies_inv[i], prediction[i])
        for i, rating in enumerate(prediction)
        if movies_inv[i] <= num_items_for_prediction
    ]
    items = sorted(items, key=lambda x: (x[1], -x[0]), reverse=True)[:topk_items]
    return items


def item_based(umatrix, user_id, movies_inv):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using item-based collaborative
    filtering.
    """

    nmatrix = umatrix - np.nanmean(umatrix, axis=1)[:, np.newaxis]
    nmatrix = np.where(np.isnan(nmatrix), 0, nmatrix)

    # calculate cosine similarity by matrix multiplication to enhance performance
    l2norm = np.linalg.norm(nmatrix, ord=2, axis=0)

    numerator = np.dot(nmatrix.transpose(), nmatrix)
    denominator = l2norm[:, np.newaxis] * l2norm

    cos_sim = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, -np.inf),
        where=denominator != 0,
    )

    # When you find similar items, you consider all movies except ID=1 to ID=1000
    exclude_indices = [
        k for k, v in movies_inv.items() if v <= num_items_for_prediction
    ]
    cos_sim[:, exclude_indices] = -np.inf
    # To ignore calculations involving identical vectors
    np.fill_diagonal(cos_sim, -np.inf)

    # Calculate the index of items with high similarity
    # indices_mtx[i] returns the indices of j similar to `i`.
    indices_mtx = np.argpartition(cos_sim, -topk_items_to_average, axis=1)[
        :, -topk_items_to_average:
    ]

    predict_ratings = np.zeros(umatrix.shape[1], dtype=np.float32)
    for i in range(umatrix.shape[1]):
        indices = indices_mtx[i]
        nix = umatrix[user_id, indices]

        if not np.isnan(nix).all():
            predict_ratings[i] = np.nanmean(nix)

            # complex
            # weights = cos_sim[i, indices]
            #
            # numerator = np.nansum(nix * weights)
            # denominator = np.nansum(np.where(np.isnan(nix), 0, 1) * weights)
            #
            # if denominator != 0:
            #     predict_ratings[i] = numerator / denominator
            # else:
            #     predict_ratings[i] = 0

    items = [
        (movies_inv[i], r)
        for i, r in enumerate(predict_ratings)
        if movies_inv[i] <= num_items_for_prediction
    ]
    items = sorted(items, key=lambda x: (x[1], -x[0]), reverse=True)[:topk_items]
    return items


if __name__ == "__main__":
    """
    This is just an example of the main function.
    """
    umatrix, users, users_inv, movies, movies_inv = get_matrix(sys.argv[1])
    res = user_based(umatrix, users[target_user_id], movies_inv)
    for item, prediction in res:
        print(f"{item}\t{prediction}")
        ...

    res = item_based(umatrix, users[target_user_id], movies_inv)
    for item, prediction in res:
        # print(f"{item}\t{prediction}")
        ...
