import sys
import re
import numpy as np
from os import times

# started time
start = times().elapsed

# Set the random seed for reproducibility
np.random.seed(0)

# Configurations
k = 3  # Length of shingles
threshold = 0.9  # Similarity threshold
b = 6  # Number of bands
r = 20  # Rows per band


def preprocess_content(line):
    """Extract document ID and clean content."""
    # fill this part

    doc_id, *_ = line.strip().split(" ")
    doc_id = doc_id.strip()

    # • Ignore non-alphabet characters except the white space
    # • Convert all characters to lower case
    content = re.sub(r"[^a-zA-Z0-9\s]", "", line.replace(doc_id, "").strip().lower())

    return doc_id, content


def generate_shingles(content, k):
    """Generate k-length shingles from content."""

    # Consider a shingle unit as an alphabetic character
    return [content[i : i + k] for i in range(len(content) - k + 1)]


def is_prime(num):
    if num <= 1:
        return False

    # inspecting until the sqrt(n) is enough.
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False

    return True


def next_prime(n):
    """Find the next prime number greater than or equal to n."""
    # fill this part
    while not is_prime(n):
        n += 1

    return n


# def compute_minhash_signatures(rows, num_docs, h_a, h_b, h_c):
def compute_minhash_signatures(char_mtx, rows, num_docs, h_a, h_b, h_c):
    """Compute MinHash signatures for all documents."""
    # fill this part
    signatures = np.full((rows, num_docs), h_c << 2)

    hash_func = lambda x, i: (h_a[i] * x + h_b[i]) % h_c

    for cmtx_row, (_, names) in enumerate(
        sorted(char_mtx.items(), key=lambda x: x[0], reverse=True)
        # char_mtx.items()
    ):
        for i in range(rows):
            smtx_row = hash_func(cmtx_row, i)
            signatures[i, names] = np.where(
                signatures[i, names] is None or smtx_row < signatures[i, names],
                smtx_row,
                signatures[i, names],
            )

    return signatures


def lsh(signatures, b, r):
    """Perform Locality Sensitive Hashing to find candidate pairs."""
    # fill this part

    """
    by Classum
    When hashing the signature components in each band during the LSH step, 
    you can use the band's signature (a tuple of integers) directly as a key 
    in a hash table or dictionary.
    This leverages built-in hash functions and ensures 
    that documents with identical band signatures are grouped together. 
    You do not need to implement a custom hash function 
    for hashing bands unless you choose to do so.
    """

    candidates = {}
    chunk = 0
    while chunk < b:
        candidates = dict()
        band = signatures[r * chunk : r * (chunk + 1)]

        if band.size == 0:
            break

        for col in range(band.shape[1]):
            val = hash(tuple(band[:, col]))
            if val in candidates:
                candidates[val].append(col)
            else:
                candidates[val] = [col]

        chunk += 1

    return candidates


def get_candidate_pairs(buckets):
    """Extract candidate pairs from LSH buckets."""
    # fill this part

    pairs = set()
    for v in buckets.values():
        if len(v) >= 2:
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    pairs.add((min(v[i], v[j]), max(v[i], v[j])))

    return pairs


def jaccard_similarity(sig1, sig2):
    """Calculate Jaccard similarity between two signatures."""
    return np.mean(sig1 == sig2)


def main(input_file):
    # Step 1: Read and preprocess documents
    # fill this part

    char_mtx = dict()
    doc_names = {}
    doc_inv = {}

    # Step 2: Group shingles by shingle value
    # fill this part

    with open(input_file, "r") as f:
        for line in f:
            doc_id, content = preprocess_content(line)

            if doc_id not in doc_names:
                idx = len(doc_names)
                doc_names[doc_id] = idx
                doc_inv[idx] = doc_id

            for s in generate_shingles(content, k):
                if s in char_mtx:
                    char_mtx[s].append(doc_names[doc_id])
                else:
                    char_mtx[s] = [doc_names[doc_id]]

    num_rows = len(char_mtx)  # n is same as token num

    # Step 3: Generate hash functions
    num_hashes = b * r  # n in textbook
    h_c = next_prime(num_rows)
    h_a = np.random.randint(0, h_c, num_hashes)
    h_b = np.random.randint(0, h_c, num_hashes)

    # Step 4: Compute MinHash signatures
    signatures = compute_minhash_signatures(
        char_mtx, num_hashes, len(doc_names), h_a, h_b, h_c
    )

    # Step 5: Locality Sensitive Hashing (LSH)
    buckets = lsh(signatures, b, r)

    # Step 6: Candidate pair generation
    candidate_pairs = get_candidate_pairs(buckets)

    # Step 7: Verify candidate pairs and output results
    # fill this part
    for p1, p2 in candidate_pairs:
        js = jaccard_similarity(signatures[:, p1], signatures[:, p2])
        if js >= threshold:
            print(f"{doc_inv[p1]}\t{doc_inv[p2]}\t{js:.6f}")

    # end time
    end = times().elapsed
    print(f"elapsed time: {round(end-start,2)}s")

    # My Result
    # t448    t8535   1.000000
    # t1621   t7958   1.000000
    # t8413   t269    1.000000
    # t3268   t7998   0.991667
    # t980    t2023   0.991667


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])

# Desired output

# t7998   t3268   0.991667
# t8535   t448    1.000000
# t980    t2023   0.991667
# t269    t8413   1.000000
# t7958   t1621   1.000000
