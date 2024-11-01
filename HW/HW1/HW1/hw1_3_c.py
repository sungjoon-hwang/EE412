import re
import sys
import math
import numpy as np
from os import times

# started time
start = times().elapsed

# Configurations
threshold = 0.1  # Cosine Distance threshold
num_hyperplanes = 10  # Number of hyperplanes


def preprocess_content(line):
    """Extract document ID and clean content."""
    # fill this part

    doc_id, *_ = line.split(" ")
    doc_id = doc_id.strip()

    # • Ignore non-alphabet characters except the white space
    # • Convert all characters to lower case
    content = re.sub(
        r"[^a-zA-Z\s]", "", line.replace(doc_id, "").strip().lower()
    ).split(" ")

    return doc_id, content


def compute_tf(text):
    """
    Compute term frequency (TF) for a document.

    :return: term frequency dictionary
    """
    # Implement term frequency calculation
    # fill this part
    max_occurance = 1

    tf = {}
    for t in text:
        if t in tf:
            tf[t] += 1
            max_occurance = max(max_occurance, tf[t])
        else:
            tf[t] = 1

    return {k: v / max_occurance for k, v in tf.items()}


def compute_idf(docs):
    """
    Compute inverse document frequency (IDF) for each term across all documents.

    :return: inverse document frequency dictionary
    """
    # Implement inverse document frequency calculation
    # fill this part
    freq = {}
    num_docs = len(docs)
    for doc in docs:
        # count 1 time per word in a single document
        for word in set(doc):
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

    idf = {k: math.log2(num_docs / v) for k, v in freq.items()}
    return idf


def compute_tf_idf(tf, idf):
    """
    Compute TF-IDF for a document.

    :return: TF-IDF vector as a dictionary
    """
    # Implement TF-IDF calculation
    # fill this part

    num_words = len(idf)
    idf_extended = {
        w: (idx, val)
        for idx, (w, val) in enumerate(
            sorted(idf.items(), key=lambda x: x[0], reverse=False)
        )
    }

    tf_idf = {}
    for doc_id, tf_doc in tf.items():
        vec = np.zeros(num_words, dtype=float)

        for w, w_tf in tf_doc.items():
            idx, w_idf = idf_extended[w]
            vec[idx] = w_tf * w_idf

        tf_idf[doc_id] = vec

    return tf_idf


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two TF-IDF vectors.

    :return: cosine similarity score (between 0 and 1)
    """
    # Implement cosine similarity calculation
    # fill this part
    return (np.dot(vec1, vec2)) / (np.linalg.norm(vec1, 2) * np.linalg.norm(vec2, 2))


def generate_hyperplanes(num_dimensions, num_planes):
    """
    Generate random hyperplanes for LSH, following the reference's guidance.

    :return: list of hyperplanes
    """
    np.random.seed(0)
    # Create hyperplanes that simulate randomness but are deterministic
    # fill this part
    return np.random.rand(num_planes, num_dimensions) * 2 - 1  # btw -1 ~ 1


def hash_with_hyperplanes(vec, hyperplanes):
    """
    Hash a vector using hyperplanes for LSH.

    :return: hash code (binary string)
    """
    # fill this part
    return "".join(tuple("1" if np.dot(vec, hp) >= 0 else "0" for hp in hyperplanes))


# Step 1: Load the dataset
doc_ids = []
documents = []
with open(sys.argv[1], "r") as f:
    for line in f:
        # Parse and clean the input data
        # fill this part
        doc_id, content = preprocess_content(line)
        doc_ids.append(doc_id)
        documents.append(content)


# Step 2: Compute TF for each document
tf_docs = {}
for i in range(len(doc_ids)):
    doc_id = doc_ids[i]
    tf_doc = compute_tf(documents[i])
    tf_docs[doc_id] = tf_doc


# Step 3: Compute IDF across all documents
idf = compute_idf(docs=documents)


# Step 4: Compute TF-IDF for each document
tf_idf_docs = compute_tf_idf(tf_docs, idf)


# Step 5: Generate hyperplanes for LSH

hyperplanes = generate_hyperplanes(len(idf), num_hyperplanes)


# Step 6: Hash each document using the hyperplanes
# fill this part
buckets = {}
for doc_id, vec in tf_idf_docs.items():
    hash_val = hash_with_hyperplanes(vec, hyperplanes)

    if hash_val in buckets:
        buckets[hash_val].append(doc_id)
    else:
        buckets[hash_val] = [doc_id]


# Step 7: Calculate cosine similarities for document pairs within the same hash bucket
# fill this part

result = []
for _, items in buckets.items():
    for i, doc1 in enumerate(items):
        for j, doc2 in enumerate(items):
            if i < j:
                cs = cosine_similarity(tf_idf_docs[doc1], tf_idf_docs[doc2])
                dist = 1 - cs
                if dist < cs:  # dist
                    result.append((doc1, doc2, cs))

# Step 8: Print the results with similarity scores
for doc1, doc2, similarity in result:
    print(f"{doc1}\t{doc2}\t{similarity:.6f}")


end = times().elapsed
print(f"elapsed time: {round(end-start,2)}s")


### Desired Output ###
# t448    t8535   1.000000
# t8413   t269    1.000000
# t3268   t7998   0.993032
# t1621   t7958   1.000000
# t980    t2023   0.997186
