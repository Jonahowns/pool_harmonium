from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances_chunked
import torch
import numpy as np


import multiprocessing as mp
from functools import partial

# from SetSimilaritySearch import all_pairs, SearchIndex
# import pickle
# import pandas as pd

from pool.utils.alphabet import get_alphabet


def cat_to_seq(categorical_tensor, alphabet="protein"):
    """takes tensor of integers and returns corresponding string using provided alphabet"""
    base_to_id = get_alphabet(alphabet)
    seqs = []
    for i in range(categorical_tensor.shape[0]):
        seq = ""
        for j in range(categorical_tensor.shape[1]):
            seq += base_to_id[categorical_tensor[i][j]]
        seqs.append(seq)
    return seqs


def seq_to_cat(seqs, alphabet="protein"):
    """takes seqs as list of strings and returns a categorical vector"""
    base_to_id = get_alphabet(alphabet)
    return torch.tensor(list(map(lambda x: [base_to_id[y] for y in x], seqs)), dtype=torch.long)

# return matrix that can be graphed as a sequence logo, using sequence_logo in graph_utils
def dataframe_to_matrix(dataframe, alphabet='dna', key='sequence'):
    df = dataframe[[key, "copy_num"]]
    mat_tensor = seq_to_cat(df[key].tolist(), alphabet)

    final_mat = np.full((mat_tensor.shape[1], len(alphabet)), 0.0)
    for j in range(mat_tensor.shape[1]): # v
        for k in range(len(alphabet)):  # q
            final_mat[j, k] = (mat_tensor[:, j] == k).sum()

    final_mat /= mat_tensor.shape[0]
    return final_mat

def cat_to_one_hot(cat_seqs, q):
    """ takes a categorical vector and returns its one hot encoded representation"""
    one_hot = np.zeros((cat_seqs.shape[0], cat_seqs.shape[1]*q))
    for i in range(cat_seqs.shape[0]):
        for j in range(cat_seqs.shape[1]):
            one_hot[i, j*q:(j+1)*q][cat_seqs[i, j]] = 1
    return one_hot


def find_nearest(sequence, dataframe, hamming_threshold=0, alphabet="protein"):
    """Find sequences in dataframe within number of mutations of a given sequence"""

    # categorical vector for query sequence
    cat_query = seq_to_cat([sequence], alphabet=alphabet)

    seq_len = len(sequence)

    # categorical vector for database sequences
    database_seqs = dataframe["sequence"].tolist()

    database_cat = seq_to_cat(database_seqs, alphabet=alphabet)

    dist_matrix = pairwise_distances(cat_query, database_cat, metric="hamming") * seq_len

    seqs_of_interest = dist_matrix[0] <= hamming_threshold

    return dataframe[seqs_of_interest]


def prune_similar_sequences(dataframe, hamming_threshold=0, alphabet="protein"):
    """generate subset of sequences that are at least x mutations away from one another,
    first occurrence is kept so make sure to sort dataframe prior"""
    dataframe.reset_index(drop=True, inplace=True)
    seqs = dataframe["sequence"].tolist()
    index = dataframe.index.tolist()

    cat = seq_to_cat(seqs, alphabet=alphabet)
    X = cat.numpy().astype(np.int8)

    seq_len = len(seqs[0])
    selected_seqs, selected_indices, selected_cat = [], [], []
    total_seqs = len(seqs)
    for i in range(total_seqs):  # len(m1_seqs)
        if i == 0:
            selected_seqs.append(seqs[i])
            selected_indices.append(index[i])
            selected_cat.append(X[i])
        else:
            # number of mutations this sequence is from all sequences in the selected subset
            dist_matrix = pairwise_distances([X[i]], selected_cat, metric="hamming") * seq_len
            if min(dist_matrix[0]) > hamming_threshold:
                selected_seqs.append(seqs[i])
                selected_indices.append(index[i])
                selected_cat.append(X[i])

    print(f"Kept {len(selected_seqs)} of {total_seqs}")

    dataframe = dataframe.iloc[selected_indices, :]
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def prune_similar_sequences_df(df1, df2, hamming_threshold=0, alphabet="protein", return_min_distances=False):
    """generate subset of sequences in df1 that are at least x mutations away from all sequences in df2"""
    df1.reset_index(drop=True, inplace=True)
    df1_seqs = df1["sequence"].tolist()
    df1_index = df1.index.tolist()

    df1_cat = seq_to_cat(df1_seqs, alphabet=alphabet)
    X = df1_cat.numpy().astype(np.int8)

    df2.reset_index(drop=True, inplace=True)
    df2_seqs = df2["sequence"].tolist()
    df2_index = df2.index.tolist()

    df2_cat = seq_to_cat(df2_seqs, alphabet=alphabet)
    Y = df2_cat.numpy().astype(np.int8)

    seq_len = len(df1_seqs[0])

    def reduce_func(D_chunk, start):
        # print(D_chunk)
        return np.asarray(D_chunk).min(1).tolist()

    min_distances_chunked = pairwise_distances_chunked(X, Y, reduce_func=reduce_func, metric="hamming")

    mdists = []
    for n1 in min_distances_chunked:
        mdists += n1

    if return_min_distances:
        return [x * seq_len for x in mdists]
    else:
        keep = np.asarray(mdists)*seq_len > hamming_threshold

        dataframe = df1.iloc[keep, :]

        print(f"Kept {dataframe.index.__len__()} of {df1.index.__len__()} in df1")

        dataframe.reset_index(drop=True, inplace=True)
        return dataframe


# Indexes cannot be modified!
# class LSHIndex:
#     def __init__(self):
#         self.index = None
#
#     def create_index(self, tokens, similarity_threshold=0.1, similarity_function="jaccard"):
#         if self.index is not None:
#             print("Index is already created and cannot be updated")
#         else:
#             self.index = SearchIndex(tokens, similarity_func_name=similarity_function, similarity_threshold=similarity_threshold)
#
#     def save_index(self, index, filename):
#         # open a file, where you ant to store the data
#         file = open(filename, 'wb')
#         # dump information to that file
#         pickle.dump(index, file)
#
#     def load_index(self, filename):
#         # open a file, where you stored the pickled data
#         file = open(filename, 'rb')
#         # dump information to that file
#         self.index = pickle.load(file)
#
#     def query(self, query_token):
#         return self.index.query(query_token)


def tokenize(seq, k=5):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

def create_tokens(seqs, token_function, cpus=1, k=5):
    pool = mp.Pool(processes=cpus)
    return pool.map(partial(token_function, k=k), seqs)

# conver

# def calculate_pair_distances_lsh(tokens, similarity_function="jaccard", similarity_threshold=0.1):
#     """calculate pairwise distances of all sequences in dataframe using lsh scheme and return distance matrix"""
#     pairs = all_pairs(tokens, similarity_func_name=similarity_function, similarity_threshold=similarity_threshold)
#     sim_mat = np.zeros((len(tokens), len(tokens)))
#     sep = [*zip(*pairs)]
#     # make symmetric distance matrix
#     sim_mat[tuple(sep[:-1])] = sep[-1]
#     sim_mat[tuple(sep[-2::-1])] = sep[-1]
#     return 1. / (sim_mat + sim_mat[np.nonzero(sim_mat)].min())
#
# # Multidimensional Scaling -> Allows visualization of high dimensions based off a pairwise metric
# def mds_transform(pairwise_distances, dataframe, random_state=0):
#     from sklearn.manifold import MDS
#     mds = MDS(dissimilarity='precomputed', random_state=random_state)
#     # Get the embeddings
#     X = mds.fit_transform(pairwise_distances)